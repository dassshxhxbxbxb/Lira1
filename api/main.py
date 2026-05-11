"""FastAPI app exposing the Lira ↔ FlowCare integration endpoints.

Endpoints
---------

* ``POST /v1/activate`` — legacy activation-code redemption (kept as
  fallback for users who have not paired via Telegram).
* ``GET  /v1/subscription?cycle_code=...`` — auto-sync lookup by the
  cycle-sync code already stored on the user's profile.
* ``POST /v1/pair/init`` — issue a fresh pairing token. The Lira app
  opens ``t.me/<bot>?start=link_<token>`` in Telegram; the bot's
  ``/start link_<token>`` handler binds the token to the calling
  Telegram user. The app then polls ``GET /v1/pair/<token>`` to learn
  who claimed the token and what their active subscription is.
"""
from __future__ import annotations

import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bot.config import get_settings
from bot.db import engine, session_scope
from bot.models import Base, User
from bot.services.codes import redeem_code
from bot.services.pairing import (
    create_pair_token,
    find_pair_token,
    is_expired,
)
from bot.services.forecasts import ForecastEntry, replace_user_forecast
from bot.services.subscriptions import (
    canonicalise_cycle_code,
    find_active_subscription_by_cycle_code,
    get_active_subscription,
)

log = logging.getLogger("flowcare-api")
logging.basicConfig(level=logging.INFO)


# Shared aiogram Bot handle so the API can send notifications (e.g. the
# "Прогноз получен" confirmation after the app uploads the forecast)
# through the same Telegram session that the polling worker uses.
_shared_bot: object | None = None


async def _run_bot_polling() -> None:
    """Run the aiogram long-polling bot inside the FastAPI process.

    Co-locating the bot and the API in the same process means they share
    the SQLite database on the persistent volume — pairing tokens written
    by the bot's ``/start link_<token>`` handler are immediately visible
    to ``GET /v1/pair/<token>`` calls. Skipped when ``BOT_TOKEN`` is
    empty (e.g. local API dev or first Fly deploy before secrets are
    configured).
    """
    global _shared_bot
    settings = get_settings()
    if not settings.bot_token:
        log.info("BOT_TOKEN unset — bot polling not started")
        return
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties

    from bot.handlers import router as root_router

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )
    _shared_bot = bot
    dp = Dispatcher()
    dp.include_router(root_router)
    log.info("Starting Telegram bot polling…")
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        _shared_bot = None
        await bot.session.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    settings = get_settings()
    if settings.bot_token:
        # Catalog seeding is bot-only (used to assemble subscription
        # boxes); skip it for api-only deploys to keep memory low.
        from bot.services.catalog import seed_catalog

        async with session_scope() as session:
            await seed_catalog(session)
        bot_task: asyncio.Task[None] | None = asyncio.create_task(
            _run_bot_polling()
        )
    else:
        bot_task = None
    try:
        yield
    finally:
        if bot_task is not None:
            bot_task.cancel()
            try:
                await bot_task
            except (asyncio.CancelledError, Exception):
                pass


app = FastAPI(
    title="FlowCare Activation API",
    version="1.1.0",
    description=(
        "Validates activation codes issued by the FlowCare Telegram bot, "
        "auto-syncs subscriptions by cycle-code, and powers the "
        "Telegram-pairing flow that replaces the activation code with a "
        "single deep-link tap."
    ),
    lifespan=lifespan,
)

# CORS — the Lira web bundle (served on devinapps.com) POSTs to /v1/lira/*
# from the browser, which is a cross-origin request. Open it up to all
# origins: the endpoints are notification-only and the data shipped is
# self-reported (no auth tokens / no secret data to protect server-side).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------------------------------------------------------------------- #
# /v1/activate — activation-code fallback                                  #
# ---------------------------------------------------------------------- #


class ActivateIn(BaseModel):
    code: str = Field(..., min_length=4, max_length=16)
    device_id: str | None = Field(default=None, max_length=128)


class ActivateOut(BaseModel):
    valid: bool
    tariff: str | None = None
    expires: str | None = None
    redeemed_at: str | None = None


@app.post("/v1/activate", response_model=ActivateOut)
async def activate(body: ActivateIn) -> ActivateOut:
    async with session_scope() as session:
        result = await redeem_code(session, body.code, device_id=body.device_id)
    if result is None:
        return ActivateOut(valid=False)
    code, sub = result
    return ActivateOut(
        valid=True,
        tariff=sub.tariff.value,
        expires=sub.expires_at.date().isoformat(),
        redeemed_at=(code.redeemed_at.isoformat() if code.redeemed_at else None),
    )


# ---------------------------------------------------------------------- #
# /v1/subscription?cycle_code=... — auto-sync via cycle-code               #
# ---------------------------------------------------------------------- #


class SubscriptionOut(BaseModel):
    valid: bool
    tariff: str | None = None
    expires: str | None = None


@app.get("/v1/subscription", response_model=SubscriptionOut)
async def subscription_by_cycle_code(
    cycle_code: str = Query(..., min_length=4, max_length=16),
) -> SubscriptionOut:
    canonical = canonicalise_cycle_code(cycle_code)
    if canonical is None:
        return SubscriptionOut(valid=False)
    async with session_scope() as session:
        sub = await find_active_subscription_by_cycle_code(session, canonical)
    if sub is None:
        return SubscriptionOut(valid=False)
    return SubscriptionOut(
        valid=True,
        tariff=sub.tariff.value,
        expires=sub.expires_at.date().isoformat(),
    )


# ---------------------------------------------------------------------- #
# /v1/pair/* — Telegram deep-link pairing                                  #
# ---------------------------------------------------------------------- #


class PairInitOut(BaseModel):
    token: str
    expires_at: str
    deep_link: str


class PairStatusOut(BaseModel):
    paired: bool
    expired: bool = False
    tariff: str | None = None
    expires: str | None = None
    telegram_username: str | None = None


def _bot_deep_link(token: str) -> str:
    """Build the Telegram deep-link the app should open."""
    settings = get_settings()
    username = (getattr(settings, "bot_username", None) or "").lstrip("@")
    if not username:
        # Fall back to a hard-coded slot the app overrides in app.json.
        username = "lowerBsk24_bot"
    return f"https://t.me/{username}?start=link_{token}"


@app.post("/v1/pair/init", response_model=PairInitOut)
async def pair_init() -> PairInitOut:
    async with session_scope() as session:
        tok = await create_pair_token(session)
        expires = tok.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return PairInitOut(
        token=tok.token,
        expires_at=expires.isoformat(),
        deep_link=_bot_deep_link(tok.token),
    )


@app.get("/v1/pair/{token}", response_model=PairStatusOut)
async def pair_status(token: str) -> PairStatusOut:
    if len(token) > 64:
        raise HTTPException(status_code=400, detail="token too long")
    async with session_scope() as session:
        row = await find_pair_token(session, token)
        if row is None:
            return PairStatusOut(paired=False, expired=False)
        if is_expired(row) and row.claimed_user_id is None:
            return PairStatusOut(paired=False, expired=True)
        if row.claimed_user_id is None:
            return PairStatusOut(paired=False, expired=False)
        user = await session.get(User, row.claimed_user_id)
        sub = await get_active_subscription(session, user) if user else None
    out = PairStatusOut(
        paired=True,
        expired=False,
        telegram_username=user.username if user else None,
    )
    if sub is not None:
        out.tariff = sub.tariff.value
        out.expires = sub.expires_at.date().isoformat()
    return out


# ---------------------------------------------------------------------- #
# /v1/pair/{token}/forecast — push 3-month cycle forecast to the bot      #
# ---------------------------------------------------------------------- #


class ForecastEntryIn(BaseModel):
    """One projected cycle. All dates are ``YYYY-MM-DD`` strings."""

    cycle_start: str
    period_end: str
    ovulation: str
    fertile_start: str
    fertile_end: str


class ForecastIn(BaseModel):
    entries: list[ForecastEntryIn] = Field(default_factory=list, max_length=12)


class ForecastOut(BaseModel):
    ok: bool
    stored: int = 0


def _parse_date(value: str):
    from datetime import date as _date

    try:
        return _date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid date: {value}"
        ) from exc


@app.post("/v1/pair/{token}/forecast", response_model=ForecastOut)
async def pair_forecast(token: str, body: ForecastIn) -> ForecastOut:
    if len(token) > 64:
        raise HTTPException(status_code=400, detail="token too long")
    if not body.entries:
        raise HTTPException(status_code=422, detail="forecast is empty")
    parsed: list[ForecastEntry] = []
    for raw in body.entries:
        parsed.append(
            ForecastEntry(
                cycle_start=_parse_date(raw.cycle_start),
                period_end=_parse_date(raw.period_end),
                ovulation=_parse_date(raw.ovulation),
                fertile_start=_parse_date(raw.fertile_start),
                fertile_end=_parse_date(raw.fertile_end),
            )
        )
    async with session_scope() as session:
        row = await find_pair_token(session, token)
        if row is None or row.claimed_user_id is None:
            raise HTTPException(status_code=404, detail="pair token not claimed")
        if is_expired(row):
            # Expired-but-claimed is fine: the user already linked, the
            # token just can't be re-claimed by anyone else.
            pass
        user = await session.get(User, row.claimed_user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="user missing")
        stored = await replace_user_forecast(session, user, parsed)
        chat_id = user.telegram_id
    if _shared_bot is not None and chat_id is not None and parsed:
        first = parsed[0]
        try:
            await _shared_bot.send_message(  # type: ignore[attr-defined]
                chat_id=chat_id,
                text=(
                    "Получил твой прогноз цикла на 3 месяца 🌸\n\n"
                    f"Ближайшие месячные: <b>{first.cycle_start:%d.%m.%Y}</b>\n"
                    f"Овуляция: <b>{first.ovulation:%d.%m.%Y}</b>\n\n"
                    "Бокс приедет к началу следующих месячных. "
                    "Прогноз обновится автоматически каждый раз, когда "
                    "ты заново привязываешь Telegram в приложении."
                ),
                parse_mode="HTML",
            )
        except Exception:  # pragma: no cover — best-effort notification
            log.exception("forecast confirmation send failed")
        # Also notify the operator (admin) so they see the updated dates.
        settings = get_settings()
        if settings.admin_chat_id:
            lines = [
                "🔄 <b>Обновился прогноз цикла</b>",
                f"Клиент: <code>{chat_id}</code>",
                "",
            ]
            for i, e in enumerate(parsed, 1):
                lines.append(
                    f"• №{i}: <b>{e.cycle_start:%d.%m}</b>—"
                    f"<b>{e.period_end:%d.%m}</b>"
                    f" • овуляция <b>{e.ovulation:%d.%m}</b>"
                )
            try:
                await _shared_bot.send_message(  # type: ignore[attr-defined]
                    chat_id=settings.admin_chat_id,
                    text="\n".join(lines),
                    parse_mode="HTML",
                )
            except Exception:  # pragma: no cover
                log.exception("admin forecast notification failed")
    return ForecastOut(ok=True, stored=stored)


# ---------------------------------------------------------------------- #
# /v1/lira/* — web-app → bot notification bridge                            #
# ---------------------------------------------------------------------- #
#
# The Lira web bundle posts to these endpoints from the browser when the
# user finishes the in-app questionnaire (onboarding), completes a payment
# (subscription), or her cycle forecast shifts by more than ±2 days
# (cycle-update). Each endpoint formats the payload as an HTML message and
# sends it to ``settings.admin_chat_id`` — i.e. the owner's Telegram chat.

import json as _json  # noqa: E402  — local alias to avoid conflict
from datetime import date as _date, timedelta as _timedelta  # noqa: E402


class _LiraOnboardingIn(BaseModel):
    """Strict-but-forgiving model: the web bundle may add extra keys; we
    accept and forward whatever is sent, but never fail on schema drift."""

    model_config = {"extra": "allow"}


class _LiraGenericOut(BaseModel):
    ok: bool = True


def _format_iso_date(raw: str | None) -> str:
    if not raw:
        return "—"
    try:
        return _date.fromisoformat(raw[:10]).strftime("%d.%m.%Y")
    except Exception:
        return raw


def _extract_period_starts(logs: dict | None) -> list[_date]:
    """Pick out the FIRST day of each period episode from the cycle logs.

    ``logs`` is a dict of ISO-date string → log entry. Entries with
    ``flow`` in {"light","medium","heavy","spotting"} are part of a
    bleeding day. Consecutive bleeding days are merged into one episode
    and its first day is the cycle start.
    """
    if not logs or not isinstance(logs, dict):
        return []
    bleeding_levels = {"light", "medium", "heavy", "spotting"}
    bleeding_days: list[_date] = []
    for k, v in logs.items():
        if not isinstance(v, dict):
            continue
        flow = v.get("flow") or v.get("flowLevel")
        if isinstance(flow, str) and flow.lower() in bleeding_levels:
            try:
                bleeding_days.append(_date.fromisoformat(k[:10]))
            except Exception:
                continue
    bleeding_days.sort()
    starts: list[_date] = []
    for d in bleeding_days:
        if not starts or (d - starts[-1]).days > 2:
            starts.append(d)
        elif (d - starts[-1]).days <= 2 and (d - starts[-1]).days >= 0:
            # contiguous-ish: skip, same episode
            continue
    return starts


def _build_forecast(
    last_start: _date | None,
    avg_cycle: int,
    avg_period: int,
    months: int = 3,
) -> str:
    if last_start is None or not avg_cycle:
        return ""
    lines = [f"🩸 <b>Прогноз цикла на {months} мес.</b>"]
    luteal = 14
    cur = last_start
    for i in range(1, months + 1):
        nxt = cur + _timedelta(days=avg_cycle)
        end = cur + _timedelta(days=max(0, avg_period - 1))
        ovu = nxt - _timedelta(days=luteal)
        lines.append(
            f"• №{i}: <b>{cur:%d.%m}</b>—<b>{end:%d.%m}</b>"
            f" • овуляция <b>{ovu:%d.%m}</b>"
        )
        cur = nxt
    return "\n".join(lines)


_TIER_LABELS = {
    "premium": "✨ Lira Premium",
    "basic": "📦 Твой ритм (Basic Box)",
    "vip": "📦 Полная симфония (VIP Box)",
}


def _safe(v) -> str:
    from html import escape as _esc

    if v is None or v == "":
        return "—"
    return _esc(str(v))


def _format_subscription_message(p: dict) -> str:
    """Render the after-payment owner digest from the web payload."""
    tier = (p.get("tier") or "").lower()
    tier_title = _TIER_LABELS.get(tier, p.get("tierTitle") or tier or "—")
    price = p.get("price")
    paid_at = p.get("paidAt") or ""
    order_id = p.get("orderId") or "—"
    card_last4 = p.get("cardLast4")

    profile = p.get("profile") or {}
    settings_block = p.get("settings") or {}
    shipping = p.get("shippingAddress") or {}
    box = p.get("boxProfile") or {}
    survey = p.get("surveyData") or {}
    logs = p.get("logs") or {}

    avg_cycle = int(
        settings_block.get("averageCycleLength")
        or box.get("cycleLength")
        or survey.get("cycleLength")
        or 28
    )
    avg_period = int(
        settings_block.get("averagePeriodLength")
        or box.get("periodLength")
        or survey.get("periodLength")
        or 5
    )

    starts = _extract_period_starts(logs)
    last_start = starts[-1] if starts else None
    # If logs don't contain bleeding days, fall back to box/survey-provided date
    if last_start is None:
        for k in ("lastPeriodStart", "lastPeriod", "periodStartDate"):
            raw = box.get(k) or survey.get(k)
            if raw:
                try:
                    last_start = _date.fromisoformat(str(raw)[:10])
                    break
                except Exception:
                    continue

    lines: list[str] = []
    lines.append("💰 <b>Новая оплата + анкета (web)</b>")
    lines.append("")
    lines.append("<b>Платёж</b>")
    lines.append(f"• Тариф: <b>{_safe(tier_title)}</b>")
    if price is not None:
        lines.append(f"• Цена: <b>{_safe(price)} ₽</b>")
    lines.append(f"• Оплачено: <b>{_format_iso_date(paid_at[:10])}</b>")
    if card_last4:
        lines.append(f"• Карта: <code>•••• {_safe(card_last4)}</code>")
    lines.append(f"• Order ID: <code>{_safe(order_id)}</code>")
    lines.append("")

    lines.append("<b>Профиль</b>")
    lines.append(f"• Имя: {_safe(profile.get('name'))}")
    if profile.get("birthdate"):
        lines.append(f"• Дата рождения: {_format_iso_date(profile.get('birthdate'))}")
    if profile.get("phone"):
        lines.append(f"• Телефон: {_safe(profile.get('phone'))}")
    lines.append("")

    if shipping:
        lines.append("<b>Адрес доставки</b>")
        lines.append(f"• Получатель: {_safe(shipping.get('recipient'))}")
        lines.append(f"• Страна: {_safe(shipping.get('country'))}")
        lines.append(f"• Город: {_safe(shipping.get('city'))}")
        lines.append(f"• Улица: {_safe(shipping.get('street'))}")
        bld = shipping.get("building")
        apt = shipping.get("apartment") or shipping.get("flat")
        lines.append(f"• Дом / кв.: {_safe(bld)} / {_safe(apt)}")
        lines.append(f"• Индекс: {_safe(shipping.get('postal') or shipping.get('postcode'))}")
        lines.append(f"• Телефон: {_safe(shipping.get('phone'))}")
        lines.append("")

    # boxProfile / surveyData — flatten any string/number/list answers
    answers: dict = {}
    for src in (box, survey):
        if isinstance(src, dict):
            for k, v in src.items():
                if k.startswith("_"):
                    continue
                if k in {"lastPeriodStart", "cycleLength", "periodLength"}:
                    continue
                answers.setdefault(k, v)
    if answers:
        lines.append("<b>Анкета</b>")
        for k, v in answers.items():
            if isinstance(v, (list, tuple)):
                v = ", ".join(_safe(x) for x in v) if v else "—"
            elif isinstance(v, dict):
                v = _json.dumps(v, ensure_ascii=False)
            lines.append(f"• {_safe(k)}: {_safe(v)}")
        lines.append("")

    lines.append("<b>Цикл</b>")
    lines.append(f"• Средний цикл: <b>{avg_cycle} дн.</b>")
    lines.append(f"• Длина месячных: <b>{avg_period} дн.</b>")
    if last_start:
        lines.append(f"• Последние месячные: <b>{last_start:%d.%m.%Y}</b>")
    lines.append("")

    forecast = _build_forecast(last_start, avg_cycle, avg_period, months=3)
    if forecast:
        lines.append(forecast)
        lines.append("")

    lines.append(
        f"<i>{_format_iso_date(paid_at[:10])} • оплата прошла, ждём сборку бокса</i>"
    )
    return "\n".join(lines)


async def _send_admin(text: str) -> bool:
    """Send a Telegram message to the operator chat. Returns False if no
    bot is available or admin chat is unset."""
    bot = _shared_bot
    settings = get_settings()
    if bot is None or not settings.admin_chat_id:
        log.warning(
            "admin send skipped: shared_bot=%s admin_chat_id=%s",
            bool(bot),
            settings.admin_chat_id,
        )
        return False
    try:
        await bot.send_message(  # type: ignore[attr-defined]
            chat_id=settings.admin_chat_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
        return True
    except Exception:
        log.exception("admin send failed")
        return False


@app.post("/v1/lira/onboarding")
async def lira_onboarding(body: _LiraOnboardingIn) -> _LiraGenericOut:
    """Web bundle calls this when the user FINISHES the questionnaire but
    before paying. Currently a no-op: we wait for the payment hook to
    fire the single combined digest. We log it for visibility."""
    payload = body.model_dump()
    log.info(
        "lira-onboarding: device=%s tz=%s",
        payload.get("device_id"),
        payload.get("timezone"),
    )
    return _LiraGenericOut(ok=True)


@app.post("/v1/lira/subscription")
async def lira_subscription(body: _LiraOnboardingIn) -> _LiraGenericOut:
    """Web bundle calls this when the user has PAID. Format and forward
    the full digest (payment + survey + cycle forecast) to the owner's
    Telegram chat."""
    payload = body.model_dump()
    log.info(
        "lira-subscription: order=%s tier=%s",
        payload.get("orderId"),
        payload.get("tier"),
    )
    text = _format_subscription_message(payload)
    ok = await _send_admin(text)
    return _LiraGenericOut(ok=ok)


@app.post("/v1/lira/cycle-update")
async def lira_cycle_update(body: _LiraOnboardingIn) -> _LiraGenericOut:
    """Web bundle calls this when the user's forecasted next-period date
    has shifted by more than ±2 days (e.g. she logged a period earlier
    than expected). Send the operator a short delta message."""
    payload = body.model_dump()
    name = payload.get("name") or "клиент"
    prev_raw = payload.get("previous_next_period")
    logs = payload.get("logs") or {}
    settings_block = payload.get("settings") or {}
    avg_cycle = int(settings_block.get("averageCycleLength") or 28)
    avg_period = int(settings_block.get("averagePeriodLength") or 5)
    starts = _extract_period_starts(logs)
    last_start = starts[-1] if starts else None
    new_forecast = _build_forecast(last_start, avg_cycle, avg_period, months=3)
    lines = [
        "🔄 <b>Цикл пересчитан</b>",
        f"Клиент: <b>{_safe(name)}</b>",
    ]
    if prev_raw:
        lines.append(f"Был прогноз следующих М: <b>{_format_iso_date(prev_raw)}</b>")
    if last_start:
        next_start = last_start + _timedelta(days=avg_cycle)
        lines.append(f"Стал прогноз: <b>{next_start:%d.%m.%Y}</b>")
    lines.append("")
    if new_forecast:
        lines.append(new_forecast)
    ok = await _send_admin("\n".join(lines))
    return _LiraGenericOut(ok=ok)


@app.get("/v1/lira/status")
async def lira_status() -> dict:
    """Stub for the bundled chat UI's status check. Reports that the
    chat backend is offline so the UI degrades gracefully."""
    return {"online": False, "models": []}


@app.post("/v1/lira/chat")
async def lira_chat(body: _LiraOnboardingIn) -> dict:
    """Stub for the bundled chat UI. Returns a polite refusal so the
    UI shows the message rather than spinning forever."""
    return {
        "reply": (
            "Ассистент Lira временно недоступен. Я могу ответить тебе "
            "напрямую в Telegram-боте @lowerBsk24_bot — там есть оператор."
        ),
        "online": False,
    }


# ---------------------------------------------------------------------- #
# Health                                                                   #
# ---------------------------------------------------------------------- #


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app", host=settings.api_host, port=settings.api_port, reload=False
    )
