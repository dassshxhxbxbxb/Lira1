"""FSM states used across the questionnaire."""
from aiogram.fsm.state import State, StatesGroup


class Onboarding(StatesGroup):
    # Step 1: basic
    name = State()
    birth_year = State()
    city = State()
    flow_code_choice = State()
    flow_code_input = State()
    cycle_length = State()
    period_length = State()
    last_period_date = State()

    # Step 2: hygiene
    pads = State()
    tampons = State()
    other_hygiene = State()
    flow_heaviness = State()

    # Step 3: allergies
    allergies = State()
    sensitive_skin = State()
    allergy_notes = State()

    # Step 4: lifestyle
    diet = State()
    goal = State()
    joys = State()
    novelty = State()
    dislikes = State()

    # Step 5: deep preferences
    favorite_season = State()
    calming = State()
    occupation = State()
    hobbies = State()

    # Step 6: address
    address_country = State()
    address_city = State()
    address_street = State()
    address_building = State()
    address_apartment = State()
    address_postal = State()
    address_phone = State()

    # Step 7: tariff + payment
    tariff = State()
    waiting_payment = State()


class EditProfile(StatesGroup):
    choosing_step = State()
