"""
=============================================================
  AgreeCulture - Rule-Based Chatbot Engine
=============================================================
  This chatbot:
  1. Detects intent from user message (greeting, how-to, fertilizer, etc.)
  2. Extracts the crop name from the message
  3. Fetches relevant data from the JSON knowledge base
  4. Returns a structured, helpful response
=============================================================
"""

import json
import re
import os


# ─────────────────────────────────────────────────────────────────────────────
#  Load the Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

def load_knowledge_base():
    """Load and return the crop knowledge base from JSON file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path = os.path.join(base_dir, "data", "knowledge_base.json")
    with open(kb_path, "r") as f:
        return json.load(f)


KB = load_knowledge_base()
CROPS = KB["crops"]
FERTILIZERS = KB["fertilizers"]


# ─────────────────────────────────────────────────────────────────────────────
#  Intent Detection
# ─────────────────────────────────────────────────────────────────────────────
# We map keywords in the user's message to predefined intents.
# This is the "rule" in rule-based chatbot.

INTENT_PATTERNS = {
    "greeting":    r"\b(hello|hi|hey|namaste|good morning|good evening|howdy)\b",
    "farewell":    r"\b(bye|goodbye|see you|exit|quit|thanks|thank you)\b",
    "how_to_grow": r"\b(how to grow|how do i grow|how to cultivate|cultivation|grow|farming of|planting)\b",
    "fertilizer":  r"\b(fertilizer|fertiliser|manure|nutrient|npk|urea|dap|compost|feed)\b",
    "water":       r"\b(water|irrigation|moisture|watering|rainfall|drought)\b",
    "season":      r"\b(season|when to grow|when to plant|best time|kharif|rabi)\b",
    "soil":        r"\b(soil|land|clay|loam|sandy|ph|field)\b",
    "disease":     r"\b(disease|pest|insect|fungus|blight|rot|worm|spray|protect)\b",
    "yield":       r"\b(yield|production|output|harvest|how much|profit)\b",
    "all_crops":   r"\b(list|all crops|which crops|what crops|available crops)\b",
    "help":        r"\b(help|what can you do|features|how does this work)\b",
}


def detect_intent(message: str) -> str:
    """
    Detect the user's intent by matching keywords in their message.
    Returns the intent string (e.g., 'how_to_grow', 'fertilizer').
    """
    message_lower = message.lower()

    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, message_lower):
            return intent

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
#  Crop Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_crop(message: str):
    """
    Scan the user's message for any known crop name.
    Returns the matched crop key (lowercase) or None if no match found.
    """
    message_lower = message.lower()

    # Build a dictionary of alternative names → canonical key
    crop_aliases = {
        "rice": "rice", "paddy": "rice", "dhaan": "rice",
        "wheat": "wheat", "gehun": "wheat",
        "maize": "maize", "corn": "maize", "makka": "maize",
        "cotton": "cotton", "kapas": "cotton",
        "sugarcane": "sugarcane", "sugar cane": "sugarcane", "ganna": "sugarcane",
        "soybean": "soybean", "soya": "soybean", "soy": "soybean",
        "potato": "potato", "aloo": "potato", "potatoes": "potato",
        "tomato": "tomato", "tamatar": "tomato", "tomatoes": "tomato",
        "groundnut": "groundnut", "peanut": "groundnut", "moongfali": "groundnut",
        "mango": "mango", "aam": "mango",
    }

    for alias, canonical in crop_aliases.items():
        # Use word-boundary matching to avoid partial matches
        if re.search(r'\b' + re.escape(alias) + r'\b', message_lower):
            return canonical

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Response Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_how_to_grow_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    steps = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(crop["steps"])])
    return (
        f"🌱 <b>How to Grow {crop['name']}</b>\n\n"
        f"📅 <b>Season:</b> {crop['season']}\n"
        f"🏔️ <b>Soil Type:</b> {crop['soil_type']}\n\n"
        f"📋 <b>Step-by-Step Guide:</b>\n{steps}\n\n"
        f"💡 <b>Pro Tip:</b> {crop['tips']}"
    )


def build_fertilizer_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    return (
        f"🧪 <b>Fertilizer Guide for {crop['name']}</b>\n\n"
        f"✅ <b>Recommended:</b> {crop['fertilizer']}\n\n"
        f"💡 <b>Tip:</b> Always apply fertilizers in split doses and ensure soil moisture "
        f"before application. Over-fertilization can damage crops."
    )


def build_water_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    return (
        f"💧 <b>Water Requirements for {crop['name']}</b>\n\n"
        f"🌧️ {crop['water_requirement']}\n\n"
        f"💡 <b>Tip:</b> {crop['tips']}"
    )


def build_season_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    return (
        f"📅 <b>Best Season for {crop['name']}</b>\n\n"
        f"🗓️ {crop['season']}\n\n"
        f"🌱 <b>Soil:</b> {crop['soil_type']}"
    )


def build_soil_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    return (
        f"🏔️ <b>Soil Requirements for {crop['name']}</b>\n\n"
        f"{crop['soil_type']}\n\n"
        f"💡 <b>Tip:</b> {crop['tips']}"
    )


def build_disease_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    diseases = ", ".join(crop["common_diseases"])
    return (
        f"🦠 <b>Common Diseases of {crop['name']}</b>\n\n"
        f"⚠️ <b>Watch out for:</b> {diseases}\n\n"
        f"💡 <b>Prevention:</b> Use certified seeds, maintain proper spacing for airflow, "
        f"and apply recommended fungicides/pesticides at early signs of infection.\n"
        f"Always consult your local agricultural extension officer for region-specific advice."
    )


def build_yield_response(crop_key: str) -> str:
    crop = CROPS[crop_key]
    return (
        f"📦 <b>Expected Yield for {crop['name']}</b>\n\n"
        f"🌾 <b>Yield:</b> {crop['yield']}\n\n"
        f"💡 <b>Note:</b> Actual yield depends on soil quality, weather, variety selection, "
        f"and management practices."
    )


def build_all_crops_response() -> str:
    crop_list = ", ".join([c.capitalize() for c in CROPS.keys()])
    return (
        f"🌾 <b>Crops in My Knowledge Base</b>\n\n"
        f"{crop_list}\n\n"
        f"Ask me about any of these crops! For example:\n"
        f"• 'How to grow wheat?'\n"
        f"• 'Best fertilizer for rice?'\n"
        f"• 'What season is good for potato?'"
    )


def build_help_response() -> str:
    return (
        "👋 <b>Hi! I'm AgreeCulture Assistant</b>\n\n"
        "Here's what you can ask me:\n\n"
        "🌱 <b>Growing guide:</b> 'How to grow rice?'\n"
        "🧪 <b>Fertilizers:</b> 'Best fertilizer for wheat?'\n"
        "💧 <b>Water needs:</b> 'Water requirements for maize?'\n"
        "📅 <b>Seasons:</b> 'When to grow potato?'\n"
        "🏔️ <b>Soil type:</b> 'What soil does mango need?'\n"
        "🦠 <b>Diseases:</b> 'Common diseases of tomato?'\n"
        "📦 <b>Yield:</b> 'How much yield from sugarcane?'\n"
        "🌾 <b>Crop list:</b> 'What crops do you know?'\n\n"
        "I know about: " + ", ".join([c.capitalize() for c in CROPS.keys()])
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Main Chat Function
# ─────────────────────────────────────────────────────────────────────────────

def get_response(user_message: str) -> str:
    """
    Main entry point for the chatbot.
    Takes user message string, returns a formatted response string.
    """
    if not user_message or not user_message.strip():
        return "Please type a message. Ask me anything about farming! 🌾"

    intent = detect_intent(user_message)
    crop = extract_crop(user_message)

    # ── Handle intents that don't need a crop ────────────────────────────────
    if intent == "greeting":
        return (
            "🌾 <b>Namaste! Welcome to AgreeCulture Assistant!</b>\n\n"
            "I'm here to help you with farming advice. Ask me about:\n"
            "• How to grow any crop\n"
            "• Fertilizer recommendations\n"
            "• Water requirements\n"
            "• Crop diseases\n\n"
            "Type <b>'help'</b> to see all I can do! 😊"
        )

    if intent == "farewell":
        return "👋 Thank you for using AgreeCulture! Happy farming! 🌱🌾"

    if intent == "help":
        return build_help_response()

    if intent == "all_crops":
        return build_all_crops_response()

    # ── Handle intents that need a crop ─────────────────────────────────────
    if crop is None:
        # Intent detected but no crop found — ask for clarification
        if intent != "unknown":
            return (
                f"🤔 I understand you're asking about <b>{intent.replace('_', ' ')}</b>, "
                f"but I couldn't identify the crop name.\n\n"
                f"Please mention a crop! For example:\n"
                f"• 'How to grow <b>wheat</b>?'\n"
                f"• 'Fertilizer for <b>rice</b>?'\n\n"
                f"Known crops: {', '.join([c.capitalize() for c in CROPS.keys()])}"
            )
        # Completely unknown message
        return (
            "🌾 I'm not sure I understood that.\n\n"
            "You can ask me things like:\n"
            "• 'How to grow tomato?'\n"
            "• 'Best fertilizer for wheat?'\n"
            "• 'Water needs of maize?'\n\n"
            "Type <b>'help'</b> to see all available topics."
        )

    # Crop found — dispatch to the right response builder
    if intent == "how_to_grow":
        return build_how_to_grow_response(crop)
    elif intent == "fertilizer":
        return build_fertilizer_response(crop)
    elif intent == "water":
        return build_water_response(crop)
    elif intent == "season":
        return build_season_response(crop)
    elif intent == "soil":
        return build_soil_response(crop)
    elif intent == "disease":
        return build_disease_response(crop)
    elif intent == "yield":
        return build_yield_response(crop)
    else:
        # Crop detected but intent unclear → give a full overview
        c = CROPS[crop]
        return (
            f"🌾 <b>{c['name']} - Quick Overview</b>\n\n"
            f"📅 <b>Season:</b> {c['season']}\n"
            f"🏔️ <b>Soil:</b> {c['soil_type']}\n"
            f"💧 <b>Water:</b> {c['water_requirement']}\n"
            f"🧪 <b>Fertilizer:</b> {c['fertilizer']}\n"
            f"📦 <b>Yield:</b> {c['yield']}\n\n"
            f"Ask me something more specific!\n"
            f"• 'How to grow {crop}?'\n"
            f"• 'Fertilizer for {crop}?'\n"
            f"• 'Diseases of {crop}?'"
        )
