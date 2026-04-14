"""Heuristic NLU used for the lightweight demo path."""

from __future__ import annotations

import re
from typing import Tuple

from src.nlu.dialogue_state import Slots

DIETARY_PATTERNS = {
    "organic": [r"\borganic\b"],
    "gluten-free": [r"\bgluten[ -]?free\b"],
    "vegan": [r"\bvegan\b"],
    "low-sodium": [r"\blow[ -]?sodium\b"],
    "nut-free": [r"\bnut[ -]?free\b", r"\bno nuts\b"],
    "dairy-free": [r"\bdairy[ -]?free\b"],
}

_PRICE_PATTERNS = [
    r"(?:under|below|less than|max(?:imum)?|budget(?: is)?|for)\s*\$?\s*(\d+(?:\.\d+)?)",
    r"\$\s*(\d+(?:\.\d+)?)",
]
_QUANTITY_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(gallons?|gallon|oz|ounces?|lb|lbs|pounds?|packs?|pack|bottles?|bags?|dozen)\b"
)
_FILLER_PHRASES = [
    "i need",
    "i want",
    "show me",
    "find me",
    "find",
    "looking for",
    "get me",
    "please",
    "can you",
    "help me",
    "for the week",
]
_STOPWORDS = {
    "a", "an", "and", "any", "cart", "for", "from", "i", "in", "is", "it",
    "me", "my", "of", "one", "please", "some", "that", "the", "this", "to",
    "under", "with",
}


def classify_intent(message: str, conversation_history: str = "") -> Tuple[str, float]:
    """Classify user intent with lightweight rules for demo reliability."""
    text = message.lower().strip()

    if any(phrase in text for phrase in ("checkout", "check out", "place order", "complete order")):
        return "checkout", 0.98
    if any(phrase in text for phrase in ("remove", "delete", "take out")) and "cart" in text:
        return "remove_from_cart", 0.93
    if any(phrase in text for phrase in ("add ", "put ", "add the", "put the")) and "cart" in text:
        return "add_to_cart", 0.95
    if any(phrase in text for phrase in ("recommend", "suggest", "what should i get", "what goes with")):
        return "get_recommendation", 0.86
    if any(word in text for word in ("hello", "hi", "hey", "thanks", "thank you")) and len(text.split()) <= 5:
        return "chit_chat", 0.9

    slots = extract_slots(message, conversation_history)
    has_constraint = (
        slots.max_price is not None
        or bool(slots.dietary_flags)
        or slots.quantity is not None
    )

    if has_constraint and not slots.item:
        return "set_constraint", 0.83
    return "search_product", 0.84


def extract_slots(message: str, conversation_history: str = "") -> Slots:
    """Extract common shopping slots from a short grocery request."""
    text = message.strip()
    lowered = text.lower()

    dietary_flags = []
    for flag, patterns in DIETARY_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            dietary_flags.append(flag)

    max_price = None
    for pattern in _PRICE_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            max_price = float(match.group(1))
            break

    quantity = None
    unit = None
    quantity_match = _QUANTITY_UNIT_RE.search(lowered)
    if quantity_match:
        quantity = float(quantity_match.group(1))
        raw_unit = quantity_match.group(2).lower()
        unit = raw_unit[:-1] if raw_unit.endswith("s") and raw_unit != "lbs" else raw_unit

    brand_preference = None
    brand_match = re.search(r"\b([a-z0-9&' -]+?)\s+brand\b", lowered)
    if brand_match:
        brand_preference = brand_match.group(1).strip().title()

    item = _extract_item(lowered)
    return Slots(
        item=item,
        quantity=quantity,
        unit=unit,
        max_price=max_price,
        dietary_flags=dietary_flags,
        brand_preference=brand_preference,
    )


def _extract_item(lowered: str) -> str | None:
    cleaned = lowered

    for pattern in _PRICE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = _QUANTITY_UNIT_RE.sub(" ", cleaned)

    for patterns in DIETARY_PATTERNS.values():
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned)

    for phrase in _FILLER_PHRASES:
        cleaned = cleaned.replace(phrase, " ")

    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if " and " in cleaned:
        cleaned = cleaned.split(" and ", 1)[0].strip()

    tokens = [token for token in cleaned.split() if token not in _STOPWORDS and not token.isdigit()]
    if not tokens:
        return None

    disallowed = {"checkout", "check", "out", "add", "remove", "recommend", "suggest"}
    tokens = [token for token in tokens if token not in disallowed]
    if not tokens:
        return None

    return " ".join(tokens)
