"""Slot filling using Mistral 7B via Ollama.

Extracts: item, quantity, unit, max_price, dietary_flags, brand_preference
"""

from __future__ import annotations

import logging

from src.llm import ollama_client as llm
from src.nlu import demo_parser
from src.nlu.dialogue_state import Slots
from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA_HINT = """\
{
  "item": "<string or null>",
  "quantity": <number or null>,
  "unit": "<string or null>",
  "max_price": <number or null>,
  "dietary_flags": ["<string>", ...],
  "brand_preference": "<string or null>"
}"""

_SYSTEM_PROMPT = f"""\
You are a slot extractor for a grocery shopping assistant.
Given a user message (and optional conversation context), extract:
- item: the grocery product name
- quantity: numeric amount (null if not specified)
- unit: unit of measure like "gallon", "oz", "lb", "pack" (null if not specified)
- max_price: maximum price in USD (null if not specified)
- dietary_flags: list of dietary requirements like ["organic", "gluten-free", "low-sodium", "vegan"]
- brand_preference: preferred brand name (null if not specified)

Return ONLY valid JSON matching this schema:
{_SCHEMA_HINT}
"""

_FEW_SHOT_PAIRS = [
    (
        "I need 2 gallons of organic whole milk under $6",
        '{"item": "whole milk", "quantity": 2, "unit": "gallon", "max_price": 6.0, "dietary_flags": ["organic"], "brand_preference": null}',
    ),
    (
        "Add 3 packs of gluten-free pasta, Barilla brand",
        '{"item": "pasta", "quantity": 3, "unit": "pack", "max_price": null, "dietary_flags": ["gluten-free"], "brand_preference": "Barilla"}',
    ),
    (
        "I want some bananas",
        '{"item": "bananas", "quantity": null, "unit": null, "max_price": null, "dietary_flags": [], "brand_preference": null}',
    ),
    (
        "low sodium chicken broth, 32 oz, under $3",
        '{"item": "chicken broth", "quantity": 32, "unit": "oz", "max_price": 3.0, "dietary_flags": ["low-sodium"], "brand_preference": null}',
    ),
]


def extract_slots(user_message: str, conversation_history: str = "") -> Slots:
    """Extract slots from user message. Returns a Slots instance."""
    app_cfg = get_settings().get("app", {})
    if app_cfg.get("demo_mode", False):
        return demo_parser.extract_slots(user_message, conversation_history)

    # Build few-shot prompt
    few_shot_text = "\n".join(
        f"User: {u}\nSlots: {s}" for u, s in _FEW_SHOT_PAIRS
    )

    if conversation_history.strip():
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Examples:\n{few_shot_text}\n\n"
            f"Conversation so far:\n{conversation_history}\n\n"
            f"User: {user_message}\nSlots:"
        )
    else:
        prompt = (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Examples:\n{few_shot_text}\n\n"
            f"User: {user_message}\nSlots:"
        )

    try:
        data = llm.generate_json(prompt=prompt, role="nlu")

        # Normalise dietary_flags to a list
        flags = data.get("dietary_flags", [])
        if isinstance(flags, str):
            flags = [flags]

        return Slots(
            item=data.get("item"),
            quantity=data.get("quantity"),
            unit=data.get("unit"),
            max_price=data.get("max_price"),
            dietary_flags=flags,
            brand_preference=data.get("brand_preference"),
        )

    except (ValueError, KeyError) as exc:
        logger.error("Slot extraction failed: %s", exc)
        return demo_parser.extract_slots(user_message, conversation_history)


def merge_slots(existing: Slots, new: Slots) -> Slots:
    """Merge new slot values into existing, keeping non-null existing values unless overridden."""
    merged = existing.model_copy()
    for field in ["item", "quantity", "unit", "max_price", "brand_preference"]:
        new_val = getattr(new, field)
        if new_val is not None:
            setattr(merged, field, new_val)
    if new.dietary_flags:
        merged.dietary_flags = list(set(merged.dietary_flags + new.dietary_flags))
    return merged
