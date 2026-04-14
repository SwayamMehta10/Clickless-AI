from __future__ import annotations

from src.nlu.demo_parser import classify_intent, extract_slots


def test_demo_intent_checkout():
    intent, confidence = classify_intent("Let's checkout")
    assert intent == "checkout"
    assert confidence > 0.9


def test_demo_slots_extract_budget_and_dietary():
    slots = extract_slots("I need gluten-free bread under $5")
    assert slots.item == "bread"
    assert slots.max_price == 5.0
    assert "gluten-free" in slots.dietary_flags
