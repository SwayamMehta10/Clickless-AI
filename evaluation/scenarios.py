"""Test scenarios for ablation study.

15 scenarios split into three categories:
  - Weekly grocery list (5)
  - Dietary-restricted meal prep (5)
  - Budget-capped bulk (5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Scenario:
    id: str
    category: str
    utterance: str
    expected_items: List[str]
    constraints: dict = field(default_factory=dict)
    max_budget: Optional[float] = None


WEEKLY = [
    Scenario(
        id="W1", category="weekly",
        utterance="I need milk, eggs, bread, chicken, and broccoli under $30",
        expected_items=["milk", "eggs", "bread", "chicken", "broccoli"],
        max_budget=30.0,
    ),
    Scenario(
        id="W2", category="weekly",
        utterance="Weekly groceries: apples, yogurt, pasta, ground beef, carrots",
        expected_items=["apples", "yogurt", "pasta", "ground beef", "carrots"],
    ),
    Scenario(
        id="W3", category="weekly",
        utterance="I need staples: rice, beans, onions, garlic, olive oil",
        expected_items=["rice", "beans", "onions", "garlic", "olive oil"],
    ),
    Scenario(
        id="W4", category="weekly",
        utterance="Breakfast items: cereal, oatmeal, orange juice, blueberries, bagels",
        expected_items=["cereal", "oatmeal", "orange juice", "blueberries", "bagels"],
    ),
    Scenario(
        id="W5", category="weekly",
        utterance="Dinner ingredients for a family of 4: salmon, potatoes, asparagus, lemon",
        expected_items=["salmon", "potatoes", "asparagus", "lemon"],
    ),
]

DIETARY = [
    Scenario(
        id="D1", category="dietary",
        utterance="Gluten-free pasta, low-sodium tomato sauce, organic ground turkey",
        expected_items=["pasta", "tomato sauce", "ground turkey"],
        constraints={"dietary": ["gluten-free", "low-sodium", "organic"]},
    ),
    Scenario(
        id="D2", category="dietary",
        utterance="Vegan protein sources -- tofu, lentils, quinoa, almond milk",
        expected_items=["tofu", "lentils", "quinoa", "almond milk"],
        constraints={"dietary": ["vegan"]},
    ),
    Scenario(
        id="D3", category="dietary",
        utterance="Diabetic-friendly snacks with no added sugar",
        expected_items=["nuts", "seeds", "dark chocolate"],
        constraints={"dietary": ["no-sugar"]},
    ),
    Scenario(
        id="D4", category="dietary",
        utterance="Nut-free school lunch items: bread, cheese, turkey, apples, pretzels",
        expected_items=["bread", "cheese", "turkey", "apples", "pretzels"],
        constraints={"dietary": ["nut-free"]},
    ),
    Scenario(
        id="D5", category="dietary",
        utterance="Dairy-free breakfast: oat milk, coconut yogurt, gluten-free granola",
        expected_items=["oat milk", "coconut yogurt", "granola"],
        constraints={"dietary": ["dairy-free", "gluten-free"]},
    ),
]

BULK = [
    Scenario(
        id="B1", category="bulk",
        utterance="Snacks for 20 people, $50 budget, no nuts",
        expected_items=["chips", "crackers", "fruit", "cookies"],
        constraints={"dietary": ["nut-free"], "people": 20},
        max_budget=50.0,
    ),
    Scenario(
        id="B2", category="bulk",
        utterance="Party drinks for 15 people under $40: soda, juice, water",
        expected_items=["soda", "juice", "water"],
        constraints={"people": 15},
        max_budget=40.0,
    ),
    Scenario(
        id="B3", category="bulk",
        utterance="Bulk pantry restock under $75: rice, pasta, canned tomatoes, flour, sugar",
        expected_items=["rice", "pasta", "canned tomatoes", "flour", "sugar"],
        max_budget=75.0,
    ),
    Scenario(
        id="B4", category="bulk",
        utterance="Office breakfast for 10 people under $30: bagels, cream cheese, coffee, fruit",
        expected_items=["bagels", "cream cheese", "coffee", "fruit"],
        constraints={"people": 10},
        max_budget=30.0,
    ),
    Scenario(
        id="B5", category="bulk",
        utterance="BBQ supplies for 12 under $60: burgers, buns, lettuce, tomato, condiments",
        expected_items=["burgers", "buns", "lettuce", "tomato", "condiments"],
        constraints={"people": 12},
        max_budget=60.0,
    ),
]

ALL_SCENARIOS: List[Scenario] = WEEKLY + DIETARY + BULK
