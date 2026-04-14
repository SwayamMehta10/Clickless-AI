"""User preference model -- persists per-user preferences as JSON.

Updated after each confirmed cart:
  - preferred brands
  - allergens / dietary constraints
  - budget range
  - purchase history (product names + counts)
  - rejected items
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.api.product_schema import CartItem, Product
from src.utils.paths import PREFERENCES_DIR

logger = logging.getLogger(__name__)

_PREF_DIR = PREFERENCES_DIR
_PREF_DIR.mkdir(parents=True, exist_ok=True)


class UserPreferences:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._path = _PREF_DIR / f"{user_id}.json"
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {
            "preferred_brands": [],
            "allergens": [],
            "dietary_flags": [],
            "budget": None,
            "purchase_history": {},  # product_name -> count
            "rejected_items": [],
        }

    def save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    # --- Accessors ---
    @property
    def preferred_brands(self) -> List[str]:
        return self._data.get("preferred_brands", [])

    @property
    def allergens(self) -> List[str]:
        return self._data.get("allergens", [])

    @property
    def dietary_flags(self) -> List[str]:
        return self._data.get("dietary_flags", [])

    @property
    def budget(self) -> Optional[float]:
        return self._data.get("budget")

    @property
    def purchase_history(self) -> Dict[str, int]:
        return self._data.get("purchase_history", {})

    # --- Mutators ---
    def update_preferences(
        self,
        confirmed_cart: List[CartItem],
        rejected_items: Optional[List[str]] = None,
        new_dietary_flags: Optional[List[str]] = None,
        new_budget: Optional[float] = None,
    ) -> None:
        """Update preferences after cart confirmation."""
        # Purchase history
        for item in confirmed_cart:
            name = item.product.name or ""
            if name:
                self._data["purchase_history"][name] = \
                    self._data["purchase_history"].get(name, 0) + item.quantity

            # Infer brand preference from repeated purchases
            if item.product.brand:
                brand = item.product.brand
                history_count = self._data["purchase_history"].get(name, 0)
                if history_count >= 2 and brand not in self._data["preferred_brands"]:
                    self._data["preferred_brands"].append(brand)

        # Rejected items
        if rejected_items:
            for item_name in rejected_items:
                if item_name not in self._data["rejected_items"]:
                    self._data["rejected_items"].append(item_name)

        # Dietary flags
        if new_dietary_flags:
            for flag in new_dietary_flags:
                if flag not in self._data["dietary_flags"]:
                    self._data["dietary_flags"].append(flag)

        # Budget
        if new_budget is not None:
            self._data["budget"] = new_budget

        self.save()
        logger.info("Updated preferences for user %s", self.user_id)

    def get_top_products(self, n: int = 10) -> List[str]:
        """Return top-n most frequently purchased product names."""
        history = self.purchase_history
        sorted_items = sorted(history.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_items[:n]]

    def to_context_str(self) -> str:
        """Return a concise string summary for use in LLM prompts."""
        parts = []
        if self.preferred_brands:
            parts.append(f"Prefers: {', '.join(self.preferred_brands)}")
        if self.dietary_flags:
            parts.append(f"Dietary: {', '.join(self.dietary_flags)}")
        if self.allergens:
            parts.append(f"Allergic to: {', '.join(self.allergens)}")
        if self.budget:
            parts.append(f"Budget: ${self.budget:.0f}")
        top = self.get_top_products(5)
        if top:
            parts.append(f"Usually buys: {', '.join(top)}")
        return " | ".join(parts) if parts else "No preferences set yet."


# Module-level cache
_cache: Dict[str, UserPreferences] = {}


def get_preferences(user_id: str) -> UserPreferences:
    if user_id not in _cache:
        _cache[user_id] = UserPreferences(user_id)
    return _cache[user_id]


def update_preferences(
    user_id: str,
    confirmed_cart: List[CartItem],
    rejected_items: Optional[List[str]] = None,
    new_dietary_flags: Optional[List[str]] = None,
    new_budget: Optional[float] = None,
) -> None:
    prefs = get_preferences(user_id)
    prefs.update_preferences(confirmed_cart, rejected_items, new_dietary_flags, new_budget)
