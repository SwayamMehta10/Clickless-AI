"""Pydantic DialogueState -- tracks conversation across turns."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.api.product_schema import CartItem


class Slots(BaseModel):
    item: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    max_price: Optional[float] = None
    dietary_flags: List[str] = Field(default_factory=list)
    brand_preference: Optional[str] = None
    # Allow arbitrary extra slots without failing validation
    extras: Dict[str, Any] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return all(v is None or v == [] for v in [
            self.item, self.quantity, self.unit, self.max_price,
            self.dietary_flags, self.brand_preference,
        ])


class TurnRecord(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class DialogueState(BaseModel):
    """Full conversation state passed through the LangGraph pipeline."""

    current_intent: Optional[str] = None
    slots: Slots = Field(default_factory=Slots)
    history: List[TurnRecord] = Field(default_factory=list)
    cart: List[CartItem] = Field(default_factory=list)
    active_constraints: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    zip_code: Optional[str] = None
    budget: Optional[float] = None
    dietary_preferences: List[str] = Field(default_factory=list)

    # Max history turns kept in context
    max_history: int = 10

    def add_turn(self, role: str, content: str) -> None:
        self.history.append(TurnRecord(role=role, content=content))
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history_text(self) -> str:
        return "\n".join(f"{t.role.upper()}: {t.content}" for t in self.history[-self.max_history:])

    def reset_slots(self) -> None:
        self.slots = Slots()
        self.current_intent = None

    def add_to_cart(self, item: CartItem) -> None:
        # Update quantity if product already in cart
        for existing in self.cart:
            if existing.product.instacart_id == item.product.instacart_id:
                existing.quantity += item.quantity
                return
        self.cart.append(item)

    def remove_from_cart(self, product_id: str) -> bool:
        before = len(self.cart)
        self.cart = [i for i in self.cart if i.product.instacart_id != product_id]
        return len(self.cart) < before

    @property
    def cart_total(self) -> Optional[float]:
        totals = [i.line_total for i in self.cart if i.line_total is not None]
        return sum(totals) if totals else None
