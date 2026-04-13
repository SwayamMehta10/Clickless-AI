"""Canonical Product dataclass used across the entire pipeline."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class NutriScore(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    UNKNOWN = "unknown"


class NovaGroup(int, Enum):
    UNPROCESSED = 1
    CULINARY = 2
    PROCESSED = 3
    ULTRA_PROCESSED = 4


class Product(BaseModel):
    """Canonical product representation used across NLU, API, KG, ranking, and UI layers."""

    instacart_id: str
    name: str
    brand: Optional[str] = None
    price: Optional[float] = None
    availability: bool = True
    platform: str = "instacart"
    nutriscore: NutriScore = NutriScore.UNKNOWN
    nova_group: Optional[NovaGroup] = None
    allergens: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    image_url: Optional[str] = None
    # Additional enrichment fields
    ingredients: Optional[str] = None
    serving_size: Optional[str] = None
    energy_kcal: Optional[float] = None
    fat_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    carbohydrates_g: Optional[float] = None
    sugars_g: Optional[float] = None
    fiber_g: Optional[float] = None
    protein_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    # Instacart-specific
    aisle: Optional[str] = None
    department: Optional[str] = None
    reorder_rate: Optional[float] = None

    class Config:
        use_enum_values = True

    def short_description(self) -> str:
        parts = [self.name]
        if self.brand:
            parts.append(f"by {self.brand}")
        if self.price is not None:
            parts.append(f"${self.price:.2f}")
        if self.nutriscore != NutriScore.UNKNOWN:
            parts.append(f"Nutri-Score {self.nutriscore}")
        return " | ".join(parts)


class RankedProduct(BaseModel):
    """Product with ranking metadata for display."""

    product: Product
    score: float
    rank: int
    score_breakdown: dict = Field(default_factory=dict)
    copurchase_suggestions: List[str] = Field(default_factory=list)
    kg_context: Optional[str] = None


class CartItem(BaseModel):
    """A product in the user's cart."""

    product: Product
    quantity: int = 1
    note: Optional[str] = None

    @property
    def line_total(self) -> Optional[float]:
        if self.product.price is not None:
            return self.product.price * self.quantity
        return None
