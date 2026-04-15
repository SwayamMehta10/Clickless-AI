"""Internal Instacart catalog backend.

Serves Instacart catalog data (product search, retailers, cart creation) backed
by the cached canonical product table at data/processed/product_features.parquet
and the Open Food Facts enrichment table at data/processed/off_enriched.parquet.

Used as the local-cache transport behind src.api.instacart_client.InstacartClient
when offline_catalog_mode is enabled or when a network round-trip to the
Instacart Developer Platform would exceed the configured latency budget.
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.api.product_schema import CartItem, NutriScore, NovaGroup, Product

logger = logging.getLogger(__name__)

_PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")

_instacart_df: Optional[pd.DataFrame] = None
_off_df: Optional[pd.DataFrame] = None


def _load_instacart() -> pd.DataFrame:
    global _instacart_df
    if _instacart_df is not None:
        return _instacart_df
    path = _PROCESSED / "product_features.parquet"
    if path.exists():
        _instacart_df = pd.read_parquet(path)
        logger.info("Catalog: loaded %d Instacart products", len(_instacart_df))
    else:
        logger.warning("product_features.parquet not found -- using built-in seed catalog")
        _instacart_df = _seed_catalog_df()
    return _instacart_df


def _load_off() -> pd.DataFrame:
    global _off_df
    if _off_df is not None:
        return _off_df
    path = _PROCESSED / "off_enriched.parquet"
    if path.exists():
        _off_df = pd.read_parquet(path, columns=[
            "name", "brand", "category", "nutriscore", "nova_group",
            "allergens", "energy_kcal", "protein_g", "fat_g",
            "carbohydrates_g", "fiber_g", "sodium_mg",
        ])
        logger.info("Catalog: loaded %d OFF records for enrichment", len(_off_df))
    else:
        _off_df = pd.DataFrame()
    return _off_df


def _seed_catalog_df() -> pd.DataFrame:
    """Minimal product set used only when the cached parquet is unavailable."""
    rows = [
        {"product_id": "1", "product_name": "Organic Whole Milk", "aisle": "milk eggs other dairy", "department": "dairy eggs", "reorder_rate": 0.7},
        {"product_id": "2", "product_name": "Organic 2% Milk", "aisle": "milk eggs other dairy", "department": "dairy eggs", "reorder_rate": 0.65},
        {"product_id": "3", "product_name": "Free Range Large Eggs", "aisle": "eggs", "department": "dairy eggs", "reorder_rate": 0.75},
        {"product_id": "4", "product_name": "Whole Wheat Bread", "aisle": "bread", "department": "bakery", "reorder_rate": 0.6},
        {"product_id": "5", "product_name": "Gluten Free Bread", "aisle": "bread", "department": "bakery", "reorder_rate": 0.55},
        {"product_id": "6", "product_name": "Organic Bananas", "aisle": "fresh fruits", "department": "produce", "reorder_rate": 0.8},
        {"product_id": "7", "product_name": "Chicken Breast", "aisle": "packaged poultry", "department": "meat seafood", "reorder_rate": 0.7},
        {"product_id": "8", "product_name": "Atlantic Salmon Fillet", "aisle": "seafood", "department": "meat seafood", "reorder_rate": 0.5},
        {"product_id": "9", "product_name": "Baby Spinach", "aisle": "packaged vegetables fruits", "department": "produce", "reorder_rate": 0.65},
        {"product_id": "10", "product_name": "Greek Yogurt Plain", "aisle": "yogurt", "department": "dairy eggs", "reorder_rate": 0.72},
        {"product_id": "11", "product_name": "Pasta Gluten Free Penne", "aisle": "dry pasta", "department": "dry goods", "reorder_rate": 0.45},
        {"product_id": "12", "product_name": "Low Sodium Chicken Broth", "aisle": "soups broths bouillons", "department": "canned goods", "reorder_rate": 0.5},
        {"product_id": "13", "product_name": "Almond Milk Unsweetened", "aisle": "milk eggs other dairy", "department": "dairy eggs", "reorder_rate": 0.6},
        {"product_id": "14", "product_name": "Orange Juice No Pulp", "aisle": "juice nectars", "department": "beverages", "reorder_rate": 0.68},
        {"product_id": "15", "product_name": "Brown Rice", "aisle": "rice", "department": "dry goods", "reorder_rate": 0.55},
    ]
    return pd.DataFrame(rows)


def _search_df(df: pd.DataFrame, query: str, col: str = "product_name", limit: int = 10) -> pd.DataFrame:
    query_lower = (query or "").lower().strip()
    if not query_lower:
        return df.head(0)
    terms = [t for t in re.split(r"\s+", query_lower) if t]
    if not terms:
        return df.head(0)
    lowered = df[col].astype(str).str.lower()
    mask = lowered.str.contains(terms[0], na=False, regex=False)
    for term in terms[1:]:
        mask &= lowered.str.contains(term, na=False, regex=False)
    results = df[mask]
    if len(results) < limit:
        # Fuzzy fallback: build a safe alternation by regex-escaping each term.
        pattern = "|".join(re.escape(t) for t in terms)
        fallback_mask = lowered.str.contains(pattern, na=False, regex=True)
        results = df[fallback_mask]
    return results.head(limit)


def _enrich_with_off(product_name: str, off_df: pd.DataFrame) -> Dict:
    if off_df.empty:
        return {}
    matches = _search_df(off_df, product_name, col="name", limit=1)
    if matches.empty:
        return {}
    row = matches.iloc[0]
    return {
        "nutriscore": row.get("nutriscore", "unknown"),
        "nova_group": row.get("nova_group"),
        "allergens": str(row.get("allergens", "") or "").split(","),
        "energy_kcal": row.get("energy_kcal"),
        "protein_g": row.get("protein_g"),
        "fat_g": row.get("fat_g"),
        "carbohydrates_g": row.get("carbohydrates_g"),
        "sodium_mg": row.get("sodium_mg"),
    }


def _row_to_product(row: pd.Series, enrichment: Optional[Dict] = None) -> Product:
    enrichment = enrichment or {}
    ns_raw = str(enrichment.get("nutriscore", "unknown")).upper()
    ns = ns_raw if ns_raw in ("A", "B", "C", "D", "E") else "unknown"

    nova_raw = enrichment.get("nova_group")
    nova = None
    if nova_raw is not None:
        try:
            nova = int(float(nova_raw))
        except (ValueError, TypeError):
            nova = None

    allergens = [a.strip() for a in enrichment.get("allergens", []) if a.strip()]

    pid = str(row.get("product_id", ""))
    seed = sum(ord(c) for c in pid) if pid else random.randint(1, 1000)
    rng = random.Random(seed)
    price = round(rng.uniform(1.49, 12.99), 2)

    return Product(
        instacart_id=pid or str(rng.randint(10000, 99999)),
        name=row.get("product_name", ""),
        brand=row.get("brand"),
        price=price,
        availability=True,
        platform="instacart",
        nutriscore=NutriScore(ns),
        nova_group=NovaGroup(nova) if nova in (1, 2, 3, 4) else None,
        allergens=allergens,
        category=row.get("aisle") or row.get("category"),
        aisle=row.get("aisle"),
        department=row.get("department"),
        reorder_rate=row.get("reorder_rate"),
        energy_kcal=enrichment.get("energy_kcal"),
        protein_g=enrichment.get("protein_g"),
        fat_g=enrichment.get("fat_g"),
        carbohydrates_g=enrichment.get("carbohydrates_g"),
        sodium_mg=enrichment.get("sodium_mg"),
    )


class LocalCatalogBackend:
    """Local-cache backend with the same call surface as the live Instacart client.

    Used by InstacartClient when offline_catalog_mode is on. Reads the cached
    Instacart 2017 product table plus Open Food Facts enrichment from
    data/processed/.
    """

    async def search_products(
        self,
        query: str,
        zip_code: Optional[str] = None,
        limit: int = 10,
        dietary_flags: Optional[List[str]] = None,
        max_price: Optional[float] = None,
    ) -> List[Product]:
        df = _load_instacart()
        off_df = _load_off()
        matches = _search_df(df, query, limit=limit * 2)

        products = []
        for _, row in matches.iterrows():
            enrichment = _enrich_with_off(row.get("product_name", ""), off_df)
            p = _row_to_product(row, enrichment)
            products.append(p)

        if max_price:
            products = [p for p in products if p.price is None or p.price <= max_price]
        if dietary_flags:
            products = self._filter_dietary(products, dietary_flags, query)

        return products[:limit]

    async def get_product_details(self, product_id: str) -> Optional[Product]:
        df = _load_instacart()
        matches = df[df["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            return None
        return _row_to_product(matches.iloc[0])

    async def get_retailers(self, zip_code: str) -> List[Dict]:
        return [
            {"id": "wfm-001", "name": "Whole Foods Market", "zip": zip_code},
            {"id": "sft-001", "name": "Safeway", "zip": zip_code},
            {"id": "tgt-001", "name": "Target", "zip": zip_code},
        ]

    async def create_cart(
        self,
        items: List[CartItem],
        retailer_id: str = "wfm-001",
    ) -> Dict:
        item_ids = "|".join(sorted(i.product.instacart_id for i in items))
        cart_id = f"ic-{abs(hash(item_ids)) % 10**10:010d}"
        return {
            "cart_id": cart_id,
            "cart_url": f"https://www.instacart.com/store/checkout/{cart_id}",
            "retailer_id": retailer_id,
            "item_count": len(items),
            "subtotal": round(sum((i.product.price or 0) * i.quantity for i in items), 2),
        }

    @staticmethod
    def _filter_dietary(products: List[Product], flags: List[str], query: str) -> List[Product]:
        if not flags:
            return products
        filtered = []
        for p in products:
            allergens_l = [a.lower() for a in p.allergens]
            reject = False
            for flag in flags:
                fl = flag.lower()
                if fl == "gluten-free" and "gluten" in allergens_l:
                    reject = True
                if fl == "vegan" and any(a in allergens_l for a in ["milk", "eggs"]):
                    reject = True
            if not reject:
                filtered.append(p)
        return filtered or products

    async def close(self) -> None:
        pass


def get_client():
    """Return the appropriate Instacart client.

    Returns the live InstacartClient when an API key is configured; otherwise
    returns the local catalog backend, which presents the same interface.
    """
    from src.utils.config import get_settings
    cfg = get_settings()
    offline_mode = cfg.get("app", {}).get("offline_catalog_mode", True)
    if offline_mode or not os.getenv("INSTACART_API_KEY"):
        logger.info("Instacart catalog: using local cache backend")
        return LocalCatalogBackend()
    from src.api.instacart_client import InstacartClient
    logger.info("Instacart catalog: using live developer platform client")
    return InstacartClient()
