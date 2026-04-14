"""Preprocess Open Food Facts CSV into off_enriched.parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "raw" / "openfoodfacts" / "products.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

KEEP_COLS = [
    "code",
    "product_name",
    "brands",
    "categories_en",
    "ingredients_text",
    "allergens",
    "nutriscore_grade",
    "nova_group",
    "serving_size",
    "energy-kcal_100g",
    "fat_100g",
    "saturated-fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "sodium_100g",
    "image_url",
]

RENAME = {
    "code": "barcode",
    "product_name": "name",
    "brands": "brand",
    "categories_en": "category",
    "ingredients_text": "ingredients",
    "nutriscore_grade": "nutriscore",
    "energy-kcal_100g": "energy_kcal",
    "fat_100g": "fat_g",
    "saturated-fat_100g": "saturated_fat_g",
    "carbohydrates_100g": "carbohydrates_g",
    "sugars_100g": "sugars_g",
    "fiber_100g": "fiber_g",
    "proteins_100g": "protein_g",
    "sodium_100g": "sodium_mg",
}


if __name__ == "__main__":
    print("Loading Open Food Facts CSV (this may take a minute)...")
    df = pd.read_csv(
        RAW,
        sep="\t",
        usecols=[c for c in KEEP_COLS if c in pd.read_csv(RAW, sep="\t", nrows=0).columns],
        low_memory=False,
    )

    # Keep only rows with a product name
    df = df[df["product_name"].notna() & (df["product_name"].str.strip() != "")]

    # Rename
    existing = {k: v for k, v in RENAME.items() if k in df.columns}
    df = df.rename(columns=existing)

    # Normalise nutriscore to uppercase
    if "nutriscore" in df.columns:
        df["nutriscore"] = df["nutriscore"].str.upper().fillna("unknown")

    # sodium: OFF stores g/100g, convert to mg
    if "sodium_mg" in df.columns:
        df["sodium_mg"] = df["sodium_mg"] * 1000

    print(f"off_enriched: {len(df):,} products")
    out = PROCESSED / "off_enriched.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved: {out}")
