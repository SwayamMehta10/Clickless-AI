"""Preprocess Instacart 2017 CSVs into product_features.parquet and transactions.pkl."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

RAW = Path("/scratch/smehta90/Clickless AI/data/raw/instacart_2017")
PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")
PROCESSED.mkdir(exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = pd.read_csv(RAW / "products.csv")
    aisles = pd.read_csv(RAW / "aisles.csv")
    departments = pd.read_csv(RAW / "departments.csv")
    orders = pd.read_csv(RAW / "orders.csv")
    order_products_prior = pd.read_csv(RAW / "order_products__prior.csv")
    return products, aisles, departments, orders, order_products_prior


def build_product_features(
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
    order_products: pd.DataFrame,
) -> pd.DataFrame:
    """Build product-level feature table."""
    # Merge aisle and department names
    df = products.merge(aisles, on="aisle_id").merge(departments, on="department_id")

    # Compute reorder rate per product
    reorder = (
        order_products.groupby("product_id")
        .agg(
            total_orders=("order_id", "count"),
            reorders=("reordered", "sum"),
        )
        .reset_index()
    )
    reorder["reorder_rate"] = reorder["reorders"] / reorder["total_orders"]

    df = df.merge(reorder[["product_id", "reorder_rate", "total_orders"]], on="product_id", how="left")
    df["reorder_rate"] = df["reorder_rate"].fillna(0.0)
    df["total_orders"] = df["total_orders"].fillna(0).astype(int)

    print(f"product_features: {len(df):,} products")
    return df


def build_transactions(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    products: pd.DataFrame,
    max_orders: int = 100_000,
) -> list[list[str]]:
    """Build transaction list for Apriori mining."""
    # Use prior orders only, limit size for memory
    prior_orders = orders[orders["eval_set"] == "prior"]["order_id"].values
    op = order_products[order_products["order_id"].isin(prior_orders)]

    # Sample for tractability
    sampled_orders = op["order_id"].unique()[:max_orders]
    op_sampled = op[op["order_id"].isin(sampled_orders)]

    # Map product_id -> product_name
    id_to_name = products.set_index("product_id")["product_name"].to_dict()
    op_sampled = op_sampled.copy()
    op_sampled["product_name"] = op_sampled["product_id"].map(id_to_name)

    transactions = (
        op_sampled.groupby("order_id")["product_name"]
        .apply(list)
        .tolist()
    )
    print(f"transactions: {len(transactions):,} orders")
    return transactions


if __name__ == "__main__":
    print("Loading raw Instacart data...")
    products, aisles, departments, orders, order_products_prior = load_raw()

    print("Building product features...")
    features = build_product_features(products, aisles, departments, order_products_prior)
    features.to_parquet(PROCESSED / "product_features.parquet", index=False)
    print(f"Saved: {PROCESSED / 'product_features.parquet'}")

    print("Building transactions...")
    transactions = build_transactions(orders, order_products_prior, products)
    with open(PROCESSED / "transactions.pkl", "wb") as f:
        pickle.dump(transactions, f)
    print(f"Saved: {PROCESSED / 'transactions.pkl'}")

    print("Done.")
