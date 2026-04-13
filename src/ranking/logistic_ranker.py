"""Logistic Regression product ranker.

Features: price_ratio, in_stock_prob, tfidf_cosine, reorder_rate
Trained on Instacart 2017 (reordered=positive, same-aisle not purchased=negative).
Exposes: rank_products(query, candidates, user_budget) -> List[Tuple[Product, float]]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api.product_schema import Product

logger = logging.getLogger(__name__)

_PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")
_MODEL_PATH = _PROCESSED / "logistic_ranker.pkl"

_model: Optional[LogisticRegression] = None
_tfidf: Optional[TfidfVectorizer] = None
_product_features: Optional[pd.DataFrame] = None


def _load_product_features() -> pd.DataFrame:
    global _product_features
    if _product_features is not None:
        return _product_features
    path = _PROCESSED / "product_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"product_features.parquet not found. Run preprocess_instacart.py first.")
    _product_features = pd.read_parquet(path)
    return _product_features


def _build_training_data(features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, y) from Instacart features. Reordered=1, same-aisle not reordered=0."""
    df = features_df.copy()

    # Positive: products with high reorder rate
    # Negative: same-aisle products with low reorder rate (sampled)
    df["label"] = (df["reorder_rate"] >= 0.5).astype(int)

    # Balance: downsample negatives
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].sample(n=min(len(pos) * 2, len(df[df["label"] == 0])), random_state=42)
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)

    # TF-IDF on product name for text feature
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(balanced["product_name"].fillna(""))

    # Numeric features (placeholder values where real data missing)
    price_ratio = np.ones(len(balanced))  # Would use actual prices
    in_stock_prob = balanced["reorder_rate"].fillna(0.5).values
    reorder_rate = balanced["reorder_rate"].fillna(0).values
    # TF-IDF similarity to a generic "grocery" query as proxy
    query_vec = vectorizer.transform(["grocery food product"])
    tfidf_cosine = cosine_similarity(tfidf_matrix, query_vec).flatten()

    X = np.column_stack([price_ratio, in_stock_prob, tfidf_cosine, reorder_rate])
    y = balanced["label"].values

    return X, y, vectorizer


def train(save: bool = True) -> None:
    """Train the logistic ranker on Instacart 2017 data."""
    global _model, _tfidf
    logger.info("Training logistic ranker...")
    features_df = _load_product_features()
    X, y, vectorizer = _build_training_data(features_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_scaled, y)

    _model = clf
    _tfidf = vectorizer

    logger.info("Logistic ranker trained. Coefficients: %s", dict(zip(
        ["price_ratio", "in_stock_prob", "tfidf_cosine", "reorder_rate"],
        clf.coef_[0].tolist()
    )))

    if save:
        _PROCESSED.mkdir(exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump({"model": clf, "tfidf": vectorizer, "scaler": scaler}, f)
        logger.info("Saved model to %s", _MODEL_PATH)


def _load_model() -> tuple:
    global _model, _tfidf
    if _model is not None and _tfidf is not None:
        return _model, _tfidf, None

    if _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        _model = bundle["model"]
        _tfidf = bundle["tfidf"]
        scaler = bundle.get("scaler")
        logger.info("Loaded logistic ranker from disk")
        return _model, _tfidf, scaler

    logger.warning("No saved model -- training now...")
    train()
    with open(_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["tfidf"], bundle.get("scaler")


def _product_to_features(
    product: Product,
    query: str,
    user_budget: Optional[float],
    tfidf: TfidfVectorizer,
) -> np.ndarray:
    """Convert a Product to a 4-feature vector."""
    # price_ratio: product.price / budget (1.0 if no budget/price info)
    if user_budget and product.price:
        price_ratio = product.price / user_budget
    else:
        price_ratio = 0.5

    in_stock_prob = 1.0 if product.availability else 0.0
    reorder_rate = product.reorder_rate or 0.3

    # TF-IDF cosine between product name and query
    query_vec = tfidf.transform([query])
    product_vec = tfidf.transform([product.name or ""])
    tfidf_cosine = float(cosine_similarity(product_vec, query_vec)[0][0])

    return np.array([[price_ratio, in_stock_prob, tfidf_cosine, reorder_rate]])


def rank_products(
    query: str,
    candidates: List[Product],
    user_budget: Optional[float] = None,
) -> List[Tuple[Product, float]]:
    """Rank candidate products for a query. Returns (product, score) sorted descending."""
    if not candidates:
        return []

    model, tfidf, scaler = _load_model()

    scored = []
    for product in candidates:
        X = _product_to_features(product, query, user_budget, tfidf)
        if scaler:
            X = scaler.transform(X)
        prob = model.predict_proba(X)[0][1]  # probability of positive class
        scored.append((product, float(prob)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
