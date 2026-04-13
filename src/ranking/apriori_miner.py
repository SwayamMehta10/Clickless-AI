"""Apriori co-purchase mining on Instacart 2017 transactions.

Exposes: get_copurchase_suggestions(product_name, top_k) -> List[str]
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

logger = logging.getLogger(__name__)

_PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")
_RULES_PATH = _PROCESSED / "association_rules.pkl"

_rules_cache: Optional[pd.DataFrame] = None


def _load_transactions() -> List[List[str]]:
    path = _PROCESSED / "transactions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"transactions.pkl not found at {path}. Run preprocess_instacart.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def mine_rules(
    min_support: float = 0.01,
    min_confidence: float = 0.3,
    save: bool = True,
) -> pd.DataFrame:
    """Mine association rules from transactions and optionally cache to disk."""
    logger.info("Mining Apriori rules (min_support=%.3f, min_confidence=%.2f)...", min_support, min_confidence)
    transactions = _load_transactions()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_te = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = apriori(df_te, min_support=min_support, use_colnames=True, low_memory=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    logger.info("Mined %d rules", len(rules))

    if save:
        _PROCESSED.mkdir(exist_ok=True)
        with open(_RULES_PATH, "wb") as f:
            pickle.dump(rules, f)
        logger.info("Saved rules to %s", _RULES_PATH)

    return rules


def _get_rules() -> pd.DataFrame:
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    if _RULES_PATH.exists():
        with open(_RULES_PATH, "rb") as f:
            _rules_cache = pickle.load(f)
        logger.info("Loaded %d association rules from disk", len(_rules_cache))
    else:
        logger.warning("No cached rules found -- mining now (this may take several minutes)...")
        _rules_cache = mine_rules()

    return _rules_cache


def get_copurchase_suggestions(product_name: str, top_k: int = 5) -> List[str]:
    """Return top-k products frequently bought with product_name."""
    rules = _get_rules()

    # Find rules whose antecedents contain this product
    mask = rules["antecedents"].apply(lambda s: product_name in s)
    matching = rules[mask].head(top_k * 3)  # over-fetch then deduplicate

    suggestions: List[str] = []
    seen = {product_name}
    for _, row in matching.iterrows():
        for item in row["consequents"]:
            if item not in seen:
                suggestions.append(item)
                seen.add(item)
            if len(suggestions) >= top_k:
                return suggestions

    logger.debug("Found %d suggestions for '%s'", len(suggestions), product_name)
    return suggestions


def get_rules_df() -> pd.DataFrame:
    """Expose the full rules DataFrame for evaluation."""
    return _get_rules()
