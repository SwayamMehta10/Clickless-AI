"""Standardize entity names from SPO triples.

Uses TF-IDF cosine clustering (threshold 0.85) to group near-duplicates,
then Mistral 7B selects the canonical name for each cluster.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.llm import ollama_client as llm

logger = logging.getLogger(__name__)

_CANONICAL_PROMPT = """\
You are standardizing food product entity names.
Given these similar names that refer to the same entity, choose the best canonical name.
Pick the most specific, commonly used, and correctly spelled version.

Names: {names}

Return ONLY the canonical name as a plain string (no JSON, no explanation).
"""


def _cluster_entities(
    entities: List[str],
    threshold: float = 0.85,
) -> List[List[str]]:
    """Group entity names by TF-IDF cosine similarity."""
    if not entities:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        matrix = vectorizer.fit_transform(entities)
    except ValueError:
        return [[e] for e in entities]

    sim = cosine_similarity(matrix)
    n = len(entities)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if not visited[j] and sim[i, j] >= threshold:
                cluster.append(j)
                visited[j] = True
        clusters.append([entities[k] for k in cluster])

    return clusters


def _pick_canonical(cluster: List[str]) -> str:
    """Use Mistral 7B to pick the best canonical name from a cluster."""
    if len(cluster) == 1:
        return cluster[0]

    prompt = _CANONICAL_PROMPT.format(names=", ".join(f'"{n}"' for n in cluster))
    try:
        canonical = llm.generate(prompt, role="spo").strip().strip('"').strip("'")
        # Fallback: longest name (usually most specific)
        if not canonical or canonical not in cluster:
            canonical = max(cluster, key=len)
        return canonical
    except Exception as exc:
        logger.debug("Canonical name selection failed: %s", exc)
        return max(cluster, key=len)


def build_entity_map(entities: List[str], threshold: float = 0.85) -> Dict[str, str]:
    """Return a mapping of raw entity name -> canonical name."""
    unique = list(dict.fromkeys(e.lower().strip() for e in entities if e.strip()))
    clusters = _cluster_entities(unique, threshold=threshold)

    entity_map: Dict[str, str] = {}
    for cluster in clusters:
        canonical = _pick_canonical(cluster)
        for member in cluster:
            entity_map[member] = canonical

    logger.info("Standardized %d entities into %d canonical forms", len(unique), len(clusters))
    return entity_map


def standardize_triples(
    triples: List[dict],
    threshold: float = 0.85,
) -> Tuple[List[dict], Dict[str, str]]:
    """Standardize all entities in a list of SPO triples.

    Returns (standardized_triples, entity_map).
    """
    all_entities = []
    for t in triples:
        all_entities.extend([t.get("s", ""), t.get("o", "")])

    entity_map = build_entity_map(all_entities, threshold=threshold)

    standardized = []
    for t in triples:
        s = entity_map.get(t["s"].lower().strip(), t["s"])
        p = t["p"].lower().strip()
        o = entity_map.get(t["o"].lower().strip(), t["o"])
        standardized.append({"product": t.get("product", ""), "s": s, "p": p, "o": o})

    return standardized, entity_map
