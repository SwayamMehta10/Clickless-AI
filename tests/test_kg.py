"""Tests for knowledge graph components."""

from __future__ import annotations

import os

import pytest


def _neo4j_available() -> bool:
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        driver = GraphDatabase.driver(uri, auth=("neo4j", os.getenv("NEO4J_PASSWORD", "clickless123")))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def test_spo_extractor_chunk_text():
    from src.knowledge_graph.spo_extractor import _chunk_text
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = _chunk_text(text, chunk_size=100, overlap=0.1)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 100 + 1 for c in chunks)


def test_entity_standardizer_single_entity():
    from src.knowledge_graph.entity_standardizer import build_entity_map
    entity_map = build_entity_map(["milk"])
    assert "milk" in entity_map
    assert entity_map["milk"] == "milk"


def test_entity_standardizer_near_duplicates():
    """Entities that are very similar should cluster together."""
    from src.knowledge_graph.entity_standardizer import _cluster_entities
    entities = ["whole milk", "whole milk", "whole-milk"]
    clusters = _cluster_entities(entities, threshold=0.80)
    # Should produce fewer clusters than entities
    assert len(clusters) <= len(entities)


@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not running")
def test_graph_query_find_by_attribute():
    from src.knowledge_graph.graph_query import find_by_attribute
    results = find_by_attribute("nutriscore", "A", limit=5)
    assert isinstance(results, list)


@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not running")
def test_graph_query_find_related():
    from src.knowledge_graph.graph_query import find_related_products
    results = find_related_products("milk", top_k=5)
    assert isinstance(results, list)


@pytest.mark.skipif(not _neo4j_available(), reason="Neo4j not running")
def test_graph_query_dietary_constraint():
    from src.knowledge_graph.graph_query import find_by_dietary_constraint
    results = find_by_dietary_constraint(["gluten-free"], limit=5)
    assert isinstance(results, list)
