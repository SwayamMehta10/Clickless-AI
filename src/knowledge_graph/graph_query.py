"""Cypher query helpers for the ClickLess AI knowledge graph."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase, Driver

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

_driver: Optional[Driver] = None


def _get_driver() -> Driver:
    global _driver
    if _driver is None:
        cfg = get_settings()
        neo4j_cfg = cfg["neo4j"]
        uri = os.getenv("NEO4J_URI", neo4j_cfg["uri"])
        user = os.getenv("NEO4J_USER", neo4j_cfg.get("user", "neo4j"))
        password = os.getenv("NEO4J_PASSWORD", neo4j_cfg.get("password", "clickless123"))
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def find_related_products(product_name: str, top_k: int = 5) -> List[Dict]:
    """Find products related to a given product via KG relationships."""
    query = """
    MATCH (p:Product {name: $name})-[r:RELATES|CONTAINS*1..2]-(related:Product)
    WHERE related.name <> $name
    RETURN DISTINCT related.name AS name,
           related.nutriscore AS nutriscore,
           related.nova_group AS nova_group,
           count(r) AS relationship_count
    ORDER BY relationship_count DESC
    LIMIT $top_k
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, name=product_name, top_k=top_k)
        return [dict(r) for r in result]


def find_by_attribute(
    attribute: str,
    value: Any,
    limit: int = 10,
) -> List[Dict]:
    """Find products by a specific attribute (nutriscore, nova_group, etc.)."""
    query = f"""
    MATCH (p:Product)
    WHERE p.{attribute} = $value
    RETURN p.name AS name, p.nutriscore AS nutriscore,
           p.nova_group AS nova_group, p.brand AS brand
    LIMIT $limit
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, value=value, limit=limit)
        return [dict(r) for r in result]


def find_by_dietary_constraint(
    dietary_flags: List[str],
    query_name: str = "",
    limit: int = 10,
) -> List[Dict]:
    """Find products matching dietary constraints."""
    # Build a Cypher query based on flags
    conditions = []
    if "gluten-free" in [f.lower() for f in dietary_flags]:
        conditions.append("NOT p.allergens CONTAINS 'gluten'")
    if "organic" in [f.lower() for f in dietary_flags]:
        conditions.append("toLower(p.name) CONTAINS 'organic'")
    if "vegan" in [f.lower() for f in dietary_flags]:
        conditions.append("NOT (p.allergens CONTAINS 'milk' OR p.allergens CONTAINS 'eggs')")
    if "low-sodium" in [f.lower() for f in dietary_flags]:
        conditions.append("(p.sodium_mg IS NULL OR p.sodium_mg < 140)")

    where_clause = " AND ".join(conditions) if conditions else "true"
    name_filter = f"toLower(p.name) CONTAINS toLower('{query_name}') AND " if query_name else ""

    cypher = f"""
    MATCH (p:Product)
    WHERE {name_filter}{where_clause}
    RETURN p.name AS name, p.nutriscore AS nutriscore,
           p.nova_group AS nova_group, p.brand AS brand,
           p.allergens AS allergens
    ORDER BY p.nutriscore ASC
    LIMIT $limit
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(cypher, limit=limit)
        return [dict(r) for r in result]


def get_product_subgraph(product_name: str, depth: int = 2) -> Dict:
    """Get nodes and relationships for a product's subgraph (for visualization)."""
    depth = max(1, min(int(depth), 5))
    product_query = f"""
    MATCH path = (p:Product {{name: $name}})-[*1..{depth}]-(neighbor)
    RETURN nodes(path) AS nodes, relationships(path) AS rels
    LIMIT 30
    """
    entity_query = """
    MATCH (e:Entity)
    WHERE toLower(e.name) CONTAINS toLower($name)
    WITH e LIMIT 2
    MATCH path = (e)-[r:RELATES]-(neighbor)
    RETURN nodes(path) AS nodes, relationships(path) AS rels
    LIMIT 25
    """
    fallback_query = """
    MATCH (e:Entity)-[r:RELATES]-(n)
    WITH e, count(r) AS deg ORDER BY deg DESC LIMIT 2
    MATCH path = (e)-[:RELATES]-(neighbor)
    RETURN nodes(path) AS nodes, relationships(path) AS rels
    LIMIT 25
    """
    driver = _get_driver()
    nodes_seen = {}
    edges = []

    def _ingest(records):
        for record in records:
            for node in record["nodes"]:
                nid = node.element_id
                if nid not in nodes_seen:
                    nodes_seen[nid] = {
                        "id": nid,
                        "label": list(node.labels)[0] if node.labels else "Entity",
                        "name": node.get("name", ""),
                    }
            for rel in record["rels"]:
                edges.append({
                    "source": rel.start_node.element_id,
                    "target": rel.end_node.element_id,
                    "predicate": rel.get("predicate", rel.type),
                })

    with driver.session() as session:
        _ingest(list(session.run(product_query, name=product_name)))
        if not nodes_seen:
            # Try entity substring match using product name tokens
            tokens = [t for t in product_name.split() if len(t) > 3]
            for token in tokens:
                _ingest(list(session.run(entity_query, name=token)))
                if nodes_seen:
                    break
        if not nodes_seen:
            _ingest(list(session.run(fallback_query)))
    return {"nodes": list(nodes_seen.values()), "edges": edges}


def get_nutrition_context(product_name: str) -> str:
    """Return a human-readable nutritional summary for a product from the KG."""
    query = """
    MATCH (p:Product)
    WHERE toLower(p.name) CONTAINS toLower($name)
    RETURN p.name AS name, p.nutriscore AS nutriscore, p.nova_group AS nova_group,
           p.energy_kcal AS kcal, p.protein_g AS protein,
           p.fat_g AS fat, p.sodium_mg AS sodium, p.allergens AS allergens
    LIMIT 1
    """
    driver = _get_driver()
    with driver.session() as session:
        result = session.run(query, name=product_name)
        record = result.single()
        if not record:
            return ""
        r = dict(record)
        parts = [f"Nutri-Score: {r.get('nutriscore', 'N/A')}"]
        if r.get("kcal"):
            parts.append(f"{r['kcal']:.0f} kcal/100g")
        if r.get("protein"):
            parts.append(f"protein {r['protein']:.1f}g")
        if r.get("fat"):
            parts.append(f"fat {r['fat']:.1f}g")
        if r.get("sodium"):
            parts.append(f"sodium {r['sodium']:.0f}mg")
        if r.get("allergens"):
            parts.append(f"allergens: {r['allergens']}")
        return " | ".join(parts)


def close():
    global _driver
    if _driver:
        _driver.close()
        _driver = None
