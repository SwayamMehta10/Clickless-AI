"""Session-scoped runtime controls and mock adapters for the demo UI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.api.instacart_client import InstacartClient
from src.api.instacart_mock import MockInstacartClient, get_client
from src.api.product_schema import Product, RankedProduct
from src.browser.checkout_agent import run_checkout
from src.knowledge_graph.graph_query import get_product_subgraph
from src.nlu.dialogue_state import Slots
from src.ranking.kg_ranker import rank_with_kg
from src.utils.config import get_settings

MOCKED = "Mocked"
LIVE = "Live"

SHOPPING_API = "shopping_api"
LLM = "llm"
KNOWLEDGE_GRAPH = "knowledge_graph"
RANKING = "ranking"
CHECKOUT = "checkout"

SERVICE_ORDER = [SHOPPING_API, LLM, KNOWLEDGE_GRAPH, RANKING, CHECKOUT]
SERVICE_LABELS = {
    SHOPPING_API: "Shopping API",
    LLM: "LLM",
    KNOWLEDGE_GRAPH: "Knowledge Graph",
    RANKING: "Ranking",
    CHECKOUT: "Checkout",
}

_PROCESSED = Path("/scratch/smehta90/Clickless AI/data/processed")
_BUDGET_RE = re.compile(r"\b(?:under|below|max(?:imum)?|budget(?: of)?|for)\s+\$?(\d+(?:\.\d+)?)", re.IGNORECASE)
_QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(gallon|gallons|oz|ounce|ounces|lb|lbs|pound|pounds|pack|packs|dozen|ct)\b",
    re.IGNORECASE,
)
_PRICE_TOKEN_RE = re.compile(r"\$?\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class LiveReadiness:
    ready: bool
    message: str


@dataclass(frozen=True)
class DemoRuntimeConfig:
    service_modes: Dict[str, str]

    @classmethod
    def all_mocked(cls) -> "DemoRuntimeConfig":
        return cls({service: MOCKED for service in SERVICE_ORDER})

    @classmethod
    def all_live(cls) -> "DemoRuntimeConfig":
        return cls({service: LIVE for service in SERVICE_ORDER})

    def mode_for(self, service: str) -> str:
        return self.service_modes.get(service, MOCKED)

    def with_mode(self, service: str, mode: str) -> "DemoRuntimeConfig":
        updated = dict(self.service_modes)
        updated[service] = mode
        return DemoRuntimeConfig(updated)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _overlap_score(query: str, product_name: str) -> float:
    query_tokens = set(_tokenize(query))
    name_tokens = set(_tokenize(product_name))
    if not query_tokens or not name_tokens:
        return 0.0
    return len(query_tokens & name_tokens) / len(query_tokens)


def _keyword_present(text: str, keywords: List[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _slug(text: str) -> str:
    tokens = _tokenize(text)
    return "-".join(tokens) or "item"


def _infer_allergens(product_name: str) -> List[str]:
    lowered = (product_name or "").lower()
    allergens = []
    if any(token in lowered for token in ["milk", "yogurt", "cheese", "butter"]):
        allergens.append("milk")
    if "egg" in lowered:
        allergens.append("eggs")
    if any(token in lowered for token in ["bread", "wheat", "pasta"]):
        allergens.append("gluten")
    if "peanut" in lowered:
        allergens.append("peanuts")
    return allergens


def _infer_nutrition_context(product_name: str) -> str:
    lowered = (product_name or "").lower()
    if any(token in lowered for token in ["spinach", "banana", "organic", "brown rice"]):
        score = "A"
    elif any(token in lowered for token in ["broth", "almond milk", "salmon"]):
        score = "B"
    elif any(token in lowered for token in ["bread", "milk", "yogurt"]):
        score = "C"
    else:
        score = "B"

    sodium = "120mg"
    if "broth" in lowered:
        sodium = "480mg"
    allergens = _infer_allergens(product_name)
    allergen_text = ", ".join(allergens) if allergens else "none listed"
    return f"Nutri-Score: {score} | sodium {sodium} | allergens: {allergen_text}"


def classify_intent_mock(user_message: str, conversation_history: str = "") -> Tuple[str, float]:
    text = (user_message or "").lower()
    if _keyword_present(text, ["checkout", "check out", "pay now", "place order"]):
        return "checkout", 0.98
    if _keyword_present(text, ["remove", "delete", "take out"]):
        return "remove_from_cart", 0.94
    if _keyword_present(text, ["add", "put in cart", "put this in my cart"]):
        return "add_to_cart", 0.95
    if _keyword_present(text, ["recommend", "suggest", "what goes well", "ideas for"]):
        return "get_recommendation", 0.9

    has_constraint = bool(_BUDGET_RE.search(text)) or _keyword_present(
        text,
        ["gluten-free", "gluten free", "vegan", "organic", "low sodium", "dairy-free", "nut-free", "budget"],
    )
    search_verbs = ["need", "want", "find", "search", "show me", "looking for", "buy"]
    if has_constraint and not _keyword_present(text, search_verbs):
        cleaned = extract_slots_mock(user_message, conversation_history)
        if cleaned.item is None:
            return "set_constraint", 0.88

    if _keyword_present(text, ["hello", "hi", "hey", "how are you"]):
        return "chit_chat", 0.85
    return "search_product", 0.9


def extract_slots_mock(user_message: str, conversation_history: str = "") -> Slots:
    raw = user_message or ""
    text = raw.lower()
    dietary_flags = []
    for flag in ["organic", "gluten-free", "vegan", "low-sodium", "dairy-free", "nut-free"]:
        if flag in text or flag.replace("-", " ") in text:
            dietary_flags.append(flag)

    brand_match = re.search(r"\b([A-Z][a-zA-Z0-9]+)\s+brand\b", raw)
    brand_preference = brand_match.group(1) if brand_match else None

    max_price = None
    budget_match = _BUDGET_RE.search(raw)
    if budget_match:
        max_price = float(budget_match.group(1))

    quantity = None
    unit = None
    quantity_match = _QUANTITY_RE.search(raw)
    if quantity_match:
        quantity = float(quantity_match.group(1))
        unit = quantity_match.group(2).lower()
    else:
        plain_qty = re.search(r"\b(\d+)\b", raw)
        if plain_qty and not max_price:
            quantity = float(plain_qty.group(1))

    cleaned = text
    cleaned = re.sub(r"\b(i need|i want|find me|show me|looking for|please|get me|search for|add|remove)\b", " ", cleaned)
    cleaned = _BUDGET_RE.sub(" ", cleaned)
    cleaned = _QUANTITY_RE.sub(" ", cleaned)
    cleaned = _PRICE_TOKEN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\b(organic|gluten[- ]free|vegan|low sodium|low-sodium|dairy[- ]free|nut[- ]free|brand|cart|to my|from my|for the week)\b", " ", cleaned)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\b(of|some|a|an|the)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    item = cleaned or None

    return Slots(
        item=item,
        quantity=quantity,
        unit=unit,
        max_price=max_price,
        dietary_flags=dietary_flags,
        brand_preference=brand_preference,
    )


def generate_response_mock(state: dict) -> str:
    dialogue_state = state["dialogue_state"]
    ranked = state.get("ranked_results", [])
    error = state.get("error")

    if error:
        return f"I hit a live-service limitation, so I stayed in demo mode: {error}"

    if dialogue_state.current_intent == "checkout":
        return f"Demo checkout is ready for {len(dialogue_state.cart)} item(s)."

    if dialogue_state.current_intent == "add_to_cart":
        return f"Added to your cart. You now have {len(dialogue_state.cart)} item(s)."

    if dialogue_state.current_intent == "remove_from_cart":
        return f"Updated your cart. You now have {len(dialogue_state.cart)} item(s)."

    if ranked:
        preview = []
        for ranked_product in ranked[:3]:
            product = ranked_product.product
            if product.price is not None:
                preview.append(f"{product.name} (${product.price:.2f})")
            else:
                preview.append(product.name)
        return f"I found {len(ranked)} option(s). Top picks: {', '.join(preview)}."

    if dialogue_state.cart:
        return f"Your cart currently has {len(dialogue_state.cart)} item(s)."

    return "Demo mode is on. Tell me what groceries you want and I’ll mock the rest."


def _mock_copurchase_suggestions(product_name: str, top_k: int = 3) -> List[str]:
    lowered = (product_name or "").lower()
    if "milk" in lowered:
        suggestions = ["Cereal", "Bananas", "Bread"]
    elif "bread" in lowered:
        suggestions = ["Butter", "Eggs", "Jam"]
    elif "broth" in lowered:
        suggestions = ["Soup Noodles", "Crackers", "Carrots"]
    else:
        suggestions = ["Bananas", "Spinach", "Eggs"]
    return suggestions[:top_k]


def _passes_dietary_filters(product: Product, dietary_flags: List[str]) -> bool:
    allergens = [allergen.lower() for allergen in (product.allergens or _infer_allergens(product.name))]
    for flag in dietary_flags:
        lowered = flag.lower()
        if lowered == "gluten-free" and "gluten" in allergens:
            return False
        if lowered == "vegan" and any(allergen in allergens for allergen in ["milk", "eggs"]):
            return False
    return True


def mock_rank_products(
    query: str,
    candidates: List[Product],
    dietary_flags: Optional[List[str]] = None,
    user_budget: Optional[float] = None,
    cart_item_names: Optional[List[str]] = None,
) -> List[RankedProduct]:
    dietary_flags = dietary_flags or []
    cart_item_names = cart_item_names or []
    ranked: List[RankedProduct] = []

    for product in candidates:
        if not _passes_dietary_filters(product, dietary_flags):
            continue

        text_match = _overlap_score(query, product.name)
        availability = 1.0 if product.availability else 0.0
        if user_budget and product.price is not None:
            budget_fit = 1.0 if product.price <= user_budget else max(0.0, 1.0 - ((product.price - user_budget) / max(user_budget, 1.0)))
        else:
            budget_fit = 0.7
        reorder = product.reorder_rate if product.reorder_rate is not None else 0.3
        cart_boost = 0.0
        if cart_item_names and any(token in (product.name or "").lower() for token in _tokenize(" ".join(cart_item_names))):
            cart_boost = 0.2

        composite = (0.45 * text_match) + (0.2 * availability) + (0.2 * budget_fit) + (0.15 * min(max(reorder, 0.0), 1.0)) + cart_boost
        ranked.append(
            RankedProduct(
                product=product,
                score=composite,
                rank=0,
                score_breakdown={
                    "query_match": round(text_match, 3),
                    "availability": round(availability, 3),
                    "budget_fit": round(budget_fit, 3),
                    "reorder_rate": round(min(max(reorder, 0.0), 1.0), 3),
                },
                copurchase_suggestions=_mock_copurchase_suggestions(product.name),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def get_relevance_score_mock(product_name: str, query: str, dietary_flags: Optional[List[str]] = None) -> float:
    product = Product(instacart_id="mock", name=product_name, allergens=_infer_allergens(product_name))
    if not _passes_dietary_filters(product, dietary_flags or []):
        return 0.0
    overlap = _overlap_score(query, product_name)
    if overlap == 0.0:
        return 0.35
    return min(1.0, 0.35 + overlap * 0.65)


def get_product_subgraph_mock(product_name: str, depth: int = 2) -> Dict:
    product_id = f"product-{_slug(product_name)}"
    nutrition_id = f"nutrition-{_slug(product_name)}"
    dietary_id = f"dietary-{_slug(product_name)}"
    pairings_id = f"pairings-{_slug(product_name)}"
    return {
        "nodes": [
            {"id": product_id, "label": "Product", "name": product_name},
            {"id": nutrition_id, "label": "Attribute", "name": _infer_nutrition_context(product_name)},
            {"id": dietary_id, "label": "Entity", "name": "Demo dietary profile"},
            {"id": pairings_id, "label": "Category", "name": ", ".join(_mock_copurchase_suggestions(product_name, top_k=2))},
        ],
        "edges": [
            {"source": product_id, "target": nutrition_id, "predicate": "HAS_NUTRITION"},
            {"source": product_id, "target": dietary_id, "predicate": "MATCHES"},
            {"source": product_id, "target": pairings_id, "predicate": "PAIR_WITH"},
        ],
    }


async def run_checkout_mock(cart_items, user_id: str = "default", cart_url: Optional[str] = None) -> Dict:
    return {
        "success": True,
        "result": f"Demo checkout completed for {len(cart_items)} item(s).",
        "items": len(cart_items),
        "cart_url": cart_url or "https://demo.instacart.local/mock-checkout",
        "mode": "mocked",
        "user_id": user_id,
    }


def _shopping_api_readiness() -> LiveReadiness:
    if os.getenv("INSTACART_API_KEY"):
        return LiveReadiness(True, "Instacart API key detected.")
    return LiveReadiness(False, "Missing INSTACART_API_KEY.")


def _llm_readiness() -> LiveReadiness:
    from src.llm import ollama_client

    nlu_ready = ollama_client.is_available("nlu")
    general_ready = ollama_client.is_available("general")
    if nlu_ready and general_ready:
        return LiveReadiness(True, "Ollama models for NLU and general responses are available.")
    return LiveReadiness(False, "Ollama is not ready with both nlu and general models.")


def _knowledge_graph_readiness() -> LiveReadiness:
    try:
        from neo4j import GraphDatabase

        cfg = get_settings()["neo4j"]
        uri = os.getenv("NEO4J_URI", cfg["uri"])
        user = os.getenv("NEO4J_USER", cfg.get("user", "neo4j"))
        password = os.getenv("NEO4J_PASSWORD", cfg.get("password", "clickless123"))
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return LiveReadiness(True, "Neo4j connection verified.")
    except Exception as exc:
        return LiveReadiness(False, f"Neo4j is unavailable: {exc}")


def _ranking_readiness() -> LiveReadiness:
    product_features = _PROCESSED / "product_features.parquet"
    transactions = _PROCESSED / "transactions.pkl"
    missing = []
    if not product_features.exists():
        missing.append("product_features.parquet")
    if not transactions.exists():
        missing.append("transactions.pkl")
    if missing:
        return LiveReadiness(False, f"Missing ranking artifacts: {', '.join(missing)}.")
    return LiveReadiness(True, "Ranking artifacts are available for live mode.")


def _checkout_readiness() -> LiveReadiness:
    try:
        import browser_use  # noqa: F401
        import playwright.async_api  # noqa: F401
        from langchain_community.llms import Ollama  # noqa: F401
    except Exception as exc:
        return LiveReadiness(False, f"Checkout dependencies are unavailable: {exc}")

    from src.llm import ollama_client

    if not (ollama_client.is_available("vision") and ollama_client.is_available("general")):
        return LiveReadiness(False, "Checkout requires the Ollama vision and general models to be available.")
    return LiveReadiness(True, "Browser checkout dependencies are available.")


_LIVE_READINESS = {
    SHOPPING_API: _shopping_api_readiness,
    LLM: _llm_readiness,
    KNOWLEDGE_GRAPH: _knowledge_graph_readiness,
    RANKING: _ranking_readiness,
    CHECKOUT: _checkout_readiness,
}


def get_live_readiness(service: str) -> LiveReadiness:
    checker = _LIVE_READINESS[service]
    return checker()


def get_checkout_readiness() -> LiveReadiness:
    return get_live_readiness(CHECKOUT)


def apply_service_mode(config: DemoRuntimeConfig, service: str, desired_mode: str) -> Tuple[DemoRuntimeConfig, Optional[str]]:
    if desired_mode == MOCKED:
        return config.with_mode(service, MOCKED), None

    readiness = get_live_readiness(service)
    if not readiness.ready:
        return config, f"{SERVICE_LABELS[service]} stayed mocked: {readiness.message}"
    return config.with_mode(service, LIVE), None


def apply_preset_selection(config: DemoRuntimeConfig, preset: str) -> Tuple[DemoRuntimeConfig, List[str]]:
    if preset == "All Mocked":
        return DemoRuntimeConfig.all_mocked(), []

    if preset != "All Live":
        return config, []

    updated = DemoRuntimeConfig.all_mocked()
    errors: List[str] = []
    for service in SERVICE_ORDER:
        updated, error = apply_service_mode(updated, service, LIVE)
        if error:
            errors.append(error)
    return updated, errors


def get_api_client_for_runtime(runtime_config: Optional[DemoRuntimeConfig] = None):
    if runtime_config is None:
        return get_client()
    if runtime_config.mode_for(SHOPPING_API) == LIVE:
        return InstacartClient()
    return MockInstacartClient()


def rank_products_for_runtime(
    runtime_config: Optional[DemoRuntimeConfig],
    query: str,
    candidates: List[Product],
    dietary_flags: Optional[List[str]] = None,
    user_budget: Optional[float] = None,
    cart_item_names: Optional[List[str]] = None,
) -> List[RankedProduct]:
    if runtime_config is None or runtime_config.mode_for(RANKING) == LIVE:
        relevance_scorer = None
        if runtime_config is not None and runtime_config.mode_for(KNOWLEDGE_GRAPH) == MOCKED:
            relevance_scorer = get_relevance_score_mock
        return rank_with_kg(
            query=query,
            candidates=candidates,
            dietary_flags=dietary_flags,
            user_budget=user_budget,
            cart_item_names=cart_item_names,
            relevance_scorer=relevance_scorer,
        )

    return mock_rank_products(
        query=query,
        candidates=candidates,
        dietary_flags=dietary_flags,
        user_budget=user_budget,
        cart_item_names=cart_item_names,
    )


def get_product_subgraph_for_runtime(
    product_name: str,
    depth: int = 2,
    runtime_config: Optional[DemoRuntimeConfig] = None,
) -> Dict:
    if runtime_config is not None and runtime_config.mode_for(KNOWLEDGE_GRAPH) == MOCKED:
        return get_product_subgraph_mock(product_name, depth=depth)
    return get_product_subgraph(product_name, depth=depth)


async def run_checkout_for_runtime(
    cart_items,
    user_id: str = "default",
    cart_url: Optional[str] = None,
    runtime_config: Optional[DemoRuntimeConfig] = None,
) -> Dict:
    if runtime_config is not None and runtime_config.mode_for(CHECKOUT) == MOCKED:
        return await run_checkout_mock(cart_items, user_id=user_id, cart_url=cart_url)
    return await run_checkout(cart_items, user_id=user_id, cart_url=cart_url)
