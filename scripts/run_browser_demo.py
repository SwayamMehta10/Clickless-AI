"""Drive the BrowserUse Cloud checkout against the three proposal scenarios.

Reads scenarios from evaluation/scenarios.py, builds a representative cart for
each (10 items for weekly, 8 for dietary, 12 for bulk) using the live Instacart
client + KG ranker, then invokes the CheckoutAgent against Browser Use Cloud
for each scenario. Per-scenario screenshots, action logs and live_url
recordings are persisted under artifacts/checkout/<scenario_id>/.

Usage:
  BROWSERUSE_API_KEY=... python scripts/run_browser_demo.py --scenarios all

The aggregate manifest is written to artifacts/checkout/manifest.json so the
final paper / demo deck can reference each scenario's artifacts directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import List

from src.api.instacart_client import InstacartClient
from src.api.product_schema import CartItem
from src.browser.checkout_agent import run_checkout
from src.ranking.kg_ranker import rank_with_kg

logger = logging.getLogger(__name__)

_ARTIFACTS = Path("/scratch/smehta90/Clickless AI/artifacts/checkout")
_ARTIFACTS.mkdir(parents=True, exist_ok=True)


SCENARIO_SPECS = [
    {
        "id": "scenario_1_weekly",
        "label": "Weekly grocery (10 items)",
        "items": ["milk", "eggs", "bread", "chicken breast", "broccoli",
                  "apples", "rice", "pasta", "yogurt", "olive oil"],
        "dietary": [],
        "max_price": 60.0,
    },
    {
        "id": "scenario_2_dietary",
        "label": "Dietary-restricted (gluten-free)",
        "items": ["gluten free bread", "gluten free pasta", "almond milk",
                  "oat milk", "tofu", "spinach", "quinoa", "blueberries"],
        "dietary": ["gluten-free", "dairy-free"],
        "max_price": 75.0,
    },
    {
        "id": "scenario_3_bulk",
        "label": "Budget-capped bulk",
        "items": ["chicken broth", "canned tomatoes", "rice", "pasta",
                  "flour", "sugar", "olive oil", "frozen vegetables",
                  "frozen chicken", "eggs", "bread", "yogurt"],
        "dietary": [],
        "max_price": 50.0,
    },
]


async def _build_cart(spec: dict) -> List[CartItem]:
    client = InstacartClient()
    cart: List[CartItem] = []
    for item in spec["items"]:
        candidates = await client.search_products(
            query=item,
            limit=8,
            dietary_flags=spec["dietary"],
            max_price=spec["max_price"],
        )
        if not candidates:
            continue
        ranked = rank_with_kg(
            query=item,
            candidates=candidates,
            dietary_flags=spec["dietary"],
            user_budget=spec["max_price"],
        )
        if ranked:
            cart.append(CartItem(product=ranked[0].product, quantity=1))
    return cart


async def _run_one(spec: dict) -> dict:
    logger.info("=== %s: %s ===", spec["id"], spec["label"])
    cart = await _build_cart(spec)
    logger.info("Cart built: %d items, est subtotal $%.2f",
                len(cart), sum((c.product.price or 0) for c in cart))
    result = await run_checkout(
        cart_items=cart,
        scenario_id=spec["id"],
        user_id="demo",
    )
    return {"scenario": spec, "cart_size": len(cart), "result": result}


async def _run_all(selected: List[str]) -> dict:
    if "all" in selected:
        specs = SCENARIO_SPECS
    else:
        specs = [s for s in SCENARIO_SPECS if s["id"] in selected]
    runs = []
    for spec in specs:
        try:
            run = await _run_one(spec)
        except Exception as exc:
            logger.exception("Scenario %s failed", spec["id"])
            run = {"scenario": spec, "result": {"success": False, "error": str(exc)}}
        runs.append(run)
    manifest = {
        "n_scenarios": len(runs),
        "n_success": sum(1 for r in runs if r.get("result", {}).get("success")),
        "scenarios": runs,
    }
    (_ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    manifest = asyncio.run(_run_all(args.scenarios))
    print(json.dumps({"n": manifest["n_scenarios"], "n_success": manifest["n_success"]}, indent=2))
    print(f"Artifacts in: {_ARTIFACTS}")


if __name__ == "__main__":
    main()
