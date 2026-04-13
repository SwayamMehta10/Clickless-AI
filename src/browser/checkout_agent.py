"""Browser checkout agent using BrowserUse + Playwright.

Workflow:
  1. Get session token from vault
  2. Navigate to Instacart cart URL
  3. Verify cart contents match expected items
  4. Proceed to checkout (stops before payment for demo safety)

Llama 3.2 90B vision is used as fallback for screenshot interpretation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from src.api.product_schema import CartItem
from src.llm.session_manager import SessionManager

logger = logging.getLogger(__name__)


class CheckoutAgent:
    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self._session_mgr = SessionManager(user_id)

    async def checkout(
        self,
        cart_items: List[CartItem],
        cart_url: Optional[str] = None,
        stop_before_payment: bool = True,
    ) -> dict:
        """Execute the checkout flow. Returns status dict."""
        try:
            from browser_use import Agent as BrowserAgent
            from playwright.async_api import async_playwright
        except ImportError:
            logger.error("browser-use or playwright not installed")
            return {"success": False, "error": "browser-use/playwright not available"}

        token = self._session_mgr.get_or_create()
        target_url = cart_url or "https://www.instacart.com/store/checkout"

        task = self._build_task(cart_items, target_url, stop_before_payment)

        logger.info("Starting browser checkout for %d items", len(cart_items))
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()

                # Inject session token as a cookie
                await context.add_cookies([{
                    "name": "instacart_session",
                    "value": token,
                    "domain": ".instacart.com",
                    "path": "/",
                }])

                agent = BrowserAgent(
                    task=task,
                    llm=_get_browser_llm(),
                    browser_context=context,
                )
                result = await agent.run()
                await browser.close()

            logger.info("Browser checkout result: %s", result)
            return {"success": True, "result": str(result), "items": len(cart_items)}

        except Exception as exc:
            logger.error("Browser checkout failed: %s", exc)
            return {"success": False, "error": str(exc)}

    def _build_task(
        self,
        cart_items: List[CartItem],
        cart_url: str,
        stop_before_payment: bool,
    ) -> str:
        items_str = "\n".join(
            f"- {item.product.name} x{item.quantity}"
            for item in cart_items
        )
        stop_note = "IMPORTANT: Stop at the payment page - do NOT enter payment information." if stop_before_payment else ""
        return f"""
Navigate to {cart_url} and complete the grocery checkout process.

Expected cart items:
{items_str}

Steps:
1. Verify the cart contains the expected items
2. Proceed to checkout
3. {stop_note}
4. Take a screenshot of the final checkout page
"""


def _get_browser_llm():
    """Return LLM config for BrowserUse (uses Ollama via langchain)."""
    try:
        from langchain_community.llms import Ollama
        from src.utils.config import get_settings
        cfg = get_settings()
        model = cfg["ollama"]["models"].get("vision", "llama3.2-vision:11b")
        return Ollama(model=model, base_url=cfg["ollama"].get("base_url", "http://localhost:11434"))
    except ImportError:
        logger.warning("langchain_community not available for browser LLM")
        return None


async def run_checkout(
    cart_items: List[CartItem],
    user_id: str = "default",
    cart_url: Optional[str] = None,
) -> dict:
    """Top-level convenience function."""
    agent = CheckoutAgent(user_id=user_id)
    return await agent.checkout(cart_items, cart_url=cart_url)
