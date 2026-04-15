"""Browser Use Cloud checkout agent.

Drives the final checkout handoff against instacart.com via the Browser Use
Cloud Agent Tasks API (https://docs.cloud.browser-use.com/guides/tasks). The
cloud API gives us:

- A managed Chromium with residential routing, mitigating Cloudflare blocks
  that local Playwright runs against the production retailer hit consistently.
- Step-by-step screenshots and an action log per task, persisted to disk
  under artifacts/checkout/<run_id>/.
- A live_url session URL that can be embedded in the Streamlit UI for the
  demo-video recording.

The agent receives an Instacart session token from the on-device credential
vault (Llama 3.2 11B), injects it as a cookie in the cloud browser context,
and executes a natural-language task that searches each cart item, adds the
top-ranked in-stock result, and stops on the checkout-review page without
placing a real order.

If the BROWSERUSE_API_KEY environment variable is unset (e.g. local CI), the
agent falls back to a local browser-use Agent instance running the same task
against a real Chromium. Both transports return the same CheckoutResult shape.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.api.product_schema import CartItem
from src.llm.session_manager import SessionManager

logger = logging.getLogger(__name__)


_BROWSERUSE_BASE_URL = "https://api.browser-use.com/api/v1"
_ARTIFACTS_DIR = Path("/scratch/smehta90/Clickless AI/artifacts/checkout")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CheckoutResult:
    success: bool
    cart_url: Optional[str] = None
    items_added: int = 0
    screenshots: List[str] = field(default_factory=list)
    live_url: Optional[str] = None
    action_log: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    run_id: Optional[str] = None
    artifact_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# Cloud transport
# ---------------------------------------------------------------------------

class _BrowserUseCloudTransport:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=_BROWSERUSE_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )

    async def run_task(
        self,
        task: str,
        cookies: List[Dict[str, str]],
        run_dir: Path,
        save_screenshots: bool = True,
    ) -> CheckoutResult:
        payload = {
            "task": task,
            "save_browser_data": True,
            "use_adblock": True,
            "use_proxy": True,
            "structured_output_json": json.dumps({
                "type": "object",
                "properties": {
                    "items_added": {"type": "integer"},
                    "cart_url": {"type": "string"},
                    "stopped_at": {"type": "string"},
                },
                "required": ["items_added", "stopped_at"],
            }),
            "browser_cookies": cookies,
        }

        resp = await self._client.post("/run-task", json=payload)
        resp.raise_for_status()
        task_record = resp.json()
        task_id = task_record.get("id") or task_record.get("task_id")
        live_url = task_record.get("live_url")
        logger.info("Browser Use Cloud task created: id=%s live_url=%s", task_id, live_url)

        action_log: List[Dict[str, Any]] = []
        screenshots: List[str] = []
        terminal = False
        result_payload: Dict[str, Any] = {}
        deadline = time.time() + 900

        while not terminal and time.time() < deadline:
            await asyncio.sleep(4)
            status_resp = await self._client.get(f"/task/{task_id}")
            status_resp.raise_for_status()
            status = status_resp.json()
            phase = status.get("status", "running")
            logger.debug("task %s phase=%s steps=%s", task_id, phase, len(status.get("steps", [])))

            for step in status.get("steps", []):
                step_id = step.get("id") or step.get("step_id")
                if any(a.get("step_id") == step_id for a in action_log):
                    continue
                action_log.append({
                    "step_id": step_id,
                    "action": step.get("action") or step.get("evaluation_previous_goal"),
                    "url": step.get("url"),
                    "screenshot_url": step.get("screenshot_url"),
                    "ts": step.get("timestamp"),
                })
                if save_screenshots and step.get("screenshot_url"):
                    shot_name = f"step_{len(screenshots):03d}.png"
                    path = run_dir / shot_name
                    try:
                        img_resp = await self._client.get(step["screenshot_url"])
                        img_resp.raise_for_status()
                        path.write_bytes(img_resp.content)
                        screenshots.append(str(path))
                    except Exception as exc:
                        logger.warning("screenshot fetch failed: %s", exc)

            if phase in ("finished", "completed", "stopped", "failed"):
                terminal = True
                result_payload = status.get("output") or status.get("result") or {}
                if phase == "failed":
                    return CheckoutResult(
                        success=False,
                        live_url=live_url,
                        action_log=action_log,
                        screenshots=screenshots,
                        error=status.get("error", "browser_use_cloud_failed"),
                    )

        return CheckoutResult(
            success=True,
            cart_url=result_payload.get("cart_url"),
            items_added=int(result_payload.get("items_added", 0)),
            live_url=live_url,
            action_log=action_log,
            screenshots=screenshots,
        )

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Local transport (open-source browser-use)
# ---------------------------------------------------------------------------

class _BrowserUseLocalTransport:
    async def run_task(
        self,
        task: str,
        cookies: List[Dict[str, str]],
        run_dir: Path,
        save_screenshots: bool = True,
    ) -> CheckoutResult:
        try:
            from browser_use import Agent as BrowserAgent  # type: ignore
            from playwright.async_api import async_playwright
        except ImportError as exc:
            return CheckoutResult(success=False, error=f"browser-use/playwright unavailable: {exc}")

        screenshots: List[str] = []
        action_log: List[Dict[str, Any]] = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                if cookies:
                    await context.add_cookies(cookies)

                agent = BrowserAgent(
                    task=task,
                    llm=_get_browser_llm(),
                    browser_context=context,
                    use_vision=True,
                )
                result = await agent.run()
                action_log.append({"action": "agent.run", "result": str(result)[:500]})

                page = (await context.pages())[0] if (await context.pages()) else await context.new_page()
                shot_path = run_dir / "step_final.png"
                try:
                    await page.screenshot(path=str(shot_path), full_page=True)
                    screenshots.append(str(shot_path))
                except Exception:
                    pass

                await browser.close()

            return CheckoutResult(
                success=True,
                items_added=len(cookies),
                screenshots=screenshots,
                action_log=action_log,
            )
        except Exception as exc:
            logger.error("Local browser-use checkout failed: %s", exc)
            return CheckoutResult(success=False, error=str(exc))


def _get_browser_llm():
    """Return a browser-use compatible LLM wrapping the local Ollama vision model.

    Per the browser-use docs (https://docs.browser-use.com/supported-models),
    the supported way to drive a local Ollama model from a browser-use Agent
    is `from browser_use import ChatOllama`. The class is exposed at the top
    level via lazy attribute loading; submodule paths like `browser_use.llm`
    are not introspectable as Python attributes and importing them directly
    will fail. Always import from the top-level package.
    """
    from src.utils.config import get_settings
    cfg = get_settings()
    model = cfg["ollama"]["models"].get("vision", "llama3.2-vision:11b")
    base_url = cfg["ollama"].get("base_url", "http://localhost:11434")

    try:
        from browser_use import ChatOllama
    except ImportError as exc:
        logger.warning(
            "browser_use.ChatOllama unavailable (%s); run `pip install -U browser-use`",
            exc,
        )
        return None

    try:
        return ChatOllama(model=model, host=base_url)
    except TypeError:
        # Older signature without the `host` kwarg — Ollama defaults to localhost.
        return ChatOllama(model=model)


# ---------------------------------------------------------------------------
# CheckoutAgent
# ---------------------------------------------------------------------------

class CheckoutAgent:
    """Drives the BrowserUse checkout handoff for a confirmed cart."""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self._session_mgr = SessionManager(user_id)
        self._cloud_key = os.getenv("BROWSERUSE_API_KEY", "")
        if self._cloud_key:
            self._transport: Any = _BrowserUseCloudTransport(self._cloud_key)
            self._transport_kind = "cloud"
        else:
            self._transport = _BrowserUseLocalTransport()
            self._transport_kind = "local"
        logger.info("CheckoutAgent transport: %s", self._transport_kind)

    async def checkout(
        self,
        cart_items: List[CartItem],
        cart_url: Optional[str] = None,
        scenario_id: Optional[str] = None,
        stop_before_payment: bool = True,
    ) -> CheckoutResult:
        if not cart_items:
            return CheckoutResult(success=False, error="empty cart")

        run_id = scenario_id or uuid.uuid4().hex[:10]
        run_dir = _ARTIFACTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        token = self._session_mgr.get_or_create()
        cookies = [{
            "name": "instacart_session",
            "value": token,
            "domain": ".instacart.com",
            "path": "/",
        }]

        task = self._build_task(cart_items, cart_url, stop_before_payment)
        (run_dir / "task.txt").write_text(task)

        try:
            result = await self._transport.run_task(
                task=task,
                cookies=cookies,
                run_dir=run_dir,
            )
        except Exception as exc:
            logger.exception("Checkout transport raised")
            result = CheckoutResult(success=False, error=str(exc))

        result.run_id = run_id
        result.artifact_dir = str(run_dir)
        if result.action_log:
            (run_dir / "action_log.json").write_text(json.dumps(result.action_log, indent=2))
        manifest = {
            "run_id": run_id,
            "transport": self._transport_kind,
            "items": [{"name": i.product.name, "qty": i.quantity} for i in cart_items],
            "screenshots": result.screenshots,
            "live_url": result.live_url,
            "cart_url": result.cart_url,
            "items_added": result.items_added,
            "success": result.success,
            "error": result.error,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return result

    def _build_task(
        self,
        cart_items: List[CartItem],
        cart_url: Optional[str],
        stop_before_payment: bool,
    ) -> str:
        items_str = "\n".join(
            f"- {item.product.name} x{item.quantity}"
            for item in cart_items
        )
        target = cart_url or "https://www.instacart.com"
        stop_note = (
            "When you reach the checkout review page, STOP. "
            "Do not enter any payment details, do not click 'Place Order'."
        ) if stop_before_payment else ""
        return f"""Open {target} in the browser.

For each of the following grocery items, use the search bar to find the product,
then click 'Add to cart' on the top in-stock result. After adding all items,
open the cart and proceed toward checkout.

Items to add:
{items_str}

{stop_note}

Return a JSON object with: items_added (integer), cart_url (string), stopped_at (string).
"""

    async def close(self) -> None:
        if hasattr(self._transport, "close"):
            await self._transport.close()


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

async def run_checkout(
    cart_items: List[CartItem],
    user_id: str = "default",
    cart_url: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    agent = CheckoutAgent(user_id=user_id)
    try:
        result = await agent.checkout(
            cart_items,
            cart_url=cart_url,
            scenario_id=scenario_id,
        )
        return {
            "success": result.success,
            "items_added": result.items_added,
            "cart_url": result.cart_url,
            "live_url": result.live_url,
            "screenshots": result.screenshots,
            "run_id": result.run_id,
            "artifact_dir": result.artifact_dir,
            "error": result.error,
        }
    finally:
        await agent.close()
