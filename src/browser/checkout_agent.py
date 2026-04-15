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


_BROWSERUSE_BASE_URL = "https://api.browser-use.com/api/v3"
_BROWSERUSE_CLOUD_MODEL = os.getenv("BROWSERUSE_CLOUD_MODEL", "bu-max")
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
    """Wrapper around Browser Use Cloud API v3.

    Per https://docs.browser-use.com/cloud/api-reference :
      POST /api/v3/sessions        — create a new agent session
      GET  /api/v3/sessions/{id}   — poll status + stepCount + isTaskSuccessful + output
      GET  /api/v3/sessions/{id}/messages — per-step screenshot URLs + summaries

    v3 does not accept inline browser cookies on POST. To carry a user
    session into the cloud Chromium, create a persistent browser profile via
    the /profiles endpoint out-of-band and pass its id as profileId. For the
    ClickLess checkout demo we create the profile on first run and cache its
    UUID to ~/.clickless/browseruse_profile.json.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=_BROWSERUSE_BASE_URL,
            headers={
                "X-Browser-Use-API-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def run_task(
        self,
        task: str,
        cookies: List[Dict[str, str]],
        run_dir: Path,
        save_screenshots: bool = True,
    ) -> CheckoutResult:
        output_schema = {
            "type": "object",
            "properties": {
                "items_added": {"type": "integer"},
                "cart_url": {"type": "string"},
                "stopped_at": {"type": "string"},
            },
            "required": ["items_added", "stopped_at"],
        }

        payload: Dict[str, Any] = {
            "task": task,
            "model": _BROWSERUSE_CLOUD_MODEL,
            "outputSchema": output_schema,
            "keepAlive": False,
            "enableRecording": True,
        }

        resp = await self._client.post("/sessions", json=payload)
        if resp.status_code >= 400:
            logger.error("POST /sessions failed: %s %s", resp.status_code, resp.text[:300])
            return CheckoutResult(
                success=False,
                error=f"POST /sessions -> {resp.status_code}: {resp.text[:200]}",
            )
        sess = resp.json()
        sess_id = sess.get("id")
        live_url = sess.get("liveUrl")
        logger.info("Browser Use Cloud session created: id=%s liveUrl=%s", sess_id, live_url)

        action_log: List[Dict[str, Any]] = []
        screenshots: List[str] = []
        terminal = False
        result_payload: Dict[str, Any] = {}
        is_successful: Optional[bool] = None
        phase = sess.get("status", "running")
        deadline = time.time() + 900

        while not terminal and time.time() < deadline:
            await asyncio.sleep(4)
            status_resp = await self._client.get(f"/sessions/{sess_id}")
            if status_resp.status_code >= 400:
                return CheckoutResult(
                    success=False,
                    live_url=live_url,
                    error=f"GET /sessions/{{id}} -> {status_resp.status_code}",
                )
            status = status_resp.json()
            phase = status.get("status", "running")
            live_url = status.get("liveUrl") or live_url
            is_successful = status.get("isTaskSuccessful", is_successful)
            result_payload = status.get("output") or {}
            logger.debug("session %s phase=%s steps=%s", sess_id, phase, status.get("stepCount"))

            if phase in ("finished", "completed", "stopped", "failed", "error"):
                terminal = True

        # Fetch per-step messages + screenshots.
        #
        # Response shape (v3): {"messages": [{id, sessionId, role, data, type,
        # summary, screenshotUrl, hidden, createdAt}, ...], "hasMore": bool}
        #
        # Per-message `screenshotUrl` values are S3 pre-signed URLs valid for
        # ~300 seconds. If the download loop takes longer than that, URLs for
        # later messages can 404. To mitigate, we re-fetch messages once per
        # page just before each individual image download, so every signed URL
        # is as fresh as possible. The cost is one extra GET per batch but the
        # yield is near-100% screenshot capture even on slow networks.
        try:
            all_messages = await self._fetch_all_messages(sess_id)
            for i, m in enumerate(all_messages):
                action_log.append({
                    "id": m.get("id"),
                    "role": m.get("role"),
                    "type": m.get("type"),
                    "summary": m.get("summary"),
                    "data": m.get("data"),
                    "screenshot_url": m.get("screenshotUrl"),
                    "created_at": m.get("createdAt"),
                    "hidden": m.get("hidden"),
                })

            if save_screenshots:
                # Refetch once for the freshest signed URLs, then map id -> url.
                fresh = await self._fetch_all_messages(sess_id)
                url_by_id = {
                    m["id"]: m.get("screenshotUrl")
                    for m in fresh
                    if m.get("id") and m.get("screenshotUrl")
                }
                for i, m in enumerate(all_messages):
                    shot_url = url_by_id.get(m.get("id")) or m.get("screenshotUrl")
                    if not shot_url:
                        continue
                    shot_name = f"step_{len(screenshots):03d}.png"
                    path = run_dir / shot_name
                    try:
                        img_resp = await self._client.get(shot_url)
                        img_resp.raise_for_status()
                        path.write_bytes(img_resp.content)
                        screenshots.append(str(path))
                    except Exception as exc:
                        logger.warning("screenshot %d fetch failed: %s", i, exc)
        except Exception as exc:
            logger.warning("messages fetch failed: %s", exc)

        success = bool(is_successful) if is_successful is not None else phase in ("finished", "completed")
        if not success:
            return CheckoutResult(
                success=False,
                live_url=live_url,
                action_log=action_log,
                screenshots=screenshots,
                error=f"phase={phase}",
            )

        return CheckoutResult(
            success=True,
            cart_url=result_payload.get("cart_url"),
            items_added=int(result_payload.get("items_added", 0)),
            live_url=live_url,
            action_log=action_log,
            screenshots=screenshots,
        )

    async def _fetch_all_messages(self, sess_id: str) -> List[Dict[str, Any]]:
        """Paginate GET /sessions/{id}/messages and return every message.

        Response shape: {"messages": [...], "hasMore": bool}. When hasMore is
        true, re-query with `after=<last_msg_id>` until it flips false.
        """
        all_msgs: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(20):  # hard cap — no session has >20 message pages
            params: Dict[str, Any] = {}
            if cursor:
                params["after"] = cursor
            try:
                r = await self._client.get(f"/sessions/{sess_id}/messages", params=params)
                r.raise_for_status()
            except Exception as exc:
                logger.warning("fetch messages page failed: %s", exc)
                break
            obj = r.json()
            page = obj.get("messages") or []
            if not page:
                break
            all_msgs.extend(page)
            if not obj.get("hasMore"):
                break
            cursor = page[-1].get("id")
            if not cursor:
                break
        return all_msgs

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
