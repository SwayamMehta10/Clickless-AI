"""Instacart Connect API client.

Wraps the Instacart Developer Platform API:
https://docs.instacart.com/developer_platform_api/

All methods return canonical Product objects.

Transport selection:
- When offline_catalog_mode is enabled (or no API key is provisioned in the
  environment), the client routes requests through the local cache backend
  defined in src.api._instacart_backend, which serves the same canonical
  product shape from the cached Instacart 2017 + Open Food Facts tables.
- Otherwise, requests go over HTTPS to the Instacart Developer Platform.

In both modes, callers see an identical async interface and identical
canonical Product / cart payloads.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from src.api.product_schema import CartItem, NutriScore, Product
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class InstacartClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url = cfg["instacart"].get("base_url", "https://connect.dev.instacart.tools/idp/v1")
        self._api_key = os.getenv("INSTACART_API_KEY", "")
        self._timeout = cfg["instacart"].get("timeout", 30)
        self._max_retries = cfg["instacart"].get("max_retries", 3)
        self._backoff = cfg["instacart"].get("retry_backoff", 2.0)
        self._per_query = cfg["instacart"].get("results_per_query", 10)

        self._offline_mode = (
            cfg.get("app", {}).get("offline_catalog_mode", True)
            or not self._api_key
        )
        if self._offline_mode:
            from src.api._instacart_backend import LocalCatalogBackend
            self._local = LocalCatalogBackend()
            self._client = None
            logger.info("InstacartClient: local cache transport active")
        else:
            self._local = None
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"InstacartAPI {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            logger.info("InstacartClient: live developer platform transport active")

    async def search_products(
        self,
        query: str,
        zip_code: Optional[str] = None,
        limit: int = 10,
        dietary_flags: Optional[List[str]] = None,
        max_price: Optional[float] = None,
    ) -> List[Product]:
        """Search for products matching a query."""
        if self._offline_mode:
            return await self._local.search_products(
                query=query,
                zip_code=zip_code,
                limit=limit,
                dietary_flags=dietary_flags,
                max_price=max_price,
            )

        params: Dict[str, Any] = {"q": query, "limit": limit}
        if zip_code:
            params["zip_code"] = zip_code

        resp = await self._get("/products/search", params=params)
        products = [self._parse_product(p) for p in resp.get("products", [])]

        if max_price:
            products = [p for p in products if p.price is None or p.price <= max_price]
        if dietary_flags:
            products = self._filter_dietary(products, dietary_flags)

        return products

    async def get_product_details(self, product_id: str) -> Optional[Product]:
        """Get full details for a single product."""
        if self._offline_mode:
            return await self._local.get_product_details(product_id)

        try:
            resp = await self._get(f"/products/{product_id}")
            return self._parse_product(resp)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def get_retailers(self, zip_code: str) -> List[Dict]:
        """Get available retailers for a zip code."""
        if self._offline_mode:
            return await self._local.get_retailers(zip_code)
        resp = await self._get("/retailers", params={"zip_code": zip_code})
        return resp.get("retailers", [])

    async def create_cart(self, items: List[CartItem], retailer_id: str = "wfm-001") -> Dict:
        """Create a cart on Instacart. Returns cart URL and ID."""
        if self._offline_mode:
            return await self._local.create_cart(items, retailer_id=retailer_id)

        payload = {
            "retailer_id": retailer_id,
            "items": [
                {"product_id": i.product.instacart_id, "quantity": i.quantity}
                for i in items
            ],
        }
        return await self._post("/carts", json=payload)

    async def _get(self, path: str, **kwargs) -> Dict:
        import asyncio
        last_exc = None
        for attempt in range(self._max_retries):
            try:
                r = await self._client.get(path, **kwargs)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_exc = exc
            except httpx.RequestError as exc:
                last_exc = exc
            await asyncio.sleep(self._backoff ** attempt)
        raise RuntimeError(f"GET {path} failed after {self._max_retries} retries") from last_exc

    async def _post(self, path: str, **kwargs) -> Dict:
        r = await self._client.post(path, **kwargs)
        r.raise_for_status()
        return r.json()

    def _parse_product(self, data: Dict) -> Product:
        return Product(
            instacart_id=str(data.get("id", "")),
            name=data.get("name", ""),
            brand=data.get("brand"),
            price=data.get("price") or data.get("price_cents", 0) / 100 if data.get("price_cents") else None,
            availability=data.get("available", True),
            platform="instacart",
            category=data.get("category"),
            image_url=data.get("image_url"),
            aisle=data.get("aisle"),
            department=data.get("department"),
        )

    @staticmethod
    def _filter_dietary(products: List[Product], flags: List[str]) -> List[Product]:
        """Best-effort dietary filtering based on allergens and name."""
        filtered = []
        for p in products:
            ok = True
            name_lower = (p.name or "").lower()
            allergens_lower = [a.lower() for a in p.allergens]
            for flag in flags:
                flag_l = flag.lower()
                if flag_l == "gluten-free" and "gluten" in allergens_lower:
                    ok = False
                elif flag_l == "vegan" and any(a in allergens_lower for a in ["milk", "eggs", "honey"]):
                    ok = False
                elif flag_l == "organic" and "organic" not in name_lower:
                    pass  # Don't hard-filter; just lower priority in ranking
            if ok:
                filtered.append(p)
        return filtered

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


def get_client() -> "InstacartClient":
    """Top-level factory used by orchestration agents."""
    return InstacartClient()
