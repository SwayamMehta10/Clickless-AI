"""Tests for the Instacart API client."""

from __future__ import annotations

import pytest

from src.api.product_schema import CartItem, NutriScore, Product


def test_product_schema_defaults():
    p = Product(instacart_id="123", name="Organic Milk")
    assert p.availability is True
    assert p.nutriscore == NutriScore.UNKNOWN
    assert p.allergens == []


def test_product_short_description():
    p = Product(instacart_id="1", name="Whole Milk", brand="Horizon", price=3.99)
    desc = p.short_description()
    assert "Whole Milk" in desc
    assert "Horizon" in desc
    assert "$3.99" in desc


def test_cart_item_line_total():
    p = Product(instacart_id="1", name="Eggs", price=4.50)
    item = CartItem(product=p, quantity=3)
    assert item.line_total == pytest.approx(13.50)


def test_cart_item_no_price():
    p = Product(instacart_id="1", name="Eggs")
    item = CartItem(product=p, quantity=2)
    assert item.line_total is None


@pytest.mark.asyncio
async def test_search_products():
    from src.api.instacart_client import InstacartClient
    client = InstacartClient()
    results = await client.search_products("milk", limit=5)
    assert len(results) >= 1
    assert all(isinstance(p, Product) for p in results)


@pytest.mark.asyncio
async def test_search_with_max_price():
    from src.api.instacart_client import InstacartClient
    client = InstacartClient()
    results = await client.search_products("milk", max_price=3.0, limit=10)
    for p in results:
        if p.price is not None:
            assert p.price <= 3.0


@pytest.mark.asyncio
async def test_get_retailers():
    from src.api.instacart_client import InstacartClient
    client = InstacartClient()
    retailers = await client.get_retailers("85281")
    assert len(retailers) >= 1


@pytest.mark.asyncio
async def test_create_cart():
    from src.api.instacart_client import InstacartClient
    client = InstacartClient()
    p = Product(instacart_id="1", name="Milk", price=3.99)
    items = [CartItem(product=p, quantity=2)]
    result = await client.create_cart(items)
    assert "cart_id" in result
    assert "cart_url" in result
