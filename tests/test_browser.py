"""Tests for browser checkout agent (smoke tests only -- real browser tests are slow)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.api.product_schema import CartItem, Product


def test_session_manager_create_and_retrieve(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_MASTER_PASSWORD", "test_password_123")
    # Patch vault path to tmp
    from src.llm import credential_vault as cv
    monkeypatch.setattr(cv, "_VAULT_PATH", tmp_path / "vault.enc")
    monkeypatch.setattr(cv, "_SALT_PATH", tmp_path / ".salt")
    cv._vault = None  # reset singleton

    from src.llm.session_manager import SessionManager
    mgr = SessionManager("user_test")
    token = mgr.create_session()
    assert token
    assert mgr.get_session() == token


def test_credential_vault_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_MASTER_PASSWORD", "hunter2")
    from src.llm import credential_vault as cv
    monkeypatch.setattr(cv, "_VAULT_PATH", tmp_path / "vault.enc")
    monkeypatch.setattr(cv, "_SALT_PATH", tmp_path / ".salt")

    vault = cv.CredentialVault()
    vault.set("api_key", "secret_xyz")
    assert vault.get("api_key") == "secret_xyz"

    # New instance loads encrypted data
    vault2 = cv.CredentialVault()
    assert vault2.get("api_key") == "secret_xyz"


def test_checkout_agent_builds_task():
    from src.browser.checkout_agent import CheckoutAgent
    agent = CheckoutAgent(user_id="test")
    p = Product(instacart_id="1", name="Milk", price=3.99)
    task = agent._build_task([CartItem(product=p, quantity=2)], "http://test/cart", stop_before_payment=True)
    assert "Milk" in task
    assert "x2" in task
    assert "payment" in task.lower()


def test_miniwob_report_empty():
    from src.browser.miniwob_eval import EvalReport
    report = EvalReport()
    assert report.success_rate == 0.0
    assert report.mean_reward == 0.0


def test_miniwob_report_metrics():
    from src.browser.miniwob_eval import EvalReport, TaskResult
    report = EvalReport(results=[
        TaskResult(task="a", success=True, reward=1.0),
        TaskResult(task="b", success=True, reward=0.8),
        TaskResult(task="c", success=False, reward=0.0),
    ])
    assert report.success_rate == pytest.approx(2 / 3)
    assert report.mean_reward == pytest.approx(0.6, rel=1e-3)
