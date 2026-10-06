"""Tests for fx1.serve.byok_audit + the OpenAICompatBackend contract."""

from __future__ import annotations

import os

import pytest

from fx1.serve import get_backend
from fx1.serve.byok_audit import byok_audit, byok_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_ENVS = (
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_BYOK_ALLOW_PRIVATE_NETWORKS",
)


def test_all_probes_hold() -> None:
    results = byok_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = byok_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = byok_audit_bench()
    b = byok_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_byok_missing_env_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="FX1_BYOK_"):
        get_backend("byok")


def test_byok_constructs_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_BYOK_BASE_URL", "https://example.local/v1")
    monkeypatch.setenv("FX1_BYOK_API_KEY", "k")
    monkeypatch.setenv("FX1_BYOK_MODEL", "m")
    backend = get_backend("byok")
    assert backend is not None
    assert os.environ.get("FX1_BYOK_API_KEY") == "k"  # sanity: fixture-scoped
