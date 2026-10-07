"""Tests for the chat_signing_audit lane."""

from __future__ import annotations

import pytest

from fx1.serve.chat_signing_audit import (
    chat_signing_audit,
    chat_signing_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)


def test_all_probes_hold() -> None:
    results = chat_signing_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = chat_signing_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = chat_signing_audit_bench()
    b = chat_signing_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_result_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.chat_signing_audit.chat_signing_audit", lambda: {})
    blob = chat_signing_audit_bench()
    assert blob["claim"]["ok"] is False
    assert "no_probes" in blob["interpretation"]
