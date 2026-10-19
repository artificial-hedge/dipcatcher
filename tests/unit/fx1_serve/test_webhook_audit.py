"""Tests for fx1.serve.webhook_audit — the signed-delivery contract lane."""

from __future__ import annotations

import pytest

from fx1.serve.webhook_audit import webhook_audit, webhook_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = webhook_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = webhook_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = webhook_audit_bench()
    b = webhook_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_injected_wall_clock_controls_staleness_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fx1.serve import webhook_audit as audit_mod
    from fx1.serve.webhooks import sign_webhook, verify_webhook

    ts = "1700000000"
    body = b'{"status":"ok"}'
    sig = sign_webhook(audit_mod._SECRET, ts, body)
    monkeypatch.setattr(audit_mod, "_WALL_CLOCK", lambda: 1_700_000_000.0)
    assert (
        verify_webhook(audit_mod._SECRET, ts, sig, body, now=audit_mod._wall_time() + 600.0)
        is False
    )
    assert (
        verify_webhook(
            audit_mod._SECRET,
            ts,
            sig,
            body,
            tolerance_s=-1.0,
            now=audit_mod._wall_time() + 10**9,
        )
        is True
    )
