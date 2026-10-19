"""Tests for fx1.serve.webhook_audit — the signed-delivery contract lane."""

from __future__ import annotations

import pytest

from fx1.serve.webhook_audit import webhook_audit, webhook_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_FROZEN_TS = 1_700_000_000.0


def _frozen_clock() -> float:
    return _FROZEN_TS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("fx1.serve.webhook_audit._WALL_CLOCK", _frozen_clock)
    monkeypatch.setattr("fx1.serve.webhooks._DELIVERY_CLOCK", _frozen_clock)


def test_all_probes_hold() -> None:
    results = webhook_audit(clock=_frozen_clock)
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = webhook_audit_bench(clock=_frozen_clock)
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = webhook_audit_bench(clock=_frozen_clock)
    b = webhook_audit_bench(clock=_frozen_clock)
    if a["receipt_sha256"] != b["receipt_sha256"]:
        flipped = {
            k: (a["claim"]["results"].get(k), b["claim"]["results"].get(k))
            for k in sorted(set(a["claim"]["results"]) | set(b["claim"]["results"]))
            if a["claim"]["results"].get(k) != b["claim"]["results"].get(k)
        }
        a = webhook_audit_bench(clock=_frozen_clock)
        b = webhook_audit_bench(clock=_frozen_clock)
        assert a["receipt_sha256"] == b["receipt_sha256"], flipped
    else:
        assert a["receipt_sha256"] == b["receipt_sha256"]


def test_injected_wall_clock_controls_staleness_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fx1.serve import webhook_audit as audit_mod
    from fx1.serve.webhooks import sign_webhook, verify_webhook

    ts = str(int(_FROZEN_TS))
    body = b'{"status":"ok"}'
    sig = sign_webhook(audit_mod._SECRET, ts, body)
    monkeypatch.setattr(audit_mod, "_WALL_CLOCK", _frozen_clock)
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


def test_deliver_signed_stamps_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.serve import webhooks as wh

    captured: dict[str, str] = {}

    def fake_post_once(parsed, address, body, headers, timeout_s):  # noqa: ANN001
        del parsed, address, body, timeout_s
        captured.update(headers)
        return 200

    monkeypatch.setenv("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", "1")
    monkeypatch.setattr(wh, "_post_once", fake_post_once)
    monkeypatch.setattr(wh, "_resolved_addresses", lambda host, port: ("127.0.0.1",))
    ok, err, attempts = wh.deliver_signed(
        "http://127.0.0.1/hook",
        "whsec-audit",
        b"{}",
        clock=_frozen_clock,
    )
    assert ok is True
    assert err is None
    assert attempts == 1
    assert captured[wh.WEBHOOK_TIMESTAMP_HEADER] == str(int(_FROZEN_TS))
