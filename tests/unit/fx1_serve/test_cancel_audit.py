"""Lane-166 cancel audit: battery holds + the receipt verifies."""

from __future__ import annotations

import os
from typing import Any

import pytest

from fx1.serve.cancel_audit import cancel_audit, cancel_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_FORMER_DEFECTS = frozenset(
    {
        # a cancelled background response resurrected through the keyed
        # replay of its submit: delete + Idempotency-Key retry rehydrated
        # the idem cache's stale snapshot — 'queued' zombie when the
        # worker never started, 'completed' when the cancel landed
        # mid-flight (verdict flipped). _sync_idem_verdict now pins the
        # live/cancelled verdict into the cache and repin refuses to
        # rehydrate non-terminal responses.
        "resp.cancel_delete_replay_no_zombie",
        "resp.cancel_idem_replay_keeps_verdict",
    }
)


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    return cancel_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 90, "the battery must stay deep — do not thin it out"
    assert results, "audit returned no probes"
    failures = {k: v for k, v in results.items() if v is not True}
    assert not failures, f"probes that must hold: {sorted(failures)}"


def test_former_defects_now_hold(results: dict[str, Any]) -> None:
    for name in _FORMER_DEFECTS:
        assert results.get(name) is True, f"{name} regressed — the former defect is back"


def test_receipt_verifies() -> None:
    bench = cancel_audit_bench()
    res = verify_receipt_payload(bench)
    assert res["valid"], res
    assert bench["claim"]["ok"] is True
    assert bench["claim"]["defects"] == []


def test_receipt_deterministic() -> None:
    a = cancel_audit_bench()
    b = cancel_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"], (
        "receipt must be deterministic — fix flaky probes, never the test"
    )


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_cancel_audit.json")
    if not path.is_file():
        pytest.skip("committed receipt not present in this checkout")
    doc = json.loads(path.read_text())
    res = verify_receipt_file(str(path))
    assert res["valid"] is True and res["errors"] == [], res
    assert doc["claim"]["ok"] is True
