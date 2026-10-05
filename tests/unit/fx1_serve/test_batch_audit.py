"""Lane-153 batch audit: battery holds + the receipt verifies."""

from __future__ import annotations

import os

import pytest

from fx1.serve.batch_audit import batch_audit, batch_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_FORMER_DEFECTS = frozenset(
    {
        # batch lines ran unattributed: the worker never saw the
        # submitter's key_id, so records landed under "(none)" and the
        # managed key's token meter never charged — now re-installed on
        # the record and set by the worker per line (both dialects).
        "batch_lines_attribute_key_id",
        "batch_lines_charge_tokens",
        "abatch_lines_attribute_key_id",
        # expiry-on-read never signalled the worker: the tail of the
        # input kept running past expires_at and the worker's terminal
        # write could overwrite ``expired`` — the projection now sets
        # _cancel and every status write guards on already-terminal.
        "expired_tail_never_ran",
        "expired_never_overwritten",
    }
)


@pytest.fixture(scope="module")
def results() -> dict[str, object]:
    return batch_audit()


def test_all_probes_hold(results: dict[str, object]) -> None:
    assert len(results) >= 80, "the battery must stay deep — do not thin it out"
    assert results, "audit returned no probes"
    failures = {k: v for k, v in results.items() if v is not True}
    assert not failures, f"probes that must hold: {sorted(failures)}"


def test_former_defects_now_hold(results: dict[str, object]) -> None:
    for name in _FORMER_DEFECTS:
        assert results.get(name) is True, f"{name} regressed — the former defect is back"


def test_receipt_verifies() -> None:
    bench = batch_audit_bench()
    res = verify_receipt_payload(bench)
    assert res["valid"], res
    assert bench["claim"]["ok"] is True
    assert bench["claim"]["defects"] == []


def test_receipt_deterministic() -> None:
    a = batch_audit_bench()
    b = batch_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"], (
        "receipt must be deterministic — fix flaky probes, never the test"
    )
