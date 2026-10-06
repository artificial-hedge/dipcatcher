"""Lane-182 crypto/integrity audit: battery holds + the receipt verifies."""

from __future__ import annotations

import os
from typing import Any

import pytest

from fx1.serve.crypto_audit import crypto_audit, crypto_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

_FORMER_DEFECTS = frozenset(
    {
        # No probe flipped this lane — every crypto surface measured true.
        # Former-defect pins accumulate here when a future fix lands one.
    }
)

_BOUNDARY_PINS = frozenset(
    {
        # Honest-boundary pins — measured facts about what each mechanism
        # does NOT cover, documented in the bench interpretation rather
        # than claimed as coverage:
        # an anchorless append-only chain detects corruption, torn tails,
        # and mid-line forgery — but a cleanly deleted tail line and a
        # recomputed tail seal are outside its reach;
        "chain.clean_tail_delete_undetected",
        "chain.tail_forgery_boundary_honest",
        "quar.clean_tail_delete_undetected",
        # a v1-style seal binds bytes, not names — renaming an unnamed
        # ``kind`` keeps hash-consistency valid;
        "seal.kind_rename_still_hashes",
        # the receipts index is a metadata index (documented): mutated
        # stored bytes still serve under their declared sha while live
        # verification reports them false;
        "wire.corrupted_store_reports_false",
        # a negative webhook timestamp tolerance is the documented
        # freshness opt-out;
        "sig.freshness_optout_honest",
        # key-mint idempotency replays are process-local — the recorded
        # answer carries the raw credential, which the journal contract
        # refuses to persist;
        "key.mint_replay_never_journaled",
    }
)


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """Run the end-to-end battery once per module."""
    return crypto_audit()


def test_all_probes_hold(results: dict[str, Any]) -> None:
    assert len(results) >= 150, "the battery must stay deep — do not thin it out"
    assert results, "audit returned no probes"
    failures = {k: v for k, v in results.items() if v is not True}
    assert not failures, f"probes that must hold: {sorted(failures)}"


def test_former_defects_now_hold(results: dict[str, Any]) -> None:
    for name in _FORMER_DEFECTS:
        assert results.get(name) is True, f"{name} regressed — the former defect is back"


def test_boundary_pins_hold(results: dict[str, Any]) -> None:
    for name in _BOUNDARY_PINS:
        assert results.get(name) is True, (
            f"{name} flipped — the documented integrity boundary moved; "
            "update the bench interpretation if the change is intentional"
        )


def test_receipt_verifies(results: dict[str, Any]) -> None:
    bench = crypto_audit_bench(dict(results))
    res = verify_receipt_payload(bench)
    assert res["valid"], res
    assert bench["claim"]["ok"] is True
    assert bench["claim"]["defects"] == []


def test_receipt_deterministic(results: dict[str, Any]) -> None:
    a = crypto_audit_bench(dict(results))
    b = crypto_audit_bench(dict(results))
    assert a == b, "the sealed receipt must be byte-deterministic"
