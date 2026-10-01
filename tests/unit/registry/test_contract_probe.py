"""Tests for registry.contract_probe."""

from __future__ import annotations

from pathlib import Path

from quant_fund.registry.contract_probe import (
    contract_probe_bench,
    probe_receipt,
    scan_corpus,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload

_RECEIPTS = Path(__file__).resolve().parents[3] / "receipts"


def test_probe_detects_contracted_receipt() -> None:
    """A receipt under a deep contract must reject a forged claim."""
    fleet = sorted(_RECEIPTS.glob("fleet_eval*.json"))
    assert fleet, "no fleet_eval receipts committed"
    import json

    payload = json.loads(fleet[0].read_text())
    row = probe_receipt(payload, fleet[0])
    assert row["probes"], "no probes ran"
    assert row["coverage"] == "contracted"


def test_scan_corpus_covers_committed_receipts() -> None:
    rows = scan_corpus(_RECEIPTS)
    assert len(rows) == len(list(_RECEIPTS.glob("*.json")))
    assert all(
        r["coverage"] in ("contracted", "envelope_only", "unsealed", "unparseable") for r in rows
    )
    assert any(r["coverage"] == "contracted" for r in rows)


def test_bench_seals_and_verifies() -> None:
    receipt = contract_probe_bench(_RECEIPTS)
    assert receipt["schema"] == "contract_probe.v1"
    claim = receipt["claim"]
    assert claim["n_receipts"] == claim["n_contracted"] + claim["n_envelope_only"] + (
        claim["n_receipts"] - claim["n_contracted"] - claim["n_envelope_only"]
    )
    verdict = verify_receipt_payload(receipt, _RECEIPTS / "contract_probe.json")
    assert verdict["valid"], verdict["errors"]
