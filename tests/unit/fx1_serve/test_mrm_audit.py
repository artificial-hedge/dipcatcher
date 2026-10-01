"""mrm_audit lane: dossier compiler contract pinned."""

from __future__ import annotations

from fx1.mrm_audit import mrm_audit, mrm_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_mrm_contract() -> None:
    r = mrm_audit()
    assert r["full_dossier"]["complete"] is True
    assert r["full_dossier"]["ship_eligible"] is True
    assert r["flagged_ship_vetoed"] is True
    assert r["laundered_flagged"] is True  # the laundering fix
    assert r["laundered_ship_vetoed"] is True
    assert r["malformed_declared_flagged"] is True
    assert r["missing_flag_declared_flagged"] is True
    assert r["incomplete_vetoes_ship"] is True
    assert r["missing_artifact_raises"] == "raise:FileNotFoundError"
    assert r["artifact_hashes_real"] is True
    assert r["failed_card_vetoed"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = mrm_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "mrm_audit_test.json")
    assert result["valid"], result.get("errors")
