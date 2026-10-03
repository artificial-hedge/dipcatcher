"""calibration audit: extraction, bank determinism, oracle pass/fail."""

from __future__ import annotations

from fx1.eval.calibration_audit import (
    calibration_audit,
    calibration_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_extraction() -> None:
    r = calibration_audit()
    for k in (
        "extract_last_wins",
        "extract_pct_scales",
        "extract_oob_none",
        "extract_unparseable_none",
        "extract_exp",
    ):
        assert r[k] is True, k


def test_bank() -> None:
    r = calibration_audit()
    for k in (
        "bad_n_refuses",
        "bank_deterministic",
        "seed_changes_bank",
        "truths_bounded",
        "prompts_carry_label",
    ):
        assert r[k] is True, k


def test_eval() -> None:
    r = calibration_audit()
    assert r["bad_mode_refuses"] is True
    assert r["oracle_true_exact"] is True
    assert r["oracle_unknown_empty"] is True
    assert r["true_oracle_passes"] is True
    assert r["exploding_fails_closed"] is True
    assert r["constant_fails_closed"] is True
    assert r["bad_bins_refuses"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = calibration_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "calibration_audit_test.json")
    assert result["valid"], result.get("errors")
