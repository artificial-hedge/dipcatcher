"""ts_reasoning audit: seeded bank + grader + gate contract."""

from __future__ import annotations

from fx1.eval.ts_reasoning_audit import ts_reasoning_audit, ts_reasoning_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_bank() -> None:
    r = ts_reasoning_audit()
    for k in (
        "bad_n_refuses",
        "deterministic",
        "seed_changes_bank",
        "footer_covers",
        "families_cover",
        "equal_shares",
        "synthetic_header",
    ):
        assert r[k] is True, k


def test_graders() -> None:
    r = ts_reasoning_audit()
    for k in (
        "ident_exact",
        "choice_single_digit",
        "numeric_tol",
        "flag_numeric_first_wins",
        "flag_bool_first_wins",
        "bool_exact",
        "honesty_on_graded",
        "canonical_passes",
    ):
        assert r[k] is True, k


def test_eval() -> None:
    r = ts_reasoning_audit()
    for k in (
        "oracle_perfect",
        "n_tasks",
        "eval_deterministic",
        "violating_closes_gate",
        "family_rates_bounded",
    ):
        assert r[k] is True, k


def test_bench_ok_and_verifies() -> None:
    receipt = ts_reasoning_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "ts_reasoning_audit_test.json")
    assert result["valid"], result.get("errors")
