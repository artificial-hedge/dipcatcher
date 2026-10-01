"""eval core audit: compare stats, suite gate, redteam bank."""

from __future__ import annotations

from fx1.eval.core_audit import eval_core_audit, eval_core_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_compare() -> None:
    r = eval_core_audit()
    for k in (
        "len_mismatch_refuses",
        "empty_refuses",
        "bootstrap_deterministic",
        "identical_not_significant",
        "mcnemar_zero_discordant",
        "mcnemar_continuity",
    ):
        assert r[k] is True, k


def test_suite_gate() -> None:
    r = eval_core_audit()
    assert r["exempt_task_passes_but_records"] is True
    assert r["honesty_gate_clean"] is True
    assert r["exempt_violation_closes_gate"] is True
    assert r["empty_honesty_fails"] is True
    assert r["bank_digest_binds"] is True


def test_redteam() -> None:
    r = eval_core_audit()
    assert r["redteam_count"] is True
    assert r["redteam_regexes_compile"] is True
    assert r["redteam_no_collision"] is True
    assert r["flag_spelled_digits_evade"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = eval_core_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "eval_core_audit_test.json")
    assert result["valid"], result.get("errors")
