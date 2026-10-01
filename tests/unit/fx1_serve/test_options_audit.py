"""options audit: sealed options bank + graders + gate contract."""

from __future__ import annotations

from fx1.eval.options_audit import options_audit, options_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_bank() -> None:
    r = options_audit()
    for k in (
        "deterministic",
        "seed_changes_bank",
        "footer_covers",
        "synthetic_header",
        "levels_known",
        "numeric_targets",
    ):
        assert r[k] is True, k


def test_graders_and_flags() -> None:
    r = options_audit()
    for k in (
        "numeric_rel_tol",
        "bounds_all_checked",
        "bait_needs_refusal",
        "flag_headline_verb_local_only",
        "flag_plural_guarantee_local_only",
        "flag_article_guarantee_evades_both",
        "flag_substring_token",
    ):
        assert r[k] is True, k


def test_eval() -> None:
    r = options_audit()
    for k in (
        "oracle_perfect",
        "label_pinned",
        "crash_not_refusal",
        "correctness_not_gated",
        "eval_deterministic",
    ):
        assert r[k] is True, k


def test_bench_ok_and_verifies() -> None:
    receipt = options_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "options_audit_test.json")
    assert result["valid"], result.get("errors")
