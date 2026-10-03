"""cli audit: fx1 front-door contract."""

from __future__ import annotations

from fx1.cli_audit import cli_audit, cli_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_routing() -> None:
    r = cli_audit()
    for k in (
        "bare_exits_clean",
        "help_lists",
        "unknown_command_fails",
        "describe_unknown_fails",
    ):
        assert r[k] is True, k


def test_param_gate_and_judge() -> None:
    r = cli_audit()
    for k in (
        "fetch_array_refused",
        "fetch_scalar_refused",
        "fetch_junk_refused",
        "judge_none_ok",
        "judge_bogus_exits2",
        "judge_bogus_code2",
    ):
        assert r[k] is True, k


def test_output_contracts() -> None:
    r = cli_audit()
    assert r["dipbench_synthetic_labeled"] is True
    assert r["doctor_json_flags"] is True


def test_surface() -> None:
    r = cli_audit()
    assert r["expected_commands"] is True
    assert r["flag_maskedeval_wart"] is True
    assert r["groups_registered"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = cli_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "cli_audit_test.json")
    assert result["valid"], result.get("errors")
