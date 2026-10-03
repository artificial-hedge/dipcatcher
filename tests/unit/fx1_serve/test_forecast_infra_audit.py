"""forecast infra audit: signals bounds, registry fail-closed, guards."""

from __future__ import annotations

from fx1.forecast.infra_audit import (
    forecast_infra_audit,
    forecast_infra_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_signals() -> None:
    r = forecast_infra_audit()
    for k in (
        "unknown_mapping_refuses",
        "missing_score_refuses",
        "bad_threshold_refuses",
        "bounded",
        "role_stamped",
        "null_maps_null",
        "single_rank_zero",
    ):
        assert r[k] is True, k


def test_registry() -> None:
    r = forecast_infra_audit()
    for k in (
        "fx1_refuses_no_entrypoint",
        "malformed_entrypoint_refuses",
        "unresolvable_refuses",
        "fx1_entrypoint_typechecked",
        "builtins_resolve",
        "unknown_builtin_refuses",
    ):
        assert r[k] is True, k


def test_reference_models() -> None:
    r = forecast_infra_audit()
    for k in (
        "empty_frame_refuses",
        "missing_ids_refuses",
        "leakage_refuses",
        "forecast_schema_cols",
        "zero_predicts_zero",
        "momentum_needs_feature",
        "fx1_base_unimplemented",
    ):
        assert r[k] is True, k


def test_bench_ok_and_verifies() -> None:
    receipt = forecast_infra_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "forecast_infra_audit_test.json")
    assert result["valid"], result.get("errors")
