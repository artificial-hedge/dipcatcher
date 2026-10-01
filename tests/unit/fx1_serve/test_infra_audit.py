"""train infra audit: config gates, curriculum determinism, cluster physics."""

from __future__ import annotations

from fx1.train.infra_audit import train_infra_audit, train_infra_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_config_gates() -> None:
    r = train_infra_audit()
    assert r["k3_gates"] is True
    assert r["k3_valid_ok"] is True
    assert r["bounds_enforced"] is True


def test_curriculum() -> None:
    r = train_infra_audit()
    assert r["classify_order"] is True
    assert r["curriculum_deterministic"] is True
    assert r["level_counts"] is True
    assert r["refusal_last"] is True


def test_cluster() -> None:
    r = train_infra_audit()
    for k in (
        "nodes_min2",
        "mxfp4_needs_zero3",
        "long_ctx_needs_8",
        "world_size",
        "deepspeed_carries_seed",
    ):
        assert r[k] is True, k


def test_bench_ok_and_verifies() -> None:
    receipt = train_infra_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "train_infra_audit_test.json")
    assert result["valid"], result.get("errors")
