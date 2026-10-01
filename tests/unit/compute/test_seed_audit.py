"""seed_audit lane: parallel-seeding contract pinned."""

from __future__ import annotations

from quant_fund.compute.parallel import derive_seed
from quant_fund.compute.seed_audit import seed_audit, seed_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_derive_seed_pure_and_injective() -> None:
    assert derive_seed(7, 0) == derive_seed(7, 0)
    assert derive_seed(7, 0) != derive_seed(7, 1)
    assert derive_seed(8, 0) != derive_seed(7, 0)


def test_seed_audit_all_checks_pass() -> None:
    results = seed_audit()
    assert results["order_independence"]["ok"] is True
    assert results["collision_free"]["ok"] is True
    assert results["avalanche"]["ok"] is True
    assert results["base_sensitivity"]["ok"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = seed_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "seed_audit_test.json")
    assert result["valid"], result.get("errors")
