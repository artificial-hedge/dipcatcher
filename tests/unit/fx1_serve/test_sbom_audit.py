"""sbom_audit lane: release SBOM contract pinned."""

from __future__ import annotations

from fx1.sbom_audit import sbom_audit, sbom_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_sbom_contract() -> None:
    r = sbom_audit()
    assert r["n_entries"] == 2
    assert r["sorted"] is True
    assert r["hash_carried"] is True
    assert r["lockfile_sha"] is True
    assert r["missing_raises"] == "raise:FileNotFoundError"
    assert r["no_packages_raises"] == "raise:ValueError"
    assert r["half_block_raises"] == "raise:ValueError"  # the fixed hole
    assert r["injection_contained"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = sbom_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "sbom_audit_test.json")
    assert result["valid"], result.get("errors")
