"""routing audit: the served route surface is fully claimed by audit batteries."""

from __future__ import annotations

from fx1.serve.routing_audit import (
    AUDIT_ROUTE_MANIFEST,
    audit_coverage,
    routing_audit,
    routing_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    r = routing_audit()
    bad = {k: v for k, v in r.items() if v is not True}
    assert not bad, bad


def test_manifest_covers_every_audit_module() -> None:
    coverage = audit_coverage()
    assert coverage["gaps"] == []
    for mod, entries in AUDIT_ROUTE_MANIFEST.items():
        assert isinstance(entries, frozenset), mod
        for entry in entries:
            method, _, path = entry.partition(" ")
            assert method.isupper() and path.startswith("/"), entry


def test_receipt_verifies() -> None:
    receipt = routing_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "fx1_routing_audit_test.json")
    assert result["valid"], result.get("errors")


def test_receipt_deterministic() -> None:
    a = routing_audit_bench()
    b = routing_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert a["claim"]["results"] == b["claim"]["results"]
