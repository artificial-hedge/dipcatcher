"""docs audit: the committed docs match the served surface."""

from __future__ import annotations

import json
from pathlib import Path

from fx1.serve.docs_audit import docs_audit, docs_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_REPO = Path(__file__).resolve().parents[3]


def test_all_probes_hold() -> None:
    r = docs_audit()
    bad = {k: v for k, v in r.items() if v is not True}
    assert not bad, bad


def test_openapi_golden_matches_served_spec() -> None:
    from fx1.serve.api import create_app

    regen = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    committed = (_REPO / "clients" / "typescript" / "fx1" / "openapi.json").read_text()
    assert committed == regen


def test_receipt_verifies() -> None:
    receipt = docs_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "fx1_docs_audit_test.json")
    assert result["valid"], result.get("errors")


def test_receipt_deterministic() -> None:
    a = docs_audit_bench()
    b = docs_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert a["claim"]["results"] == b["claim"]["results"]
