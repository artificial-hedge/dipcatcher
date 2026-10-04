"""Tests for fx1.serve.spec_audit — vendor-spec conformance lane."""

from __future__ import annotations

import pytest

from fx1.serve.spec_audit import (
    _require_shape,
    spec_audit,
    spec_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = spec_audit()
    bad = {k: v for k, v in results.items() if v is not True}
    assert not bad, bad


def test_receipt_verifies() -> None:
    blob = spec_audit_bench()
    assert blob["claim"]["ok"] is True
    assert blob["claim"]["probes"] == len(blob["claim"]["results"])
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = spec_audit_bench()
    b = spec_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert a["claim"]["results"] == b["claim"]["results"]


def test_require_shape_unit() -> None:
    schema = {"a": int, "b?": str, "c": [{"d": frozenset({"x"})}]}
    assert _require_shape({"a": 1, "c": [{"d": "x"}]}, schema) is None
    assert _require_shape({"c": []}, schema) == "$.a: missing required key"
    assert _require_shape({"a": True, "c": []}, schema) is not None  # bool is not int
    assert _require_shape({"a": 1, "c": [{"d": "z"}]}, schema) is not None
    assert _require_shape([], schema) is not None
