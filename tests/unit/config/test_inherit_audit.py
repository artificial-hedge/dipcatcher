"""Inherit-audit lane: loader semantics pinned on a scratch tree."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.inherit_audit import inherit_audit, inherit_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_loader_semantics_on_scratch(tmp_path: Path) -> None:
    results = inherit_audit(tmp_path)
    assert results["same_dir_inherit"]["ok"] is True
    assert results["cycle"]["outcome"].startswith("raise")
    assert results["root_escape"]["outcome"].startswith("raise")
    assert results["parent_dir_inherit"]["outcome"].startswith("raise")
    assert results["list_semantics"]["overlay_list_replaces"] is True
    assert results["list_semantics"]["nested_merge_keeps_sibling"] is True


def test_bench_ok_and_verifies(tmp_path: Path) -> None:
    receipt = inherit_audit_bench(tmp_root=tmp_path / "scratch")
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "inherit_audit_test.json")
    assert result["valid"], result.get("errors")
