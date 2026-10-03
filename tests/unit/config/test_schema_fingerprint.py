"""Tests for quant_fund.config.schema_fingerprint."""

from __future__ import annotations

from pydantic import BaseModel

from quant_fund.config.models import AppConfig
from quant_fund.config.schema_fingerprint import (
    _type_name,
    compat_report,
    model_schema_tree,
    schema_fingerprint,
    schema_fingerprint_bench,
)


class _Inner(BaseModel):
    depth: int
    label: str = "x"


class _Outer(BaseModel):
    inner: _Inner
    weight: float
    note: str | None = None


def test_tree_dotted_paths():
    tree = model_schema_tree(_Outer)
    assert set(tree) == {"inner.depth", "inner.label", "weight", "note"}
    assert tree["inner.depth"]["required"] is True
    assert tree["inner.label"]["has_default"] is True
    assert tree["weight"]["type"] == "float"


def test_fingerprint_stable_and_sensitive():
    fp1 = schema_fingerprint(_Outer)
    fp2 = schema_fingerprint(_Outer)
    assert fp1 == fp2 and len(fp1) == 64

    class _Outer2(_Outer):
        pass

    class _Outer3(BaseModel):
        inner: _Inner
        weight: float
        note: str | None = None
        added: int = 0

    assert schema_fingerprint(_Outer3) != fp1


def test_type_name_union_and_model():
    assert "union[" in _type_name(int | None)
    assert _type_name(int) == "int"


def test_compat_clean():
    report = compat_report({"inner": {"depth": 3, "label": "y"}, "weight": 1.5}, _Outer)
    assert report["verdict"] == "loads_cleanly"
    assert report["extra_keys"] == []
    assert report["missing_required"] == []


def test_compat_extra_and_missing():
    report = compat_report({"inner": {"depth": 1, "bogus": 0}}, _Outer)
    assert "inner.bogus" in report["extra_keys"]
    assert "weight" in report["missing_required"]
    assert report["verdict"] == "missing_required"


def test_compat_type_drift():
    report = compat_report({"inner": {"depth": "seven"}, "weight": 1.0}, _Outer)
    assert "inner.depth" in report["type_mismatches"]
    assert report["verdict"] == "type_drift"


def test_appconfig_fingerprint_size():
    fp = schema_fingerprint(AppConfig)
    tree = model_schema_tree(AppConfig)
    assert len(fp) == 64 and len(tree) > 50


def test_bench_sealed_and_deterministic():
    r1 = schema_fingerprint_bench()
    r2 = schema_fingerprint_bench()
    assert r1["schema"] == "schema_fingerprint.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1 == r2
