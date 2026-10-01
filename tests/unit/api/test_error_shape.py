"""Tests for quant_fund.api.error_shape."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi.testclient")

from quant_fund.api.error_shape import error_shape_audit, error_shape_bench


def test_surface_clean():
    r = error_shape_audit()
    assert r["n_routes"] >= 10
    assert r["n_probes"] > 100
    assert r["n_violations"] == 0
    assert r["verdict"] == "ok"


def test_violations_recorded_when_present():
    r = error_shape_audit()
    for v in r["violations"]:
        assert not v["ok"]
        assert v["problems"]


def test_bench_sealed():
    r = error_shape_bench()
    assert r["schema"] == "error_shape.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r["data_label"] == "SYNTHETIC"
