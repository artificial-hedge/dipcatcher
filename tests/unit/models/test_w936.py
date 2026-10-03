"""Wave-936 nonsmooth-analysis canon tests."""

from __future__ import annotations

from quant_fund.models.bundle_level import bench_bundle_level
from quant_fund.models.clarke_subdiff import bench_clarke_subdiff
from quant_fund.models.epigraph_proj import bench_epigraph_proj
from quant_fund.models.gauge_duality import bench_gauge_duality
from quant_fund.models.gauge_fn import bench_gauge_fn
from quant_fund.models.subdiff_compute import bench_subdiff_compute


def test_subdiff_compute():
    assert bench_subdiff_compute()["synthetic_subdiff_compute"] == 1.0


def test_epigraph_proj():
    assert bench_epigraph_proj()["synthetic_epigraph_proj"] == 1.0


def test_gauge_fn():
    assert bench_gauge_fn()["synthetic_gauge_fn"] == 1.0


def test_gauge_duality():
    assert bench_gauge_duality()["synthetic_gauge_duality"] == 1.0


def test_bundle_level():
    assert bench_bundle_level()["synthetic_bundle_level"] == 1.0


def test_clarke_subdiff():
    assert bench_clarke_subdiff()["synthetic_clarke_subdiff"] == 1.0
