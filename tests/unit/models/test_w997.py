"""Wave-997 dispersive-PDE canon tests."""

from __future__ import annotations

from quant_fund.models.bilinear_estimates import bench_bilinear_estimates
from quant_fund.models.i_method import bench_i_method
from quant_fund.models.kdv_dispersion import bench_kdv_dispersion
from quant_fund.models.local_smoothing import bench_local_smoothing
from quant_fund.models.nls_dispersion import bench_nls_dispersion
from quant_fund.models.strichartz_estimates import bench_strichartz_estimates


def test_nls_dispersion():
    assert bench_nls_dispersion()["synthetic_nls_dispersion"] == 1.0


def test_kdv_dispersion():
    assert bench_kdv_dispersion()["synthetic_kdv_dispersion"] == 1.0


def test_strichartz_estimates():
    assert bench_strichartz_estimates()["synthetic_strichartz_estimates"] == 1.0


def test_local_smoothing():
    assert bench_local_smoothing()["synthetic_local_smoothing"] == 1.0


def test_bilinear_estimates():
    assert bench_bilinear_estimates()["synthetic_bilinear_estimates"] == 1.0


def test_i_method():
    assert bench_i_method()["synthetic_i_method"] == 1.0
