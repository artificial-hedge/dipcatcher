"""Wave-913 interpolation-3 canon tests."""

from __future__ import annotations

from quant_fund.models.akima_interp import bench_akima_interp
from quant_fund.models.makima_interp import bench_makima_interp
from quant_fund.models.monotone_interp import bench_monotone_interp
from quant_fund.models.pchip_interp import bench_pchip_interp
from quant_fund.models.scattered_interp import bench_scattered_interp
from quant_fund.models.spline_interp import bench_spline_interp


def test_scattered_interp():
    assert bench_scattered_interp()["synthetic_scattered_interp"] == 1.0


def test_spline_interp():
    assert bench_spline_interp()["synthetic_spline_interp"] == 1.0


def test_monotone_interp():
    assert bench_monotone_interp()["synthetic_monotone_interp"] == 1.0


def test_akima_interp():
    assert bench_akima_interp()["synthetic_akima_interp"] == 1.0


def test_pchip_interp():
    assert bench_pchip_interp()["synthetic_pchip_interp"] == 1.0


def test_makima_interp():
    assert bench_makima_interp()["synthetic_makima_interp"] == 1.0
