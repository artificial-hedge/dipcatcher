"""Wave-1002 turbulence canon tests."""

from __future__ import annotations

from quant_fund.models.energy_spectrum import bench_energy_spectrum
from quant_fund.models.intermittency_models import bench_intermittency_models
from quant_fund.models.kolmogorov_theory import bench_kolmogorov_theory
from quant_fund.models.reynolds_decomp import bench_reynolds_decomp
from quant_fund.models.taylor_series_hyp import bench_taylor_series_hyp
from quant_fund.models.wall_turbulence import bench_wall_turbulence


def test_kolmogorov_theory():
    assert bench_kolmogorov_theory()["synthetic_kolmogorov_theory"] == 1.0


def test_reynolds_decomp():
    assert bench_reynolds_decomp()["synthetic_reynolds_decomp"] == 1.0


def test_energy_spectrum():
    assert bench_energy_spectrum()["synthetic_energy_spectrum"] == 1.0


def test_intermittency_models():
    assert bench_intermittency_models()["synthetic_intermittency_models"] == 1.0


def test_wall_turbulence():
    assert bench_wall_turbulence()["synthetic_wall_turbulence"] == 1.0


def test_taylor_series_hyp():
    assert bench_taylor_series_hyp()["synthetic_taylor_series_hyp"] == 1.0
