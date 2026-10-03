"""Wave-952 spectral-decomposition canon tests."""

from __future__ import annotations

from quant_fund.models.eigval_bounds import bench_eigval_bounds
from quant_fund.models.power_deflation import bench_power_deflation
from quant_fund.models.qr_iteration import bench_qr_iteration
from quant_fund.models.schur_decomp import bench_schur_decomp
from quant_fund.models.spectral_gap import bench_spectral_gap
from quant_fund.models.spectral_radius import bench_spectral_radius


def test_qr_iteration():
    assert bench_qr_iteration()["synthetic_qr_iteration"] == 1.0


def test_power_deflation():
    assert bench_power_deflation()["synthetic_power_deflation"] == 1.0


def test_schur_decomp():
    assert bench_schur_decomp()["synthetic_schur_decomp"] == 1.0


def test_eigval_bounds():
    assert bench_eigval_bounds()["synthetic_eigval_bounds"] == 1.0


def test_spectral_radius():
    assert bench_spectral_radius()["synthetic_spectral_radius"] == 1.0


def test_spectral_gap():
    assert bench_spectral_gap()["synthetic_spectral_gap"] == 1.0
