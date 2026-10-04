"""Wave-999 GMT-2 canon tests."""

from __future__ import annotations

from quant_fund.models.brakke_varifolds import bench_brakke_varifolds
from quant_fund.models.currents_theory import bench_currents_theory
from quant_fund.models.flat_chains import bench_flat_chains
from quant_fund.models.integral_currents import bench_integral_currents
from quant_fund.models.rectifiable_measures import bench_rectifiable_measures
from quant_fund.models.varifold_theory import bench_varifold_theory


def test_currents_theory():
    assert bench_currents_theory()["synthetic_currents_theory"] == 1.0


def test_varifold_theory():
    assert bench_varifold_theory()["synthetic_varifold_theory"] == 1.0


def test_flat_chains():
    assert bench_flat_chains()["synthetic_flat_chains"] == 1.0


def test_integral_currents():
    assert bench_integral_currents()["synthetic_integral_currents"] == 1.0


def test_rectifiable_measures():
    assert bench_rectifiable_measures()["synthetic_rectifiable_measures"] == 1.0


def test_brakke_varifolds():
    assert bench_brakke_varifolds()["synthetic_brakke_varifolds"] == 1.0
