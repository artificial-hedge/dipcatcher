"""Wave-1134 computational-math canon tests."""

from __future__ import annotations

from quant_fund.models.adaptive_method_theory import bench_adaptive_method_theory
from quant_fund.models.finite_element_theory import bench_finite_element_theory
from quant_fund.models.high_performance_numerics import bench_high_performance_numerics
from quant_fund.models.reduced_order_modeling import bench_reduced_order_modeling
from quant_fund.models.spectral_theory_numerics import bench_spectral_theory_numerics
from quant_fund.models.uncertainty_quantification_2 import bench_uncertainty_quantification_2


def test_finite_element_theory():
    assert bench_finite_element_theory()["synthetic_finite_element_theory"] == 1.0


def test_spectral_theory_numerics():
    assert bench_spectral_theory_numerics()["synthetic_spectral_theory_numerics"] == 1.0


def test_adaptive_method_theory():
    assert bench_adaptive_method_theory()["synthetic_adaptive_method_theory"] == 1.0


def test_reduced_order_modeling():
    assert bench_reduced_order_modeling()["synthetic_reduced_order_modeling"] == 1.0


def test_uncertainty_quantification_2():
    assert bench_uncertainty_quantification_2()["synthetic_uncertainty_quantification_2"] == 1.0


def test_high_performance_numerics():
    assert bench_high_performance_numerics()["synthetic_high_performance_numerics"] == 1.0
