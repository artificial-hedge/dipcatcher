"""Wave-1133 physics-4 canon tests."""

from __future__ import annotations

from quant_fund.models.conformal_field_theory import bench_conformal_field_theory
from quant_fund.models.holography_ads import bench_holography_ads
from quant_fund.models.lattice_field_theory import bench_lattice_field_theory
from quant_fund.models.loop_quantum_gravity import bench_loop_quantum_gravity
from quant_fund.models.statistical_field_theory import bench_statistical_field_theory
from quant_fund.models.string_theory_math import bench_string_theory_math


def test_statistical_field_theory():
    assert bench_statistical_field_theory()["synthetic_statistical_field_theory"] == 1.0


def test_conformal_field_theory():
    assert bench_conformal_field_theory()["synthetic_conformal_field_theory"] == 1.0


def test_lattice_field_theory():
    assert bench_lattice_field_theory()["synthetic_lattice_field_theory"] == 1.0


def test_string_theory_math():
    assert bench_string_theory_math()["synthetic_string_theory_math"] == 1.0


def test_loop_quantum_gravity():
    assert bench_loop_quantum_gravity()["synthetic_loop_quantum_gravity"] == 1.0


def test_holography_ads():
    assert bench_holography_ads()["synthetic_holography_ads"] == 1.0
