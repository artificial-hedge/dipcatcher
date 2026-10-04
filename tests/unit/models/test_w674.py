from quant_fund.models.analytic_spec import bench_analytic_spec
from quant_fund.models.derived_k3 import bench_derived_k3
from quant_fund.models.equivariant_spec import (
    bench_equivariant_spec,
)
from quant_fund.models.graded_spec import bench_graded_spec
from quant_fund.models.spectral_curve import bench_spectral_curve
from quant_fund.models.spectral_gm import bench_spectral_gm


def test_derived_k3():
    assert bench_derived_k3()["synthetic_derived_k3"] == 1.0


def test_spectral_gm():
    assert bench_spectral_gm()["synthetic_spectral_gm"] == 1.0


def test_analytic_spec():
    assert bench_analytic_spec()["synthetic_analytic_spec"] == 1.0


def test_graded_spec():
    assert bench_graded_spec()["synthetic_graded_spec"] == 1.0


def test_equivariant_spec():
    assert bench_equivariant_spec()["synthetic_equivariant_spec"] == 1.0


def test_spectral_curve():
    assert bench_spectral_curve()["synthetic_spectral_curve"] == 1.0
