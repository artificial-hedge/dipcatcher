from quant_fund.models.comparison_thm import bench_comparison_thm
from quant_fund.models.jacobi_field import bench_jacobi_field
from quant_fund.models.levi_civita import bench_levi_civita
from quant_fund.models.ricci_scalar import bench_ricci_scalar
from quant_fund.models.riemann_curvature import bench_riemann_curvature
from quant_fund.models.riemann_metric import bench_riemann_metric


def test_riemann_metric():
    assert bench_riemann_metric()["synthetic_riemann_metric"] == 1.0


def test_levi_civita():
    assert bench_levi_civita()["synthetic_levi_civita"] == 1.0


def test_riemann_curvature():
    assert bench_riemann_curvature()["synthetic_riemann_curvature"] == 1.0


def test_ricci_scalar():
    assert bench_ricci_scalar()["synthetic_ricci_scalar"] == 1.0


def test_jacobi_field():
    assert bench_jacobi_field()["synthetic_jacobi_field"] == 1.0


def test_comparison_thm():
    assert bench_comparison_thm()["synthetic_comparison_thm"] == 1.0
