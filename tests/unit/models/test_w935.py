"""Wave-935 convex-optimization canon tests."""

from __future__ import annotations

from quant_fund.models.analytic_center import bench_analytic_center
from quant_fund.models.cvx_reform import bench_cvx_reform
from quant_fund.models.dik_ellipsoid import bench_dik_ellipsoid
from quant_fund.models.kkt_solve import bench_kkt_solve
from quant_fund.models.logbarrier_fn import bench_logbarrier_fn
from quant_fund.models.self_concordant import bench_self_concordant


def test_kkt_solve():
    assert bench_kkt_solve()["synthetic_kkt_solve"] == 1.0


def test_cvx_reform():
    assert bench_cvx_reform()["synthetic_cvx_reform"] == 1.0


def test_self_concordant():
    assert bench_self_concordant()["synthetic_self_concordant"] == 1.0


def test_logbarrier_fn():
    assert bench_logbarrier_fn()["synthetic_logbarrier_fn"] == 1.0


def test_analytic_center():
    assert bench_analytic_center()["synthetic_analytic_center"] == 1.0


def test_dik_ellipsoid():
    assert bench_dik_ellipsoid()["synthetic_dik_ellipsoid"] == 1.0
