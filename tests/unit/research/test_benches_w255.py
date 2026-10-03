"""Wave-255 adapter tests."""

from quant_fund.research.benches_w255 import (
    bench_admm_lasso_family,
    bench_barrier_ip_family,
    bench_coord_descent_family,
    bench_ellipsoid_method_family,
    bench_proj_gradient_family,
    bench_simplex_lp_family,
)


def test_bench_simplex_lp_family():
    out = bench_simplex_lp_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_ellipsoid_method_family():
    out = bench_ellipsoid_method_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_barrier_ip_family():
    out = bench_barrier_ip_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_admm_lasso_family():
    out = bench_admm_lasso_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_coord_descent_family():
    out = bench_coord_descent_family()
    assert out and all(k.startswith("synthetic_") for k in out)


def test_bench_proj_gradient_family():
    out = bench_proj_gradient_family()
    assert out and all(k.startswith("synthetic_") for k in out)
