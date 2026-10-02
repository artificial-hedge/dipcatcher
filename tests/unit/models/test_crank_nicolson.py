import numpy as np

from quant_fund.models.crank_nicolson import (
    bench_crank_nicolson,
    crank_nicolson_heat,
    ftcs_heat,
    heat_exact,
)


def test_cn_decays_mode():
    nx = 32
    x = np.linspace(0, 1, nx + 1)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    u = crank_nicolson_heat(u0, dx, 1e-4, 500)
    exact = heat_exact(x, 0.05)
    assert np.abs(u - exact).max() < 1e-3


def test_cn_unconditionally_stable():
    nx = 16
    x = np.linspace(0, 1, nx + 1)
    dx = x[1] - x[0]
    u0 = np.sin(np.pi * x)
    u = crank_nicolson_heat(u0, dx, 10 * dx**2, 20)
    assert np.isfinite(u).all()
    assert np.abs(u).max() < 1.0
    uf = ftcs_heat(u0, dx, 10 * dx**2, 20)
    assert np.abs(uf).max() > 10.0


def test_bench():
    out = bench_crank_nicolson(seed=6)
    assert 1.7 < out["synthetic_cn_order"] < 2.3
    assert out["synthetic_stability_ratio"] > 1.0
