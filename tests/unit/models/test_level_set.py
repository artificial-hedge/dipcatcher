import numpy as np

from quant_fund.models.level_set import (
    bench_level_set,
    curvature_flow,
    grad_mag,
    reinitialize,
)


def test_reinitialize_pulls_grad_to_one():
    n = 48
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    dx = x[1] - x[0]
    sdf = np.sqrt(xx**2 + yy**2) - 0.5
    phi0 = sdf * 1.8 + 0.1 * np.sin(3 * xx)
    before = np.abs(grad_mag(phi0, dx) - 1).mean()
    after = np.abs(grad_mag(reinitialize(phi0, dx, 30), dx)[2:-2, 2:-2] - 1).mean()
    assert after < before


def test_reinitialize_keeps_contour():
    n = 48
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    sdf = np.sqrt(xx**2 + yy**2) - 0.5
    phi_r = reinitialize(sdf * 1.5, x[1] - x[0], 30)
    drift = np.mean((phi_r > 0) != (sdf > 0))
    assert drift < 0.15


def test_curvature_shrinks_circle():
    n = 48
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    dx = x[1] - x[0]
    sdf = np.sqrt(xx**2 + yy**2) - 0.5
    phi = curvature_flow(sdf, dx, dt=0.2 * dx**2, steps=30)
    assert np.mean(phi < 0) < np.mean(sdf < 0)


def test_bench():
    out = bench_level_set(seed=4)
    assert out["synthetic_grad_improvement"] > 1.5
    assert out["synthetic_curv_area_ratio"] < 1.0
