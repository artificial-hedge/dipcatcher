import numpy as np

from quant_fund.models.weno import (
    bench_weno,
    linear_advect,
    weno_advect,
)


def test_weno_step_bounded():
    n = 160
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = ((x > 0.25) & (x < 0.55)).astype(np.float64)
    u = weno_advect(u0, 0.5, 60)
    assert u.max() < 1.2
    assert u.min() > -0.2


def test_weno_smooth_accuracy():
    n = 200
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = np.sin(2 * np.pi * x)
    sigma = 0.5
    steps = int(0.25 / (sigma / n))
    u = weno_advect(u0, sigma, steps)
    ex = np.sin(2 * np.pi * ((x - sigma * steps / n) % 1.0))
    assert np.sqrt(((u - ex) ** 2).mean()) < 0.05


def test_linear_upwind_diffuses():
    n = 160
    x = np.linspace(0, 1, n, endpoint=False)
    # narrow bump: first-order upwind smears the peak fast
    u0 = np.exp(-((((x - 0.3) % 1.0) / 0.02) ** 2))
    u = linear_advect(u0, 0.5, 120)
    assert u.max() < 0.8


def test_bench():
    out = bench_weno(seed=3)
    assert out["synthetic_weno_overshoot"] < 0.2
    assert out["synthetic_weno_smooth_err"] < 0.05
