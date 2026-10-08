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


def test_linear_baseline_is_oscillatory_lw():
    """The linear comparison baseline must be a high-order scheme that
    oscillates at sharp features (Lax-Wendroff) — a monotone first-order
    upwind baseline would make the WENO contrast meaningless."""
    n = 160
    x = np.linspace(0, 1, n, endpoint=False)
    # narrow bump: LW keeps more peak than upwind but rings negatively
    u0 = np.exp(-((((x - 0.3) % 1.0) / 0.02) ** 2))
    u = linear_advect(u0, 0.5, 120)
    assert u.max() > 0.4  # upwind smears this to ~0.23
    assert u.min() < -0.05  # upwind is monotone: min ~ 0


def test_linear_step_overshoots_weno_does_not():
    n = 200
    x = np.linspace(0, 1, n, endpoint=False)
    u0 = ((x > 0.3) & (x < 0.5)).astype(np.float64)
    steps = int(0.3 / (0.5 / n))
    u_lin = linear_advect(u0, 0.5, steps)
    u_w = weno_advect(u0, 0.5, steps)
    lin_over = max(u_lin.max() - 1.0, 0.0) + max(-u_lin.min(), 0.0)
    weno_over = max(u_w.max() - 1.0, 0.0) + max(-u_w.min(), 0.0)
    assert lin_over > 0.1
    assert weno_over < lin_over / 2


def test_bench():
    out = bench_weno(seed=3)
    assert out["synthetic_weno_overshoot"] < 0.2
    assert out["synthetic_weno_smooth_err"] < 0.05
