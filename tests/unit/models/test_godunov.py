import numpy as np

from quant_fund.models.godunov import (
    bench_godunov,
    burgers_flux_riemann,
    godunov_burgers,
)


def test_riemann_cases():
    # shock with positive speed → left state flux
    assert burgers_flux_riemann(1.0, 0.0) == 0.5
    # rarefaction through zero → sonic flux 0
    assert burgers_flux_riemann(-0.5, 0.5) == 0.0
    # rarefaction all-positive → left flux
    assert burgers_flux_riemann(0.4, 0.8) == 0.5 * 0.4**2


def test_shock_speed_rankine_hugoniot():
    n = 400
    x = np.linspace(0, 2, n, endpoint=False)
    dx = x[1] - x[0]
    u0 = np.where(x < 1.0, 1.0, 0.0)
    t = 0.3
    dt = 0.8 * dx
    u = godunov_burgers(u0, dx, dt, int(t / dt))
    below = np.where(u < 0.5)[0]
    x_num = x[below[0]]
    assert abs(x_num - (1.0 + 0.5 * t)) < 0.05


def test_rarefaction_fan():
    n = 400
    x = np.linspace(0, 2, n, endpoint=False)
    dx = x[1] - x[0]
    u0 = np.where(x < 1.0, 0.0, 1.0)
    t = 0.3
    u = godunov_burgers(u0, dx, 0.8 * dx, int(t / (0.8 * dx)))
    fan = np.clip((x - 1.0) / t, 0.0, 1.0)
    assert np.sqrt(((u - fan) ** 2).mean()) < 0.05


def test_bench():
    out = bench_godunov(seed=6)
    assert out["synthetic_shock_pos_err"] < 0.05
    assert out["synthetic_conservation_bonus"] >= 0.0
    assert out["synthetic_fan_err"] < 0.05
