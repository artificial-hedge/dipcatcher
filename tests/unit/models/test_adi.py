import numpy as np

from quant_fund.models.adi import adi_heat, adi_step, bench_adi


def test_adi_mode_decay():
    n = 32
    x = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    dx = x[1] - x[0]
    t = 0.02
    u = adi_heat(u0, dx, t / 50, 50)
    amp = float((u * u0).sum() / (u0 * u0).sum())
    assert abs(amp - np.exp(-2 * np.pi**2 * t)) < 2e-3


def test_adi_unconditionally_stable():
    n = 16
    x = np.linspace(0, 1, n + 1)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    u0 = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    u = adi_heat(u0, x[1] - x[0], 50 * (x[1] - x[0]) ** 2, 5)
    assert np.isfinite(u).all()
    assert np.abs(u).max() < 1.0


def test_adi_step_shape():
    u0 = np.random.default_rng(0).standard_normal((10, 12))
    assert adi_step(u0, 0.1, 0.001).shape == (10, 12)


def test_bench():
    out = bench_adi(seed=1)
    assert out["synthetic_adi_amp_err"] < 1e-3
    assert out["synthetic_adi_big_dt_finite"] == 1.0
