import numpy as np

from quant_fund.models.radau import bench_radau, radau


def test_radau_smooth():
    def f(t, y):
        return -2.0 * y

    t = np.linspace(0, 1, 21)
    y = radau(f, np.array([1.0]), t)
    assert abs(y[-1, 0] - np.exp(-2)) < 1e-6


def test_radau_stiff_stable():
    def f(t, y):
        return -100.0 * (y - np.cos(t)) - np.sin(t)

    t = np.linspace(0, 1, 26)
    y = radau(f, np.array([1.0]), t, stages=3)
    assert np.isfinite(y).all()
    assert abs(y[-1, 0] - np.cos(1)) < 0.01


def test_bench():
    out = bench_radau(seed=3)
    assert out["synthetic_radau3_order"] > 3.5
    assert out["synthetic_radau3_err"] < 1e-8
