import numpy as np
from scipy.linalg import expm

from quant_fund.models.strang import bench_strang, lie_split, strang_split


def test_strang_beats_lie():
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    B = np.array([[0.5, 0.2], [0.0, -0.5]])
    y0 = np.array([1.0, 0.0])
    t = np.linspace(0, 1, 51)
    exact = expm(A + B) @ y0

    def af(y, h):
        return expm(A * h) @ y

    def bf(y, h):
        return expm(B * h) @ y

    es = np.abs(strang_split(af, bf, y0, t)[-1] - exact).max()
    el = np.abs(lie_split(af, bf, y0, t)[-1] - exact).max()
    assert es < el


def test_commuting_exact():
    A = np.diag([1.0, -1.0])
    B = np.diag([2.0, 0.5])
    y0 = np.array([1.0, 1.0])
    t = np.linspace(0, 1, 11)

    def af(y, h):
        return expm(A * h) @ y

    def bf(y, h):
        return expm(B * h) @ y

    exact = expm(A + B) @ y0
    y = strang_split(af, bf, y0, t)
    assert np.abs(y[-1] - exact).max() < 1e-12


def test_bench():
    out = bench_strang(seed=4)
    assert 1.7 < out["synthetic_strang_order"] < 2.3
    assert out["synthetic_strang_advantage"] > 1.0
