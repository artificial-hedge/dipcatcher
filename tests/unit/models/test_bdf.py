import numpy as np

from quant_fund.models.bdf import bdf1, bdf2, bench_bdf


def test_bdf1_decay():
    def f(t, y):
        return -10.0 * y

    t = np.linspace(0, 1, 51)
    y = bdf1(f, 1.0, t)
    assert abs(y[-1] - np.exp(-10)) < 0.1


def test_bdf2_vs_bdf1_accurate():
    lam = -5.0

    def f(t, y):
        return lam * (y - np.cos(t)) - np.sin(t)

    t = np.linspace(0, 2, 101)
    e1 = abs(bdf1(f, 1.0, t)[-1] - np.cos(2))
    e2 = abs(bdf2(f, 1.0, t)[-1] - np.cos(2))
    assert e2 < e1


def test_bench_orders():
    out = bench_bdf(seed=1)
    assert 0.7 < out["synthetic_bdf1_order"] < 1.3
    assert 1.7 < out["synthetic_bdf2_order"] < 2.3
    assert out["synthetic_stiff_err_ratio"] > 1.0
