import numpy as np

from quant_fund.models.adams import ab2, ab3, bench_adams


def test_ab2_decays():
    def f(t, y):
        return -5.0 * y

    t = np.linspace(0, 1, 101)
    y = ab2(f, 1.0, t)
    assert abs(y[-1] - np.exp(-5)) < 0.01


def test_ab3_beats_ab2():
    lam = -10.0

    def f(t, y):
        return lam * (y - np.cos(t)) - np.sin(t)

    t = np.linspace(0, 2, 201)
    e2 = abs(ab2(f, 1.0, t)[-1] - np.cos(2))
    e3 = abs(ab3(f, 1.0, t)[-1] - np.cos(2))
    assert e3 < e2


def test_bench_orders():
    out = bench_adams(seed=2)
    assert 1.7 < out["synthetic_ab2_order"] < 2.3
    assert 2.7 < out["synthetic_ab3_order"] < 3.3
    assert out["synthetic_abm3_err"] < 1e-6
