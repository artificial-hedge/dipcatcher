import numpy as np

from quant_fund.models.factorization_machine import (
    bench_factorization_machine,
    fm_fit,
    fm_predict,
)


def test_fm_learns_interaction():
    rng = np.random.default_rng(0)
    n = 250
    x = rng.normal(0, 1, (n, 4))
    y = x[:, 0] * x[:, 1] + rng.normal(0, 0.05, n)
    m = fm_fit(x, y, k=6, it=60, lr=0.01, seed=0)
    pred = fm_predict(m, x)
    ss = ((y - pred) ** 2).mean() / y.var()
    assert ss < 0.3


def test_fm_predict_shape():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (40, 3))
    y = rng.normal(0, 1, 40)
    m = fm_fit(x, y, k=4, it=5, seed=1)
    assert fm_predict(m, x).shape == (40,)


def test_bench_factorization_machine():
    out = bench_factorization_machine(seed=551)
    assert out["synthetic_fm_r2"] > 0.7
    assert out["synthetic_fm_r2"] > out["synthetic_linear_r2"] + 0.3
