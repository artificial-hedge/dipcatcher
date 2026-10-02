import numpy as np

from quant_fund.models.adaboost import (
    adaboost_fit,
    adaboost_predict,
    bench_adaboost,
    logitboost_fit,
    logitboost_predict,
)


def _xorish(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(float)
    return x, y


def test_adaboost_xor():
    x, y = _xorish()
    m = adaboost_fit(x, y, it=80)
    acc = (adaboost_predict(m, x) == y).mean()
    assert acc > 0.7


def test_logitboost_improves():
    x, y = _xorish()
    m = logitboost_fit(x, y, it=40)
    acc = (logitboost_predict(m, x) == y).mean()
    assert acc > 0.7


def test_bench_adaboost():
    out = bench_adaboost(seed=565)
    assert out["synthetic_adaboost_acc"] > out["synthetic_stump_acc"]
