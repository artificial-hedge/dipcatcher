import numpy as np

from quant_fund.models.expectation_propagation import (
    bench_expectation_propagation,
    ep_probit_fit,
    ep_probit_marginal,
    ep_probit_predict,
)


def test_ep_probit_separable():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (120, 3))
    y = (x[:, 0] + x[:, 1] > 0).astype(float)
    m = ep_probit_fit(x, y, it=30)
    assert (ep_probit_predict(m, x) == y).mean() > 0.9


def test_ep_marginal_in_unit():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (80, 2))
    y = (x[:, 0] > 0).astype(float)
    m = ep_probit_fit(x, y, it=20)
    p = ep_probit_marginal(m, x)
    assert ((p >= 0) & (p <= 1)).all()


def test_bench_expectation_propagation():
    out = bench_expectation_propagation(seed=566)
    assert out["synthetic_ep_probit_acc"] > 0.75
    assert out["synthetic_ep_probit_brier"] < 0.2
