import numpy as np

from quant_fund.models.evidential import (
    bench_evidential,
    dirichlet_predict,
    evidential_alpha,
    evidential_fit,
)


def test_evidential_acc():
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(2), 100).astype(np.int64)
    x = np.zeros((200, 3))
    x[y == 0, 0] = 2.0
    x[y == 1, 0] = -2.0
    x += 0.5 * rng.standard_normal(x.shape)
    m = evidential_fit(x, y, 2)
    a = evidential_alpha(m, x)
    out = dirichlet_predict(a)
    assert np.mean(out["p"].argmax(1) == y) > 0.9


def test_vacuity_far_vs_near():
    y = np.tile(np.arange(2), 100).astype(np.int64)
    x = np.random.default_rng(2).standard_normal((200, 4)) + np.where(y[:, None] == 0, 1.5, -1.5)
    m = evidential_fit(x, y, 2)
    a_in = evidential_alpha(m, x)
    a_far = evidential_alpha(m, x + 20.0)
    vac_in = dirichlet_predict(a_in)["vacuity"].mean()
    vac_far = dirichlet_predict(a_far)["vacuity"].mean()
    assert vac_far > vac_in


def test_bench_keys():
    out = bench_evidential()
    assert out["synthetic_evi_acc"] > 0.7
    assert out["synthetic_evi_vac_far"] > out["synthetic_evi_vac_in"]
