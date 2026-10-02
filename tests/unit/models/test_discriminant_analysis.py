import numpy as np

from quant_fund.models.discriminant_analysis import (
    bench_discriminant,
    lda_fit,
    lda_predict,
    qda_fit,
    qda_predict,
)


def _blobs(seed: int = 0, n: int = 120):
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal([0, 0], 0.4, (n, 2)), rng.normal([2.5, 2.5], 0.4, (n, 2))])
    y = np.r_[np.zeros(n), np.ones(n)]
    return x, y


def test_lda_separable():
    x, y = _blobs()
    m = lda_fit(x, y)
    assert (lda_predict(m, x) == y).mean() > 0.97


def test_qda_separable():
    x, y = _blobs()
    m = qda_fit(x, y)
    assert (qda_predict(m, x) == y).mean() > 0.97


def test_bench_discriminant():
    out = bench_discriminant(seed=553)
    assert out["synthetic_qda_acc"] > out["synthetic_lda_acc"]
