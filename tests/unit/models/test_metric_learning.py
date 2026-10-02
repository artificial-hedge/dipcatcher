import numpy as np

from quant_fund.models.metric_learning import (
    bench_metric_learning,
    knn_predict,
    lmnn_learn,
    nca_learn,
)


def _fixture(seed: int = 7, n: int = 160):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = np.c_[rng.normal(y * 2.0, 0.6, n), rng.normal(0, 1.0, (n, 4))]
    return x, y


def test_nca_learns_metric():
    x, y = _fixture()
    a = nca_learn(x, y, it=150, lr=0.08, seed=1)
    raw = (knn_predict(x, y, x, k=3) == y).mean()
    tr = (knn_predict(x, y, x, k=3, m=a.T @ a) == y).mean()
    assert tr >= raw


def test_lmnn_psd():
    x, y = _fixture()
    m = lmnn_learn(x, y, k=3, it=20, seed=2)
    assert np.linalg.eigvalsh(m).min() > -1e-6


def test_knn_predict_basic():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (50, 2))
    y = (x[:, 0] > 0).astype(float)
    pred = knn_predict(x, y, x, k=1)
    assert (pred == y).mean() == 1.0


def test_bench_metric_learning():
    out = bench_metric_learning(seed=547)
    assert out["synthetic_knn_nca_acc"] > out["synthetic_knn_xor_raw_acc"]
    assert out["synthetic_knn_lmnn_acc"] > out["synthetic_knn_scale_raw_acc"]
