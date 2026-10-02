import numpy as np

from quant_fund.models.multiclass import (
    _exhaustive_code,
    bench_multiclass,
    ecoc_fit,
    ecoc_predict,
    ovr_fit,
    ovr_predict,
    softmax_fit,
    softmax_predict,
)


def _data(seed=0, n=300, k=3):
    rng = np.random.default_rng(seed)
    centers = np.array([[3, 0, 0], [-3, 0, 0], [0, 3, 0]])[:k]
    y = rng.integers(0, k, n).astype(np.int64)
    x = centers[y] + rng.standard_normal((n, 3))
    return x, y, k


def test_ovr():
    x, y, k = _data()
    w = ovr_fit(x, y, k)
    assert np.mean(ovr_predict(w, x) == y) > 0.85


def test_softmax():
    x, y, k = _data()
    w = softmax_fit(x, y, k)
    assert np.mean(softmax_predict(w, x) == y) > 0.85


def test_ecoc():
    x, y, k = _data()
    code = _exhaustive_code(k)
    w = ecoc_fit(x, y, code)
    assert np.mean(ecoc_predict(w, x, code) == y) > 0.8


def test_bench_keys():
    out = bench_multiclass()
    assert out["synthetic_ovr_acc"] > 0.8
    assert out["synthetic_softmax_acc"] > 0.8
    assert out["synthetic_ecoc_acc"] > 0.7
