import numpy as np

from quant_fund.models.svm_classifiers import (
    bench_svm,
    kernel_svm_fit,
    kernel_svm_predict,
    linear_svm_predict,
    pegasos_fit,
)


def test_pegasos_separable():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.3, (60, 2)), rng.normal([3, 3], 0.3, (60, 2))])
    y = np.r_[np.zeros(60), np.ones(60)]
    w = pegasos_fit(x, y, lam=0.01, it=3000, seed=0)
    assert (linear_svm_predict(w, x) == y).mean() > 0.95


def test_kernel_svm_circle():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 160)
    r_in = rng.uniform(0, 1, 80)
    r_out = rng.uniform(2, 3, 80)
    x = np.vstack(
        [
            np.c_[r_in * np.cos(t[:80]), r_in * np.sin(t[:80])],
            np.c_[r_out * np.cos(t[:80]), r_out * np.sin(t[:80])],
        ]
    )
    y = np.r_[np.zeros(80), np.ones(80)]
    m = kernel_svm_fit(x, y, lam=0.01, gamma=0.8, it=3000, seed=1)
    assert (kernel_svm_predict(m, x) == y).mean() > 0.85


def test_bench_svm():
    out = bench_svm(seed=552)
    assert out["synthetic_pegasos_blob_acc"] > 0.95
    assert out["synthetic_ksvm_circle_acc"] > out["synthetic_linear_circle_acc"]
