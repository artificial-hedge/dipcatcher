import numpy as np

from quant_fund.models.kernel_methods import (
    bench_kernel_methods,
    krr_fit,
    krr_predict,
    nystrom_features,
    nystrom_fit,
    rff_map,
    rff_transform,
)


def test_krr_sine():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 5, (100, 1)), axis=0)
    y = np.sin(x[:, 0])
    m = krr_fit(x, y, lam=0.05, gamma=0.8)
    pred = krr_predict(m, x)
    assert ((y - pred) ** 2).mean() < 0.05


def test_rff_approximates_rbf():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (30, 3))
    rff = rff_map(x, n_feat=800, gamma=0.5, seed=1)
    z = rff_transform(rff, x)
    approx = z @ z.T
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    exact = np.exp(-0.5 * d2)
    assert np.abs(approx - exact).mean() < 0.1


def test_nystrom_features_shape():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (50, 2))
    m = nystrom_fit(x, n_landmarks=10, gamma=1.0, seed=2)
    f = nystrom_features(m, x)
    assert f.shape == (50, 10)


def test_bench_kernel_methods():
    out = bench_kernel_methods(seed=555)
    assert out["synthetic_krr_mse"] < out["synthetic_linridge_mse"]
