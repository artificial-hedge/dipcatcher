import numpy as np

from quant_fund.models.coordinate_descent_enet import (
    bench_coordinate_descent_enet,
    enet_fit,
    enet_path,
)


def test_enet_sparsifies():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (100, 10))
    beta = np.r_[3.0, np.zeros(9)]
    y = x @ beta + rng.normal(0, 0.1, 100)
    b = enet_fit(x, y, lam=0.1, alpha=1.0, it=300)
    assert abs(b[0]) > 1.0
    assert (np.abs(b[1:]) < 1e-6).sum() >= 7


def test_enet_path_shapes():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (80, 8))
    y = x[:, 0] * 2 + rng.normal(0, 0.2, 80)
    path = enet_path(x, y, n_lam=10, alpha=0.8)
    betas = np.asarray(path["betas"])
    assert betas.shape == (10, 8)
    lams = np.asarray(path["lams"])
    assert (np.diff(lams) < 0).all()


def test_bench_coordinate_descent_enet():
    out = bench_coordinate_descent_enet(seed=554)
    assert out["synthetic_enet_true_support"] >= 4
    assert out["synthetic_enet_test_mse"] < out["synthetic_ols_test_mse"]
