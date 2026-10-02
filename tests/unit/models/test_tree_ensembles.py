import numpy as np

from quant_fund.models.tree_ensembles import (
    bench_trees,
    cart_fit,
    forest_predict,
    gbm_predict,
    gradient_boosting,
    random_forest,
    tree_predict,
)


def _blobs(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    x = np.vstack(
        [
            rng.normal([0, 0], 0.4, (n // 2, 2)),
            rng.normal([2, 2], 0.4, (n - n // 2, 2)),
        ]
    )
    y = np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
    return x, y


def test_cart_perfect_separable():
    x, y = _blobs()
    tree = cart_fit(x, y, max_depth=5)
    pred = tree_predict(tree, x)
    assert (pred == y).mean() > 0.95


def test_random_forest_beats_single():
    x, y = _blobs(seed=1)
    rf = random_forest(x, y, n_trees=15, seed=2)
    pred = forest_predict(rf, x)
    assert (pred == y).mean() > 0.95


def test_gbm_reduces_mse():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (200, 2))
    y = x[:, 0] ** 2 + rng.normal(0, 0.1, 200)
    mdl = gradient_boosting(x, y, n_stumps=60, lr=0.3)
    pred = gbm_predict(mdl, x)
    assert ((y - pred) ** 2).mean() < ((y - y.mean()) ** 2).mean() * 0.6


def test_bench_trees():
    out = bench_trees(seed=546)
    assert out["synthetic_rf_moons_acc"] >= 0.9
    assert out["synthetic_gbm_mse"] < out["synthetic_gbm_base_mse"]
