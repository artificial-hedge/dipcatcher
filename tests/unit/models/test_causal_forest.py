"""Tests for honest causal trees/forest (models/causal_forest.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.causal_forest import (
    bench_causal_forest,
    causal_forest,
    causal_tree,
    synth_cate,
)


@pytest.fixture
def data():
    return synth_cate(n=1200, seed=7)


def test_tree_splits_on_x1(data):
    out = causal_tree(np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), seed=1)
    assert out["n_leaves"] >= 2
    assert out["effect_sd"] > 0.05  # heterogeneous effects found


def test_forest_cate_correlates(data):
    f = causal_forest(
        np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), n_trees=30, seed=2
    )
    cate = np.asarray(f["cate"])
    tau = np.asarray(data["tau_true"])
    corr = float(np.corrcoef(cate, tau)[0, 1])
    assert corr > 0.7


def test_forest_subgroup_gap(data):
    f = causal_forest(
        np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), n_trees=30, seed=3
    )
    cate = np.asarray(f["cate"])
    x = np.asarray(data["x"])
    hi = x[:, 0] > 0
    gap = float(cate[hi].mean() - cate[~hi].mean())
    assert 0.4 < gap < 1.6  # true gap 1.2, shrunk by leaf averaging


def test_importance_top_feature(data):
    f = causal_forest(
        np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), n_trees=30, seed=4
    )
    assert int(np.asarray(f["importance"]).argmax()) == 0


def test_constant_effect_low_sd():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(0, 1, (n, 4))
    t = (rng.random(n) < 0.5).astype(np.float64)
    y = 0.5 * x[:, 0] + 1.0 * t + rng.normal(0, 0.4, n)
    f = causal_forest(y, t, x, n_trees=20, seed=5)
    assert abs(float(f["cate_mean"]) - 1.0) < 0.3


def test_validation():
    with pytest.raises(ValueError):
        causal_forest(np.ones(40), np.ones(40), np.ones((40, 2)))
    with pytest.raises(ValueError):
        causal_tree(np.ones(200), np.zeros(200), np.ones((200, 2)))
    with pytest.raises(ValueError):
        causal_forest(np.ones(100), np.r_[np.ones(50), np.zeros(50)], np.ones((100, 2)), n_trees=3)


def test_determinism(data):
    a = causal_forest(
        np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), n_trees=10, seed=6
    )["cate"]
    b = causal_forest(
        np.asarray(data["y"]), np.asarray(data["t"]), np.asarray(data["x"]), n_trees=10, seed=6
    )["cate"]
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_bench_keys():
    out = bench_causal_forest()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_cate_corr"] > 0.7
    assert out["synthetic_top_feature_is_x1"] == 1.0
    assert out["synthetic_ate_err"] < 0.15
    assert out["synthetic_determinism"] == 1.0
