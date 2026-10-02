"""PU-learning tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.pu_learning import (
    bench_pu,
    elkan_correct,
    elkan_e,
    fit_pu_logistic,
    pu_predict,
)


def _pu_data(seed=0, n=800):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.4
    X = np.where(
        y[:, None],
        rng.normal([1.2, 1.0], 0.8, (n, 2)),
        rng.normal([-0.8, -0.5], 1.0, (n, 2)),
    )
    s = np.zeros(n)
    idx = np.where(y)[0]
    s[idx[rng.random(idx.size) < 0.5]] = 1.0
    return X, y, s


def test_elkan_e_and_correct():
    h = np.array([0.8, 0.6, 0.9])
    e = elkan_e(h)
    assert e == pytest.approx(0.7666, rel=1e-2)
    c = elkan_correct(np.array([0.4]), e)
    assert 0 < c[0] <= 1


def test_pu_fit_runs_and_ranks():
    X, y, s = _pu_data()
    fit = fit_pu_logistic(X, s)
    scores = pu_predict(X, fit)
    n1 = int(y.sum())
    n0 = n1  # placeholder; compute below
    order = np.argsort(scores)
    ranks = np.empty(y.size)
    ranks[order] = np.arange(1, y.size + 1)
    n0 = int((~y).sum())
    auc = (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    assert auc > 0.85


def test_pu_requires_positives():
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError):
        fit_pu_logistic(X, np.zeros(20))


def test_bench_pu():
    out = bench_pu()
    assert out["synthetic_auc"] > 0.9
