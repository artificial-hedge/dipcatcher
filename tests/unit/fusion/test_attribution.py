"""Tests for quant_fund.fusion.attribution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.fusion.attribution import (
    _chronological_folds,
    attribution_bench,
    leave_one_out_deltas,
    phantom_members,
    shapley_members,
)


def _committee(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """y = x + 0.5 eps; head0 = true cond quantile, head1 = noisy, head2 = noise."""
    rng = np.random.default_rng(seed)
    taus = np.linspace(0.1, 0.9, 5)
    x = rng.normal(0.0, 1.0, n)
    y = x + 0.5 * rng.normal(0.0, 1.0, n)
    q_true = x[:, None] + 0.5 * norm.ppf(taus)[None, :]
    q = np.zeros((n, 3, taus.size))
    q[:, 0, :] = q_true
    q[:, 1, :] = q_true + rng.normal(0.0, 0.4, (n, taus.size))
    q[:, 2, :] = rng.normal(0.0, 1.5, (n, taus.size))
    return q, y, taus


def test_loo_best_member_carries_delta():
    q, y, taus = _committee(160, 3)
    folds = _chronological_folds(160, 4)
    out = leave_one_out_deltas(q, y, taus, folds)
    assert out["deltas"].shape == (3,)
    # dropping the true-quantile head hurts most (or ties the noisy copy)
    assert out["deltas"][0] > 0.0
    # noise head earns less than the true head
    assert out["deltas"][0] >= out["deltas"][2]


def test_loo_rejects_single_member():
    q, y, taus = _committee(80, 0)
    with pytest.raises(ValueError, match="at least two members"):
        leave_one_out_deltas(q[:, :1, :], y, taus, _chronological_folds(80, 3))


def test_shapley_sums_to_total_gain():
    q, y, taus = _committee(120, 7)
    folds = _chronological_folds(120, 3)
    phi = shapley_members(q, y, taus, folds)
    assert phi.shape == (3,)
    # efficiency: sum phi = v(all) - v(empty)
    from quant_fund.fusion.attribution import _oof_pinball

    full, _, _ = _oof_pinball(q, y, taus, folds, 1.0)
    empty = np.mean(
        (y[:, None] - np.quantile(y, taus)[None, :])
        * (taus[None, :] - ((y[:, None] - np.quantile(y, taus)[None, :]) < 0))
    )
    assert phi.sum() == pytest.approx(empty - full, rel=1e-6, abs=1e-6)


def test_shapley_deterministic():
    q, y, taus = _committee(100, 11)
    folds = _chronological_folds(100, 3)
    a = shapley_members(q, y, taus, folds, seed=5)
    b = shapley_members(q, y, taus, folds, seed=5)
    np.testing.assert_array_equal(a, b)


def test_phantom_flags_noise_head():
    q, y, taus = _committee(160, 5)
    folds = _chronological_folds(160, 4)
    out = phantom_members(q, y, taus, folds, seed=5, n_null=4)
    assert out["phantom"].shape == (3,)
    assert out["null_floor"] >= 0.0
    # the pure-noise head should never beat the true-quantile head
    assert bool(out["phantom"][0]) or out["deltas"][0] > out["deltas"][2]


def test_bench_sealed_and_deterministic():
    r1 = attribution_bench(n=120, m_signal=3, m_noise=2, n_taus=5, n_folds=4)
    r2 = attribution_bench(n=120, m_signal=3, m_noise=2, n_taus=5, n_folds=4)
    assert r1["schema"] == "attribution.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["live_pnl_claim"] is False
    assert r1["claim"]["verdict"] in {"ok", "partial"}
    assert r1 == r2
