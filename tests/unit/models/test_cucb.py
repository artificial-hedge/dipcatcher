"""Tests for cucb — combinatorial UCB semi-bandit."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cucb import bench_cucb


def test_ucb_unpulled_arms_priority() -> None:
    # honest init: unpulled arms outrank every pulled arm (+inf). The old
    # code fabricated rew = rng.random()*0.01 pseudo-rewards instead.
    from quant_fund.models.cucb import _ucb_scores

    s = _ucb_scores(np.zeros(4), np.array([0.0, 2.0, 0.0, 5.0]), 10)
    assert np.isinf(s[0]) and np.isinf(s[2])
    assert np.isfinite(s[1]) and np.isfinite(s[3])


def test_ucb_score_formula() -> None:
    from quant_fund.models.cucb import _ucb_scores

    s = _ucb_scores(np.array([3.0]), np.array([2.0]), 8)
    expect = 1.5 + np.sqrt(1.5 * np.log(10) / 2.0)
    assert s[0] == expect


def test_ucb_no_fabricated_init() -> None:
    # arms start with zero pulls AND zero rewards
    from quant_fund.models.cucb import _ucb_scores

    s = _ucb_scores(np.zeros(3), np.zeros(3), 0)
    assert np.isinf(s).all()


def test_bench_cucb() -> None:
    out = bench_cucb()
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
    assert out["synthetic_cucb_reward_gain"] > 0
