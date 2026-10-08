"""Tests for the Whittle index restless bandit (models/whittle_restless.py)."""

from __future__ import annotations

import numpy as np

from quant_fund.models import whittle_restless as wr


def test_whittle_index_finds_positive_to_negative_crossing() -> None:
    """A high-reward arm must get a nonzero Whittle index — the subsidy
    where the passive action overtakes the active one. A detector that
    only watched the opposite crossing direction always returned 0."""
    pa = np.array([[0.9, 0.1], [0.9, 0.1]])
    pp = np.array([[0.1, 0.9], [0.1, 0.9]])
    r = np.array([0.95, 0.8])
    w = wr._whittle(pa, pp, r)
    assert w[0] > 0.3
    assert w[1] > 0.3


def test_whittle_index_zero_when_passive_dominates_at_zero_subsidy() -> None:
    """A negative-reward arm is already dominated at zero subsidy —
    index 0 is the correct output, not a detector failure."""
    pa = np.array([[0.5, 0.5], [0.5, 0.5]])
    pp = np.array([[0.5, 0.5], [0.5, 0.5]])
    r = np.array([-0.5, -0.5])
    w = wr._whittle(pa, pp, r)
    assert w[0] == 0.0


def test_bench_whittle_beats_round_robin() -> None:
    out = wr.bench_whittle_restless()
    assert out["synthetic_whi_reward_gain"] > 0.0
