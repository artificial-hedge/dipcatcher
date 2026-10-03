"""Tests for quant_fund.fusion.diversity_bench."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.fusion.diversity_bench import diversity_bench, effective_bets


def test_identical_members_enb_one():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 300)
    resid = np.column_stack([base + rng.normal(0, 1e-9, 300) for _ in range(4)])
    r = effective_bets(resid)
    assert r["effective_bets"] < 1.5


def test_orthogonal_members_enb_m():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 1, (300, 4))
    r = effective_bets(resid)
    assert r["effective_bets"] > 2.5


def test_validation():
    with pytest.raises(ValueError):
        effective_bets(np.ones((3, 5)))  # n_obs <= n_members
    with pytest.raises(ValueError):
        effective_bets(np.array([[np.nan] * 3] * 10))


def test_bench_sealed():
    r = diversity_bench(seed=0)
    assert r["schema"] == "diversity.v1"
    assert r["claim"]["verdict"] == "ok"
    assert r == diversity_bench(seed=0)
