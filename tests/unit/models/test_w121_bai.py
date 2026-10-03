"""Tests for wave-121 best-arm-identification canon: lil_ucb,
sequential_halving, median_elim, ugape, ttts, track_stop."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lil_ucb import GaussianBandit, bench_lil_ucb, lil_ucb
from quant_fund.models.median_elim import bench_median_elim, median_elimination
from quant_fund.models.sequential_halving import (
    bench_sequential_halving,
    sequential_halving,
)
from quant_fund.models.track_stop import bench_track_stop, oracle_weights, track_stop
from quant_fund.models.ttts import bench_ttts, ttts
from quant_fund.models.ugape import bench_ugape, ugape


def test_gaussian_bandit_counts():
    env = GaussianBandit(np.array([0.0, 1.0]))
    rng = np.random.default_rng(0)
    env.pull(1, rng)
    assert env.counts[1] == 1 and env.means()[0] == 0.0


def test_lil_ucb_easy():
    rng = np.random.default_rng(1)
    best, pulls = lil_ucb(np.array([0.0, 0.5]), rng)
    assert best == 1 and pulls < 20000


def test_sequential_halving_halves():
    rng = np.random.default_rng(2)
    mu = np.array([0.0, 0.0, 0.0, 0.5])
    best, _ = sequential_halving(mu, rng, budget=1000)
    assert best == 3


def test_median_elim_eps():
    rng = np.random.default_rng(3)
    mu = np.array([0.0, 0.4, 0.1])
    best, _ = median_elimination(mu, rng, eps=0.15)
    assert mu[best] >= 0.4 - 0.15


def test_ugape_easy():
    rng = np.random.default_rng(4)
    best, _ = ugape(np.array([0.0, 0.5, 0.2]), rng, budget=800)
    assert best == 1


def test_ttts_easy():
    rng = np.random.default_rng(5)
    best, _ = ttts(np.array([0.0, 0.6]), rng)
    assert best == 1


def test_oracle_weights_simplex():
    w = oracle_weights(np.array([0.0, 0.3, 0.1]))
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_track_stop_easy():
    rng = np.random.default_rng(6)
    best, _ = track_stop(np.array([0.0, 0.5]), rng)
    assert best == 1


def test_benches():
    for fn in (
        bench_lil_ucb,
        bench_sequential_halving,
        bench_median_elim,
        bench_ugape,
        bench_ttts,
        bench_track_stop,
    ):
        out = fn()
        assert out and all(k.startswith("synthetic_") for k in out)
