"""Tests for wave-118 POMDP canon: qmdp, grid_pomdp, pbvi, perseus,
hsvi, pomcp."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.grid_pomdp import bench_grid, grid_vi
from quant_fund.models.hsvi import bench_hsvi, hsvi
from quant_fund.models.pbvi import bench_pbvi, pbvi, point_backup
from quant_fund.models.perseus import bench_perseus
from quant_fund.models.pomcp import POMCP, bench_pomcp
from quant_fund.models.qmdp import (
    belief_update,
    bench_qmdp,
    mdp_value_iteration,
    qmdp_policy,
    tiger_pomdp,
)


def test_belief_update_normalizes():
    p = tiger_pomdp()
    b = belief_update(p, np.array([0.5, 0.5]), 0, 0)
    assert np.isclose(b.sum(), 1.0)
    assert b[0] > b[1]  # heard-left → tiger more likely left


def test_qmdp_listens_when_uncertain():
    p = tiger_pomdp()
    v = mdp_value_iteration(p)
    assert qmdp_policy(p, np.array([0.5, 0.5]), v) == 0


def test_grid_vi_produces_vectors():
    p = tiger_pomdp()
    alphas = grid_vi(p, n_iters=5, grid_n=11)
    assert alphas.shape[1] == 2 and len(alphas) > 0


def test_point_backup_shape():
    p = tiger_pomdp()
    alphas = np.array([np.zeros(2)])
    a = point_backup(p, np.array([0.5, 0.5]), alphas)
    assert a.shape == (2,)


def test_pbvi_runs():
    p = tiger_pomdp()
    rng = np.random.default_rng(0)
    a = pbvi(p, n_beliefs=10, n_iters=5, rng=rng)
    assert a.shape[1] == 2


def test_hsvi_runs():
    p = tiger_pomdp()
    rng = np.random.default_rng(0)
    alphas, pts, vals = hsvi(p, n_iters=3, rng=rng, horizon=4)
    assert alphas.shape[1] == 2


def test_pomcp_particle_filter():
    p = tiger_pomdp()
    rng = np.random.default_rng(0)
    agent = POMCP(p, n_particles=100)
    agent.step(0, 0, rng)
    # after hearing left, particles skew toward state 0
    assert np.mean(agent.particles == 0) > 0.5


@pytest.mark.parametrize(
    "fn",
    [bench_qmdp, bench_grid, bench_pbvi, bench_perseus, bench_hsvi, bench_pomcp],
    ids=lambda f: f.__name__,
)
def test_w118_benches(fn):
    out = fn(seed=20261231)
    assert out and all(np.isfinite(v) for v in out.values())
    assert all(k.startswith("synthetic_") for k in out)
