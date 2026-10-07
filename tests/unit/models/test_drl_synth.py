"""Unit tests for quant_fund.models._drl_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._drl_synth import (
    cvar,
    mc_return_dist,
    sample_return,
    synth_reward_env,
)


def test_reward_env_shapes_and_determinism() -> None:
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    s1, x1 = synth_reward_env(64, rng1)
    s2, x2 = synth_reward_env(64, rng2)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(x1, x2)
    assert x1.shape == (64, 4)
    assert (x1.sum(1) == 1.0).all()
    assert ((s1 >= 0) & (s1 < 4)).all()


def test_actions_share_mean_honestly() -> None:
    rng = np.random.default_rng(1)
    s = np.zeros(200_000)
    r0 = sample_return(s, np.zeros(200_000), rng)
    r1 = sample_return(s, np.ones(200_000), rng)
    # same mean by construction; action 1 has heavy left tail
    assert abs(r0.mean() - 0.5) < 0.02
    assert abs(r1.mean() - 0.5) < 0.02
    assert r1.min() < r0.min() - 1.0  # heavy left tail really present


def test_cvar_prefers_tight_action() -> None:
    rng = np.random.default_rng(2)
    c0 = cvar(mc_return_dist(0, 0, rng, n=8000))
    c1 = cvar(mc_return_dist(0, 1, rng, n=8000))
    assert c0 > c1  # risk-aware agent prefers action 0


def test_cvar_rejects_empty_and_bad_alpha() -> None:
    with pytest.raises(ValueError, match="empty"):
        cvar(np.array([]))
    with pytest.raises(ValueError, match="alpha"):
        cvar(np.array([1.0, 2.0]), alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        cvar(np.array([1.0, 2.0]), alpha=1.5)


def test_mc_return_dist_rejects_bad_action_and_n() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="action"):
        mc_return_dist(0, 2, rng)
    with pytest.raises(ValueError, match="n"):
        mc_return_dist(0, 0, rng, n=0)


def test_cvar_monotone_in_alpha() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)
    assert cvar(x, alpha=0.05) < cvar(x, alpha=0.5) < cvar(x, alpha=1.0)
