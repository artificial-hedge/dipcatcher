"""Tests for permutation / randomization inference
(models/permutation_inference.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.permutation_inference import (
    bench_permutation_inference,
    fisher_permutation_p,
    max_t_stepdown,
    randomization_ci,
    synth_ab,
)


def test_detects_real_effect():
    d = synth_ab(seed=5, effect=0.8)
    out = fisher_permutation_p(np.asarray(d["y"]), np.asarray(d["treated"]), n_perm=499, seed=5)
    assert out["p_two_sided"] < 0.01
    assert out["stat_obs"] > 0.0


def test_null_not_rejected():
    d = synth_ab(seed=6, effect=0.0)
    out = fisher_permutation_p(np.asarray(d["y"]), np.asarray(d["treated"]), n_perm=499, seed=6)
    assert out["p_two_sided"] > 0.01


def test_p_resolution():
    d = synth_ab(seed=7, effect=3.0)
    out = fisher_permutation_p(np.asarray(d["y"]), np.asarray(d["treated"]), n_perm=99, seed=7)
    assert out["p_two_sided"] >= out["mc_resolution"] - 1e-12


def test_max_t_stepdown_monotone():
    rng = np.random.default_rng(8)
    obs = np.array([4.0, 1.0, -0.5])
    draws = rng.normal(0.0, 1.0, (400, 3))
    out = max_t_stepdown(obs, draws)
    adj = np.asarray(out["adjusted_p"])
    assert adj[0] < adj[1]
    assert np.all(adj >= 0.0) and np.all(adj <= 1.0)


def test_max_t_flags_signal_only():
    rng = np.random.default_rng(9)
    obs = np.array([5.0, 0.3])
    draws = rng.normal(0.0, 1.0, (400, 2))
    adj = np.asarray(max_t_stepdown(obs, draws)["adjusted_p"])
    assert adj[0] < 0.05 < adj[1]


def test_ci_covers_effect():
    d = synth_ab(seed=10, effect=0.7, n=300)
    ci = randomization_ci(
        np.asarray(d["y"]), np.asarray(d["treated"]), n_grid=41, n_perm=150, seed=10
    )
    assert math.isfinite(ci["ci_lo"])
    assert ci["ci_lo"] <= 0.7 <= ci["ci_hi"]
    assert 0.05 < ci["ci_width"] < 3.0


def test_validation():
    with pytest.raises(ValueError):
        fisher_permutation_p(np.ones(4), np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError):
        fisher_permutation_p(np.ones(20), np.ones(20))
    with pytest.raises(ValueError):
        fisher_permutation_p(np.ones(20), np.zeros(20))
    with pytest.raises(ValueError):
        max_t_stepdown(np.array([1.0]), np.ones((10, 2)))


def test_determinism():
    d = synth_ab(seed=11, effect=0.5)
    a = fisher_permutation_p(np.asarray(d["y"]), np.asarray(d["treated"]), n_perm=99, seed=11)
    b = fisher_permutation_p(np.asarray(d["y"]), np.asarray(d["treated"]), n_perm=99, seed=11)
    assert a["p_two_sided"] == b["p_two_sided"]


def test_bench_keys():
    out = bench_permutation_inference()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_power"] == 1.0
    assert out["synthetic_size_controlled"] == 1.0
    assert out["synthetic_ci_covers"] == 1.0
    assert out["synthetic_determinism"] == 1.0
