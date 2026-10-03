"""Tests for CTMC multi-state models (models/multistate.py)."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from quant_fund.models.multistate import (
    aalen_johansen_id,
    bench_multistate,
    ctmc_intensity_fit,
    illness_death_simulate,
    mean_sojourn,
    synth_multistate,
    transition_probabilities,
)


def test_intensity_recovery():
    d = synth_multistate(seed=11)
    q_hat = ctmc_intensity_fit(
        cast(list[tuple[int, int]], d["transitions"]),
        np.asarray(d["exposure"]),
        n_states=3,
    )
    assert abs(q_hat[0, 1] - 0.15) < 0.05
    assert abs(q_hat[0, 2] - 0.05) < 0.03
    assert abs(q_hat[1, 2] - 0.30) < 0.10
    assert np.allclose(np.diag(q_hat), -np.sum(q_hat - np.diag(np.diag(q_hat)), axis=1))


def test_transition_probs_chapman_kolmogorov():
    q = np.array([[-0.2, 0.15, 0.05], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]])
    p2 = transition_probabilities(q, 2.0)
    p4 = transition_probabilities(q, 4.0)
    assert np.allclose(p2 @ p2, p4, atol=1e-8)
    assert np.allclose(p4.sum(axis=1), 1.0)
    assert abs(p4[0, 2]) > 0.0  # death accumulates


def test_mean_sojourn():
    q = np.array([[-0.2, 0.15, 0.05], [0.0, -0.3, 0.3], [0.0, 0.0, 0.0]])
    s = mean_sojourn(q)
    assert abs(s[0] - 5.0) < 1e-9
    assert math.isinf(s[2])


def test_simulate_structure():
    rows = illness_death_simulate(0.15, 0.05, 0.3, n=200, seed=5)
    dests = {d for _t, d, _td, _c in rows}
    assert dests <= {-1, 1, 2}
    assert sum(1 for _t, d, _td, _c in rows if d == 1) > 0


def test_aalen_johansen():
    d = synth_multistate(seed=7, n=600)
    aj = aalen_johansen_id(cast(list[tuple[float, int, float, float]], d["rows"]), np.array([10.0]))
    p = aj[0]
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    p_true = transition_probabilities(np.asarray(d["q_true"]), 10.0)
    assert np.linalg.norm(p - p_true) / np.linalg.norm(p_true) < 0.3


def test_validation():
    with pytest.raises(ValueError):
        ctmc_intensity_fit([(0, 1)], np.array([0.0, 1.0]), n_states=3)
    with pytest.raises(ValueError):
        ctmc_intensity_fit([(0, 0)], np.ones(3))
    with pytest.raises(ValueError):
        transition_probabilities(np.array([[-0.1, 0.05, 0.05], [0, -0.2, 0.2], [0, 0, 0.1]]), 1.0)
    with pytest.raises(ValueError):
        transition_probabilities(np.eye(2), -1.0)
    with pytest.raises(ValueError):
        illness_death_simulate(-0.1, 0.05, 0.3)
    with pytest.raises(ValueError):
        aalen_johansen_id([], np.array([1.0]))


def test_bench_keys():
    out = bench_multistate()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_intensity_relerr"] < 0.25
    assert out["synthetic_pmat_relerr"] < 0.15
    assert out["synthetic_aj_relerr"] < 0.35
    assert out["synthetic_determinism"] == 1.0
