"""Unit tests for quant_fund.models.brownian_bridge."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.brownian_bridge import (
    bench_brownian_bridge,
    bridge_mean_var,
    bridge_sample,
    hit_prob_down,
    hit_prob_up,
)


def test_moments_midpoint() -> None:
    m, v = bridge_mean_var(0.0, 1.0, 0.0, 1.0, np.array([0.5]))
    assert m[0] == pytest.approx(0.5)
    assert v[0] == pytest.approx(0.25)


def test_hit_prob_exact_value() -> None:
    p = hit_prob_up(0.0, 0.1, 0.4, 0.5, 1.0)
    assert p == pytest.approx(np.exp(-0.96), rel=1e-9)


def test_hit_prob_symmetry() -> None:
    p_up = hit_prob_up(0.0, 0.0, 0.3, 0.6, 1.0)
    p_dn = hit_prob_down(0.0, 0.0, -0.3, 0.6, 1.0)
    assert p_up == pytest.approx(p_dn)


def test_sample_matches_moments() -> None:
    draws = np.array(
        [bridge_sample(0.0, 0.0, 0.0, 1.0, np.array([0.5]), seed=s)[0] for s in range(4000)]
    )
    assert np.mean(draws) == pytest.approx(0.0, abs=0.03)
    assert np.var(draws) == pytest.approx(0.25, abs=0.02)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        bridge_mean_var(0.0, 1.0, 0.0, 1.0, np.array([1.5]))
    with pytest.raises(ValueError):
        hit_prob_up(0.0, 1.0, 0.5, 0.5, 1.0)
    with pytest.raises(ValueError):
        hit_prob_down(0.0, 1.0, 0.5, 0.5, 1.0)


def test_bench_score() -> None:
    out = bench_brownian_bridge()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_bb_err"] < 0.02
