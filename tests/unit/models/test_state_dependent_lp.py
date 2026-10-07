"""Unit tests for quant_fund.models.state_dependent_lp."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.state_dependent_lp import (
    _nw_se,
    bench_state_dependent_lp,
    state_lp,
    transition_prob,
)


def test_transition_prob_bounded() -> None:
    s = np.linspace(-4, 4, 50)
    p = transition_prob(s, c=0.0, gamma=1.5)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert p[-1] > 0.9 > p[0]


def test_transition_prob_monotone() -> None:
    s = np.linspace(-3, 3, 40)
    p = transition_prob(s, c=0.0, gamma=2.0)
    assert np.all(np.diff(p) > 0.0)


def test_nw_se_positive() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    e = rng.standard_normal(300)
    se = _nw_se(x, e, 5)
    assert np.all(se > 0.0)


def test_state_lp_recovers_coefs() -> None:
    rng = np.random.default_rng(2)
    n = 800
    s = rng.standard_normal(n)
    p = transition_prob(s, c=0.0, gamma=2.0)
    shock = rng.standard_normal(n)
    y = (1.0 * (1.0 - p) + 3.0 * p) * shock + 0.3 * rng.standard_normal(n)
    out = state_lp(y, shock, s, horizons=[0])
    assert abs(out["beta_expansion"][0] - 1.0) < 0.5
    assert abs(out["beta_recession"][0] - 3.0) < 0.8
    assert out["se_expansion"][0] > 0.0


def test_state_lp_multi_horizon() -> None:
    rng = np.random.default_rng(5)
    n = 400
    s = rng.standard_normal(n)
    shock = rng.standard_normal(n)
    y = shock + 0.4 * rng.standard_normal(n)
    out = state_lp(y, shock, s, horizons=[0, 1, 2])
    assert out["beta_expansion"].shape == (3,)
    assert np.all(np.isfinite(out["beta_recession"]))


def test_state_lp_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        state_lp(np.zeros(3), np.zeros(3), np.zeros(3), horizons=[1])


def test_bench_state_dependent_lp_score() -> None:
    out = bench_state_dependent_lp()
    assert out["synthetic_score"] == pytest.approx(1.0)
