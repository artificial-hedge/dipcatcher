"""Welch ANOVA and Games-Howell tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.welch_anova import (
    bench_welch_anova,
    games_howell,
    welch_anova,
)


def test_welch_detects_mean_difference():
    rng = np.random.default_rng(0)
    out = welch_anova(
        rng.normal(0.0, 1.0, 30),
        rng.normal(0.0, 2.0, 30),
        rng.normal(1.5, 0.5, 30),
    )
    assert out["p"] < 0.001
    assert out["df2"] > 0


def test_welch_null():
    rng = np.random.default_rng(1)
    out = welch_anova(
        rng.normal(0.0, 1.0, 40),
        rng.normal(0.0, 2.0, 40),
        rng.normal(0.0, 3.0, 40),
    )
    assert out["p"] > 0.05


def test_games_howell_flags_shifted_group():
    rng = np.random.default_rng(2)
    out = games_howell(
        rng.normal(0.0, 1.0, 30),
        rng.normal(0.0, 1.5, 30),
        rng.normal(2.0, 1.0, 30),
    )
    pairs = out["pairs"]
    sig = [(i, j) for i, j, _q, p in pairs if p < 0.05]
    assert any(2 in t for t in sig)


def test_games_howell_null_few_false_positives():
    rng = np.random.default_rng(3)
    out = games_howell(rng.normal(0, 1, 30), rng.normal(0, 1, 30), rng.normal(0, 1, 30))
    pairs = out["pairs"]
    n_sig = sum(1 for _i, _j, _q, p in pairs if p < 0.05)
    assert n_sig <= 1


def test_input_validation():
    with pytest.raises(ValueError):
        welch_anova(np.arange(5.0))
    with pytest.raises(ValueError):
        welch_anova(np.array([1.0]), np.arange(10.0))
    with pytest.raises(ValueError):
        games_howell(np.arange(5.0))


def test_bench_passes():
    out = bench_welch_anova()
    assert out["synthetic_welch_null_p"] > 0.05
    assert out["synthetic_welch_alt_p"] < 0.01
    assert out["synthetic_gh_hit"] == 1.0
    assert out["synthetic_score"] == 1.0


def test_games_howell_p_matches_t_identity_for_two_groups():
    """For k=2, the studentized-range identity P(Q>q) = 2*t.sf(q/sqrt(2), nu)
    pins the exact GH p-value; scaling q by an extra sqrt(2) (a former bug)
    made p an order of magnitude too small."""
    from scipy import stats as _st

    g1 = np.array([0.0, 1.0, 0.5, -0.5, 0.2])
    g2 = np.array([2.0, 3.0, 2.5, 1.5, 2.2])
    out = games_howell(g1, g2)
    pairs = out["pairs"]
    assert len(pairs) == 1
    _i, _j, q, p = pairs[0]
    vi, vj = g1.var(ddof=1), g2.var(ddof=1)
    se2 = vi / g1.size + vj / g2.size
    df = se2**2 / ((vi / g1.size) ** 2 / (g1.size - 1) + (vj / g2.size) ** 2 / (g2.size - 1))
    t_stat = abs(g1.mean() - g2.mean()) / np.sqrt(se2)
    assert np.isclose(q, np.sqrt(2.0) * t_stat)
    assert np.isclose(p, 2.0 * _st.t.sf(t_stat, df), rtol=1e-9)
