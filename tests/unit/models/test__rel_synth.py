"""Probes for _rel_synth (reliability fixtures)."""

import numpy as np
import pytest

from quant_fund.models._rel_synth import (
    ARR_TEMPS,
    WB_BETA,
    WB_CENSOR,
    arrhenius_lives,
    fault_tree,
    fault_tree_top_prob,
    mc_system_state,
    weibull_lifetimes,
)


def test_weibull_shapes_censoring_consistency():
    t, f = weibull_lifetimes(0, n=200)
    assert t.shape == f.shape == (200,)
    assert (t <= WB_CENSOR).all()
    assert (t[f] <= WB_CENSOR).all()
    assert (t[~f] == WB_CENSOR).all()
    assert 0.3 < f.mean() < 0.9  # mixture of failed and censored


def test_weibull_deterministic_and_median():
    t1, f1 = weibull_lifetimes(3, n=500)
    t2, f2 = weibull_lifetimes(3, n=500)
    np.testing.assert_array_equal(t1, t2)
    # Weibull median = eta * (ln2)^(1/beta)
    med = 1200.0 * np.log(2) ** (1 / WB_BETA)
    assert np.median(t1) == pytest.approx(med, rel=0.15)


@pytest.mark.parametrize("n", [0, -10])
def test_weibull_hostile(n):
    with pytest.raises(ValueError):
        weibull_lifetimes(0, n=n)


def test_arrhenius_lives_structure():
    temps, lives = arrhenius_lives(0, n=20)
    assert temps.shape == lives.shape == (20 * len(ARR_TEMPS),)
    assert (lives > 0).all()
    # hotter stress → shorter life (group means strictly decreasing)
    means = [lives[temps == t].mean() for t in ARR_TEMPS]
    assert means[0] > means[1] > means[2]


def test_arrhenius_hostile():
    with pytest.raises(ValueError):
        arrhenius_lives(0, n=0)


def test_fault_tree_prob_matches_structure():
    tree = fault_tree()
    assert tree["gate"] == "or" and len(tree["children"]) == 3
    p = fault_tree_top_prob()
    # brute: P(ab ∪ cd ∪ e) = 1-(1-.02)(1-.015)(1-.05)
    assert p == pytest.approx(1 - (1 - 0.02) * (1 - 0.015) * (1 - 0.05))


def test_mc_system_state_no_rng_leak():
    assert mc_system_state() == 2
    assert mc_system_state(up=1) == 1
