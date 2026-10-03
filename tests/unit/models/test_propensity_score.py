"""Tests for propensity-score pipeline (models/propensity_score.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.propensity_score import (
    balance_table,
    bench_propensity_score,
    ipw_ate,
    overlap_ate,
    propensity_score,
    ps_match,
    synth_propensity,
)


def test_ps_in_unit_interval():
    d = synth_propensity(seed=5)
    ps = np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"])
    assert np.all((ps > 0) & (ps < 1))
    assert ps.shape == (len(d["y"]),)


def test_ps_separates_arms():
    d = synth_propensity(seed=6)
    ps = np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"])
    dd = np.asarray(d["d"])
    assert ps[dd == 1].mean() > ps[dd == 0].mean()


def test_ipw_recovers_ate():
    d = synth_propensity(seed=7)
    ps = np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"])
    out = ipw_ate(np.asarray(d["y"]), np.asarray(d["d"]), ps)
    assert abs(out["ate_ipw"] - 1.0) < 0.15


def test_overlap_weights_recover_ate():
    d = synth_propensity(seed=8)
    ps = np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"])
    out = overlap_ate(np.asarray(d["y"]), np.asarray(d["d"]), ps)
    assert abs(out["ato"] - 1.0) < 0.15


def test_matching_recovers_att():
    d = synth_propensity(seed=9)
    out = ps_match(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert abs(out["att"] - 1.0) < 0.25
    assert out["match_rate"] > 0.5


def test_balance_improves():
    d = synth_propensity(seed=10)
    ps = np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"])
    w = np.where(np.asarray(d["d"]) == 1, 1.0 - ps, ps)
    bal = balance_table(np.asarray(d["x"]), np.asarray(d["d"]), w)
    assert bal["max_abs_smd_weighted"] < bal["max_abs_smd_raw"]
    assert bal["max_abs_smd_weighted"] < 0.1


def test_ipw_trims_extreme_ps():
    d = synth_propensity(seed=11)
    ps = np.clip(np.asarray(propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"]), 0, 1)
    out = ipw_ate(np.asarray(d["y"]), np.asarray(d["d"]), ps, trim=0.05)
    assert 0.05 <= out["min_ps_kept"] <= out["max_ps_kept"] <= 0.95


def test_validation():
    with pytest.raises(ValueError):
        propensity_score(np.ones((10, 2)), np.array([0.5] * 10))
    with pytest.raises(ValueError):
        ipw_ate(np.ones(20), np.ones(20), np.full(20, 0.5))
    with pytest.raises(ValueError):
        ps_match(np.ones(6), np.ones(6), np.ones((6, 1)))


def test_determinism():
    d = synth_propensity(seed=12)
    a = propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"]
    b = propensity_score(np.asarray(d["x"]), np.asarray(d["d"]))["ps"]
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_bench_keys():
    out = bench_propensity_score()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_ipw_beats_raw"] == 1.0
    assert out["synthetic_balance_improved"] == 1.0
    assert out["synthetic_determinism"] == 1.0
