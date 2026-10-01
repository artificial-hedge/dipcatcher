"""Tests for honest DiD sensitivity (models/honest_did.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.honest_did import (
    bench_honest_did,
    breakdown_mbar,
    event_study_betas,
    honest_did_sd,
    synth_es,
)


def _es(seed: int = 14, **kw):
    d = synth_es(seed=seed, **kw)
    return event_study_betas(
        np.asarray(d["y"]),
        np.asarray(d["unit"]),
        np.asarray(d["time"]),
        np.asarray(d["g"]),
        rel_periods=(-6, -5, -4, -3, -2, -1, 0, 1, 2),
    )


def test_event_study_recovers_post_effect():
    es = _es(tau=1.0)
    rp = np.asarray(es["rel_periods"])
    b = np.asarray(es["beta"])
    assert abs(b[rp == 1][0] - 1.0) < 0.4


def test_pre_betas_flat_under_parallel_trends():
    es = _es(tau=1.0, pre_trend=0.0)
    rp = np.asarray(es["rel_periods"])
    b = np.asarray(es["beta"])
    assert np.abs(b[rp < 0]).max() < 0.6


def test_honest_ci_significant_clean():
    es = _es(tau=1.0, pre_trend=0.0)
    out = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=1.0)
    assert out["significant"] == 1.0
    assert out["ci_lo"] < out["ci_hi"]


def test_breakdown_collapses_under_pretrend():
    es_clean = _es(tau=1.0, pre_trend=0.0)
    es_pt = _es(seed=15, tau=1.0, pre_trend=0.5)
    mb_clean = breakdown_mbar(es_clean["beta"], es_clean["se"], es_clean["rel_periods"])
    mb_pt = breakdown_mbar(es_pt["beta"], es_pt["se"], es_pt["rel_periods"])
    assert (mb_clean if math.isfinite(mb_clean) else 6.0) > (mb_pt if math.isfinite(mb_pt) else 6.0)


def test_mbar_zero_is_standard_ci():
    es = _es()
    out0 = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=0.0)
    assert out0["bias_bound"] == 0.0
    assert abs(out0["ci_hi"] - out0["beta"] - out0["half_length"] / 2) < 1e-9


def test_wider_mbar_widens_ci():
    es = _es()
    a = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=0.5)
    b = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1, mbar=2.0)
    assert b["half_length"] > a["half_length"]


def test_validation():
    es = _es()
    with pytest.raises(ValueError):
        honest_did_sd(es["beta"][:3], es["se"][:3], es["rel_periods"][:3])
    with pytest.raises(ValueError):
        honest_did_sd(np.ones(6), -np.ones(6), np.arange(6.0) - 3, post_rp=1)
    with pytest.raises(ValueError):
        honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=7)
    with pytest.raises(ValueError):
        event_study_betas(np.ones(4), np.ones(4), np.ones(4), np.ones(4))


def test_determinism():
    es = _es()
    a = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1)
    b = honest_did_sd(es["beta"], es["se"], es["rel_periods"], post_rp=1)
    assert a == b


def test_bench_keys():
    out = bench_honest_did()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_breakdown_shrinks"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
