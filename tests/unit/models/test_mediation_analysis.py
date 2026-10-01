"""Tests for causal mediation analysis (models/mediation_analysis.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.mediation_analysis import (
    bench_mediation_analysis,
    mediation_analysis,
    synth_mediation,
)


def _panel(**kw):
    return synth_mediation(seed=23, **kw)


def test_acme_recovery():
    d = _panel(a_path=0.6, b_path=0.8)
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=300,
        seed=1,
    )
    assert abs(out["acme"] - 0.48) < 0.15


def test_ade_recovery():
    d = _panel(c_prime=0.2)
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=300,
        seed=1,
    )
    assert abs(out["ade"] - 0.2) < 0.2


def test_interval_covers_truth():
    d = _panel()
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=300,
        seed=1,
    )
    assert out["acme_lo"] <= 0.48 <= out["acme_hi"]
    assert out["ade_lo"] <= 0.2 <= out["ade_hi"]


def test_total_decomposes():
    d = _panel()
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=300,
        seed=1,
    )
    assert abs(out["total"] - (out["acme"] + out["ade"])) < 0.05


def test_zero_mediation_small_acme():
    d = _panel(a_path=0.0)
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=300,
        seed=1,
    )
    assert abs(out["acme"]) < 0.25


def test_covariates_accepted():
    d = _panel()
    rng = np.random.default_rng(0)
    c = rng.normal(0.0, 1.0, np.asarray(d["treat"]).size).reshape(-1, 1)
    out = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        covariates=c,
        n_sims=100,
        seed=1,
    )
    assert math.isfinite(out["acme"])


def test_validation():
    d = _panel()
    t = np.asarray(d["treat"])
    m = np.asarray(d["mediator"])
    y = np.asarray(d["outcome"])
    with pytest.raises(ValueError):
        mediation_analysis(t[:5], m, y)
    with pytest.raises(ValueError):
        mediation_analysis(np.zeros(y.size), m, y)
    with pytest.raises(ValueError):
        mediation_analysis(t, m, y, covariates=np.ones((y.size - 1, 1)))
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        mediation_analysis(t, m, y2)


def test_determinism():
    d = _panel()
    a = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=100,
        seed=7,
    )
    b = mediation_analysis(
        np.asarray(d["treat"]),
        np.asarray(d["mediator"]),
        np.asarray(d["outcome"]),
        n_sims=100,
        seed=7,
    )
    assert a["acme"] == b["acme"]


def test_bench_keys():
    out = bench_mediation_analysis()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
