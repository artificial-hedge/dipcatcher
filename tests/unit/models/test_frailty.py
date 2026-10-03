"""Tests for shared gamma frailty model (models/frailty.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.frailty import bench_frailty, frailty_fit, synth_frailty


def test_beta_recovered():
    d = synth_frailty(beta_x=0.7, seed=37)
    out = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    assert abs(float(out["beta_x"]) - 0.7) < 0.25


def test_theta_detected():
    d = synth_frailty(theta=0.8, seed=37)
    out = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    assert float(out["theta"]) > 0.3


def test_homogeneous_theta_small():
    d = synth_frailty(theta=0.01, seed=37)
    out = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    assert float(out["theta"]) < 0.3


def test_frailty_beats_null():
    d = synth_frailty(theta=1.0, seed=37)
    out = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    assert float(out["lr_frailty"]) > 5.0


def test_validation():
    d = synth_frailty(seed=37)
    t, e, x, c = (np.asarray(d[k]) for k in ("time", "event", "x", "cluster"))
    with pytest.raises(ValueError):
        frailty_fit(-np.abs(t), e, x, c)  # nonpositive times
    with pytest.raises(ValueError):
        frailty_fit(t[:40], e[:40], x[:40], c[:40])
    with pytest.raises(ValueError):
        frailty_fit(t, e * 2, x, c)  # event not 0/1
    with pytest.raises(ValueError):
        frailty_fit(t, e, x[:, :1] * 0.0, c)  # constant covariate
    with pytest.raises(ValueError):
        frailty_fit(t, e, x, np.arange(t.size, dtype=float))  # singletons
    bad = t.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        frailty_fit(bad, e, x, c)


def test_determinism():
    d = synth_frailty(seed=37)
    a = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    b = frailty_fit(
        np.asarray(d["time"]),
        np.asarray(d["event"]),
        np.asarray(d["x"]),
        np.asarray(d["cluster"]),
    )
    assert float(a["theta"]) == float(b["theta"])


def test_bench_keys():
    out = bench_frailty()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
