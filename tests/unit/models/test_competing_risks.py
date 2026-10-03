"""Tests for competing-risks CIF + pseudo-values (models/competing_risks.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.competing_risks import (
    aalen_johansen_cif,
    bench_competing_risks,
    pseudo_value_effects,
    synth_competing_risks,
)


def _panel(**kw):
    return synth_competing_risks(seed=24, **kw)


def test_cif_bounded_monotone():
    d = _panel()
    c = aalen_johansen_cif(np.asarray(d["time"]), np.asarray(d["event"]), cause=1)
    cif = np.asarray(c["cif"])
    assert np.all(np.diff(cif) >= -1e-12)
    assert 0.0 <= cif[0] <= cif[-1] <= 1.0


def test_cif_end_reasonable():
    d = _panel()
    c1 = aalen_johansen_cif(np.asarray(d["time"]), np.asarray(d["event"]), cause=1)
    c2 = aalen_johansen_cif(np.asarray(d["time"]), np.asarray(d["event"]), cause=2)
    # both causes present → CIFs sum < 1 minus censoring
    assert 0.3 < float(c1["cif_end"][0]) <= 1.0
    assert float(c2["cif_end"][0]) > 0.05


def test_pseudo_beta_positive_for_cause1():
    d = _panel(effect=0.6)
    pv = pseudo_value_effects(
        np.asarray(d["time"]), np.asarray(d["event"]), np.asarray(d["x"]), cause=1
    )
    assert float(pv["beta"]) > 0.0
    assert float(pv["z"]) > 1.0


def test_pseudo_beta_weak_for_cause2():
    d = _panel(effect=0.6)
    pv = pseudo_value_effects(
        np.asarray(d["time"]), np.asarray(d["event"]), np.asarray(d["x"]), cause=2
    )
    assert abs(float(pv["beta"])) < 0.1


def test_zero_effect_near_zero():
    d = _panel(effect=0.0)
    pv = pseudo_value_effects(
        np.asarray(d["time"]), np.asarray(d["event"]), np.asarray(d["x"]), cause=1
    )
    assert abs(float(pv["beta"])) < 0.08


def test_validation():
    d = _panel()
    t = np.asarray(d["time"])
    e = np.asarray(d["event"])
    x = np.asarray(d["x"])
    with pytest.raises(ValueError):
        aalen_johansen_cif(t[:5], e[:5], cause=1)
    with pytest.raises(ValueError):
        aalen_johansen_cif(t, e, cause=9)
    with pytest.raises(ValueError):
        pseudo_value_effects(t, e, x[:3], cause=1)
    e0 = np.zeros(e.size)
    with pytest.raises(ValueError):
        aalen_johansen_cif(t, e0, cause=1)


def test_determinism():
    d = _panel()
    a = pseudo_value_effects(
        np.asarray(d["time"]), np.asarray(d["event"]), np.asarray(d["x"]), cause=1
    )
    b = pseudo_value_effects(
        np.asarray(d["time"]), np.asarray(d["event"]), np.asarray(d["x"]), cause=1
    )
    assert float(a["beta"]) == float(b["beta"])


def test_bench_keys():
    out = bench_competing_risks()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
