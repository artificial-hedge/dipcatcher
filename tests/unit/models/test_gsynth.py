"""Tests for generalized synthetic control (models/gsynth.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.gsynth import (
    bench_gsynth,
    gsynth,
    synth_ife_panel,
)


def _panel(**kw):
    return synth_ife_panel(seed=4, **kw)


def test_att_recovery_stationary():
    d = _panel(tau=1.0, drift=0.0)
    est = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert abs(est["att"] - 1.0) < 0.5


def test_att_recovery_drift():
    d = _panel(tau=1.5, drift=3.0)
    est = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert abs(est["att"] - 1.5) < 0.5


def test_null_att_small():
    d = _panel(tau=0.0, drift=2.0)
    est = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert abs(est["att"]) < 0.6


def test_counterfactual_shape():
    d = _panel()
    est = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    cf = np.asarray(est["counterfactual"])
    assert cf.shape[1] == np.asarray(d["y"]).shape[1]
    assert cf.shape[0] == int(np.asarray(d["treated"]).astype(bool).sum())


def test_z_positive_on_effect():
    d = _panel(tau=1.0)
    est = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert est["z"] > 2.0


def test_validation():
    d = _panel()
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"])
    with pytest.raises(ValueError):
        gsynth(np.ones((3, 4)), tr, 2)
    with pytest.raises(ValueError):
        gsynth(y, np.zeros(y.shape[0]), 10)
    with pytest.raises(ValueError):
        gsynth(y, tr, 1)
    with pytest.raises(ValueError):
        gsynth(y, tr, y.shape[1] - 1)
    y2 = y.copy()
    y2[0, 0] = np.nan
    with pytest.raises(ValueError):
        gsynth(y2, tr, 10)


def test_determinism():
    d = _panel()
    a = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    b = gsynth(np.asarray(d["y"]), np.asarray(d["treated"]), int(np.asarray(d["t0"]).item()))
    assert a["att"] == b["att"]


def test_bench_keys():
    out = bench_gsynth()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_att_err"] < 0.5
    assert out["synthetic_beats_did"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
