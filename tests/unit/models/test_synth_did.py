"""Tests for synthetic difference-in-differences (models/synth_did.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.synth_did import bench_synth_did, synth_did, synth_panel


def test_recovers_tau():
    d = synth_panel(seed=5, tau=1.5)
    est = synth_did(
        np.asarray(d["y"]),
        np.asarray(d["treated"]),
        int(d["t0"]),
        n_placebo=80,
        seed=5,
    )
    assert abs(est["tau"] - 1.5) < 0.5


def test_beats_did_under_factor_selection():
    d = synth_panel(seed=20261231 + 187, tau=1.5)
    y = np.asarray(d["y"])
    tr = np.asarray(d["treated"]).astype(bool)
    t0 = int(d["t0"])
    tau_hat = synth_did(y, tr, t0, n_placebo=60, seed=1)["tau"]
    did = (y[tr, t0:].mean() - y[tr, :t0].mean()) - (y[~tr, t0:].mean() - y[~tr, :t0].mean())
    assert abs(tau_hat - 1.5) < abs(did - 1.5)


def test_weights_on_simplex():
    d = synth_panel(seed=6, tau=1.0)
    est = synth_did(
        np.asarray(d["y"]), np.asarray(d["treated"]), int(d["t0"]), n_placebo=40, seed=6
    )
    wu = np.asarray(est["unit_weights"])
    wt = np.asarray(est["time_weights"])
    assert np.all(wu >= -1e-9) and abs(wu.sum() - 1.0) < 1e-6
    assert np.all(wt >= -1e-9) and abs(wt.sum() - 1.0) < 1e-6
    assert est["unit_weight_max"] <= 1.0 + 1e-9


def test_placebo_se_positive():
    d = synth_panel(seed=7, tau=1.0)
    est = synth_did(
        np.asarray(d["y"]), np.asarray(d["treated"]), int(d["t0"]), n_placebo=60, seed=7
    )
    assert est["se_placebo"] > 0
    assert 0.0 < est["p_placebo"] <= 1.0


def test_null_panel_smaller_tau():
    d = synth_panel(seed=8, tau=0.0)
    est = synth_did(
        np.asarray(d["y"]), np.asarray(d["treated"]), int(d["t0"]), n_placebo=60, seed=8
    )
    assert abs(est["tau"]) < 1.0


def test_validation():
    with pytest.raises(ValueError):
        synth_did(np.ones((3, 4)), np.array([True, False, False]), 2)
    with pytest.raises(ValueError):
        y = np.random.default_rng(0).normal(size=(10, 20))
        synth_did(y, np.zeros(10), 10)  # no treated
    with pytest.raises(ValueError):
        y = np.random.default_rng(0).normal(size=(10, 20))
        tr = np.zeros(10, bool)
        tr[0] = True
        synth_did(y, tr, 1)  # t0 too small


def test_determinism():
    d = synth_panel(seed=9, tau=1.0)
    a = synth_did(np.asarray(d["y"]), np.asarray(d["treated"]), int(d["t0"]), n_placebo=50, seed=9)
    b = synth_did(np.asarray(d["y"]), np.asarray(d["treated"]), int(d["t0"]), n_placebo=50, seed=9)
    assert a["tau"] == b["tau"]


def test_bench_keys():
    out = bench_synth_did()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_sdid_beats_did"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
