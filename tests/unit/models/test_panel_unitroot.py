"""Tests for panel unit-root tests (models/panel_unitroot.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.panel_unitroot import (
    bench_panel_unitroot,
    panel_unitroot,
    synth_panel_ar,
)


def test_stationary_rejected():
    pan = synth_panel_ar(rho=0.85, seed=37)
    out = panel_unitroot(pan)
    assert float(out["ips_p"]) < 0.05
    assert float(out["llc_p"]) < 0.05


def test_unitroot_kept():
    pan = synth_panel_ar(rho=1.0, seed=37)
    out = panel_unitroot(pan)
    assert float(out["ips_p"]) > 0.01
    assert float(out["llc_p"]) > 0.01


def test_rho_pooled_reasonable():
    pan = synth_panel_ar(rho=0.9, seed=37)
    out = panel_unitroot(pan)
    # pooled (rho-1) coefficient should be negative under stationarity
    assert float(out["rho_pooled"]) < 0.05


def test_borderline_panel():
    pan = synth_panel_ar(rho=0.96, seed=37)
    out = panel_unitroot(pan)
    assert math.isfinite(float(out["ips_stat"]))
    assert math.isfinite(float(out["llc_stat"]))


def test_validation():
    pan = synth_panel_ar(seed=37)
    with pytest.raises(ValueError):
        panel_unitroot(pan[:5])
    with pytest.raises(ValueError):
        panel_unitroot(pan[:, :20])
    bad = pan.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        panel_unitroot(bad)
    flat = np.tile(np.arange(120, dtype=np.float64), (12, 1))
    flat[:, :] = 1.0
    with pytest.raises(ValueError):
        panel_unitroot(flat)


def test_determinism():
    pan = synth_panel_ar(rho=0.85, seed=37)
    a = panel_unitroot(pan)
    b = panel_unitroot(pan)
    assert float(a["ips_p"]) == float(b["ips_p"])


def test_bench_keys():
    out = bench_panel_unitroot()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
