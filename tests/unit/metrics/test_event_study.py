"""Tests for event-study abnormal returns (metrics/event_study.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.event_study import (
    bench_event_study,
    event_study,
    synth_events,
)


def test_abnormal_detected():
    d = synth_events(abnormal=0.015, seed=37)
    out = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    assert float(out["p_patell"]) < 0.01


def test_no_abnormal_keeps_null():
    d = synth_events(abnormal=0.0, seed=37)
    out = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    assert float(out["p_patell"]) > 0.05


def test_car_magnitude():
    d = synth_events(abnormal=0.02, seed=37)
    out = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    assert float(out["mean_car"]) > 0.02


def test_mostly_positive_cars():
    d = synth_events(abnormal=0.02, seed=37)
    out = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    assert float(out["share_pos_car"]) > 0.6


def test_validation():
    d = synth_events(seed=37)
    r, rm, ev = (np.asarray(d[k]) for k in ("r", "r_mkt", "event_idx"))
    with pytest.raises(ValueError):
        event_study(r[:4], rm, ev[:4])  # too few events
    with pytest.raises(ValueError):
        event_study(r, rm[:30], ev)  # market length mismatch
    with pytest.raises(ValueError):
        event_study(r, rm, ev, est_window=10)
    with pytest.raises(ValueError):
        event_study(r, rm, ev, evt_window=30)
    bad = r.copy()
    bad[0, 5] = np.nan
    with pytest.raises(ValueError):
        event_study(bad, rm, ev)


def test_determinism():
    d = synth_events(seed=37)
    a = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    b = event_study(np.asarray(d["r"]), np.asarray(d["r_mkt"]), np.asarray(d["event_idx"]))
    assert float(a["mean_car"]) == float(b["mean_car"])


def test_bench_keys():
    out = bench_event_study()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
