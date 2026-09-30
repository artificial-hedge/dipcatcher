"""Tests for risk/decay_watch.py — anytime-valid decay e-process."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.decay_watch import decay_demo, decay_eprocess


def test_decaying_stream_alarms():
    rng = np.random.default_rng(1)
    r = rng.normal(-0.004, 0.01, 400)
    out = decay_eprocess(r, 0.02)
    assert out["alarmed"] and out["e_final"] >= out["alarm_level"]


def test_clean_stream_silent():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 400)
    out = decay_eprocess(r, 0.02)
    assert not out["alarmed"]


def test_false_alarm_rate_bounded():
    # under strict null (mean exactly 0), alarm share must stay <= ~alpha-ish;
    # 200 trials, alpha=0.2 for power — empirical null control
    rng = np.random.default_rng(7)
    fires = 0
    n_trials = 200
    for _ in range(n_trials):
        r = rng.normal(0.0, 0.01, 300)
        if decay_eprocess(r, 0.02, alpha=0.2, bet=0.5, adaptive=False)["alarmed"]:
            fires += 1
    assert fires / n_trials <= 0.25  # generous cap over alpha=0.2 for MC noise


def test_adaptive_bet_alarms_on_drift():
    rng = np.random.default_rng(3)
    r = rng.normal(-0.003, 0.008, 300)
    adapt = decay_eprocess(r, 0.02, bet=None, adaptive=True)
    assert adapt["alarmed"]  # plug-in bet self-tunes under persistent drift


def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        decay_eprocess(np.array([]), 0.02)
    with pytest.raises(ValueError, match="finite"):
        decay_eprocess(np.array([0.01, np.nan]), 0.02)
    with pytest.raises(ValueError, match="bound"):
        decay_eprocess(np.array([0.01]), 0.0)
    with pytest.raises(ValueError, match="alpha"):
        decay_eprocess(np.array([0.01]), 0.02, alpha=1.5)


def test_demo_seals_and_both_legs_behave():
    out = decay_demo(seed=4)
    assert out["receipt_sha256"] and out["data_label"] == "SYNTHETIC"
    assert out["claim"]["decaying_stream"]["alarmed"]
    assert not out["claim"]["clean_stream"]["alarmed"]
