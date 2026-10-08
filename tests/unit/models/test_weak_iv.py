"""Tests for weak-instrument robust inference (models/weak_iv.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.weak_iv import (
    anderson_rubin,
    ar_confidence_interval,
    bench_weak_iv,
    conditional_lr,
    first_stage,
    synth_iv,
)


def test_first_stage_strong_vs_weak():
    s = synth_iv(seed=1, pi=0.8)
    w = synth_iv(seed=1, pi=0.05)
    fs = first_stage(np.asarray(s["d"]), np.asarray(s["z"]))
    fw = first_stage(np.asarray(w["d"]), np.asarray(w["z"]))
    assert fs["f_stat"] > 20.0
    assert fw["f_stat"] < fs["f_stat"]
    assert fs["partial_r2"] > fw["partial_r2"]


def test_ar_size_at_true_beta():
    d = synth_iv(seed=2, pi=0.4)
    out = anderson_rubin(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["z"]), 1.0)
    assert out["ar_pvalue"] > 0.05


def test_ar_power_at_false_beta():
    d = synth_iv(seed=3, pi=0.5)
    out = anderson_rubin(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["z"]), 0.0)
    assert out["ar_pvalue"] < 0.01


def test_ar_size_held_under_weak_iv():
    d = synth_iv(seed=4, pi=0.05)
    out = anderson_rubin(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["z"]), 1.0)
    assert out["ar_pvalue"] > 0.01  # exact-size test stays valid


def test_clr_single_instrument():
    d = synth_iv(seed=5, pi=0.6)
    out = conditional_lr(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["z"]), 0.0)
    assert out["clr_pvalue"] < 0.05
    with pytest.raises(ValueError):
        conditional_lr(
            np.asarray(d["y"]),
            np.asarray(d["d"]),
            np.column_stack([d["z"], d["z"]]),
            0.0,
        )


def test_ar_ci_covers_true():
    d = synth_iv(seed=6, pi=0.7)
    ci = ar_confidence_interval(
        np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["z"]), n_grid=401
    )
    assert ci["ci_lo"] <= 1.0 <= ci["ci_hi"]
    assert ci["ci_empty"] == 0.0
    assert 0.0 < ci["ci_width"] < 10.0


def test_weak_iv_widens_ci():
    strong = synth_iv(seed=7, pi=0.8)
    weak = synth_iv(seed=7, pi=0.15)
    cs = ar_confidence_interval(
        np.asarray(strong["y"]), np.asarray(strong["d"]), np.asarray(strong["z"]), n_grid=201
    )
    cw = ar_confidence_interval(
        np.asarray(weak["y"]), np.asarray(weak["d"]), np.asarray(weak["z"]), n_grid=201
    )
    # weak IV should widen or empty the set vs strong
    assert cw["ci_width"] > cs["ci_width"] or cw["ci_empty"] == 1.0 or cw["ci_width"] > 8.0


def test_validation():
    with pytest.raises(ValueError):
        first_stage(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        anderson_rubin(np.ones(10), np.ones(10), np.ones(5), 0.0)
    with pytest.raises(ValueError):
        anderson_rubin(np.ones(30), np.ones(30), np.ones(60), 0.0)


def test_bench_keys():
    out = bench_weak_iv()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_ar_size_ok"] == 1.0
    assert out["synthetic_ar_power"] == 1.0
    assert out["synthetic_ci_covers"] == 1.0
    assert out["synthetic_determinism"] == 1.0


def test_stock_yogo_table_increases_with_k():
    """10% maximal-size critical values rise with instrument count; the
    previous table dropped to 11.04 at k=2 and 22.30/k beyond, so weak
    multi-instrument sets were stamped 'relevant'."""
    rng = np.random.default_rng(7)
    n = 400
    z = rng.normal(0.0, 1.0, (n, 2))
    # pi chosen so the first-stage F lands near the k=2 boundary (~19.9)
    d = z @ np.array([0.20, 0.20]) + rng.normal(0.0, 1.0, n)
    out2 = first_stage(d, z)
    assert out2["stock_yogo_10pct_iv"] == pytest.approx(19.93)
    z3 = rng.normal(0.0, 1.0, (n, 3))
    d3 = z3 @ np.array([0.2, 0.2, 0.2]) + rng.normal(0.0, 1.0, n)
    out3 = first_stage(d3, z3)
    assert out3["stock_yogo_10pct_iv"] == pytest.approx(22.30)
    assert out3["stock_yogo_10pct_iv"] > out2["stock_yogo_10pct_iv"]
    # a mid-teens F must now read as NOT relevant at k=2 (was 11.04)
    assert out2["relevant_10pct"] == 0.0 or out2["f_stat"] > 19.93
