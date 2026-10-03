"""Tests for LP-IV instrumented local projections (models/lp_iv.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.lp_iv import bench_lp_iv, lp_iv, synth_lp_iv


@pytest.fixture
def strong():
    return synth_lp_iv(n=600, seed=4, weak=False)


@pytest.fixture
def weak():
    return synth_lp_iv(n=600, seed=4, weak=True)


def test_synth_shapes(strong):
    assert strong["y"].shape == (600,)
    assert strong["true_irf"].shape == (25,)


def test_lp_iv_output_structure(strong):
    out = lp_iv(strong["y"], strong["x"], strong["z"], horizons=8)
    for k in (
        "horizon",
        "irf",
        "se",
        "ci_low",
        "ci_high",
        "first_stage_f",
        "ar_ci_low",
        "ar_ci_high",
    ):
        assert k in out
        assert out[k].shape == (9,)
    assert np.all(np.isfinite(out["irf"]))
    assert np.all(out["first_stage_f"] > 0)


def test_lp_iv_recovers_irf(strong):
    out = lp_iv(strong["y"], strong["x"], strong["z"], horizons=10)
    truth = strong["true_irf"][:11]
    scale = out["irf"][0] / truth[0]
    relerr = np.linalg.norm(out["irf"] - truth * scale) / np.linalg.norm(truth * scale)
    assert relerr < 0.35
    # contemporaneous impact should be near 1
    assert 0.5 < out["irf"][0] < 1.6


def test_first_stage_f_drops_when_weak(strong, weak):
    f_s = lp_iv(strong["y"], strong["x"], strong["z"], horizons=4)["first_stage_f"]
    f_w = lp_iv(weak["y"], weak["x"], weak["z"], horizons=4)["first_stage_f"]
    assert f_s.mean() > 10 * f_w.mean()


def test_ar_band_widens_under_weak_iv(strong, weak):
    w_s = lp_iv(strong["y"], strong["x"], strong["z"], horizons=4)
    w_w = lp_iv(weak["y"], weak["x"], weak["z"], horizons=4)
    ar_s = np.mean(w_s["ar_ci_high"] - w_s["ar_ci_low"])
    ar_w = np.mean(w_w["ar_ci_high"] - w_w["ar_ci_low"])
    assert ar_w > ar_s


def test_lp_iv_beats_naive_on_mismeasured_x(strong):
    from quant_fund.models.local_projection import local_projection

    truth = strong["true_irf"][:9]
    out = lp_iv(strong["y"], strong["x"], strong["z"], horizons=8)
    naive = local_projection(strong["y"], strong["x"], horizons=8, control_lags=2)
    scale_i = out["irf"][0] / truth[0]
    scale_n = naive["irf"][0] / truth[0]
    err_i = np.linalg.norm(out["irf"] - truth * scale_i)
    err_n = np.linalg.norm(naive["irf"] - truth * scale_n)
    assert err_i < err_n


def test_lp_iv_determinism(strong):
    a = lp_iv(strong["y"], strong["x"], strong["z"], horizons=5)["irf"]
    b = lp_iv(strong["y"], strong["x"], strong["z"], horizons=5)["irf"]
    assert np.array_equal(a, b)


def test_lp_iv_validation():
    with pytest.raises(ValueError):
        lp_iv(np.zeros(10), np.zeros(10), np.zeros(10))
    with pytest.raises(ValueError):
        lp_iv(np.zeros(50), np.zeros(50), np.zeros(40))
    with pytest.raises(ValueError):
        lp_iv(
            np.arange(60.0),
            np.arange(60.0),
            np.arange(60.0),
            horizons=0,
        )
    bad = np.arange(60.0)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        lp_iv(bad, np.arange(60.0), np.arange(60.0))


def test_bench_keys():
    out = bench_lp_iv()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_irf_relerr"] < 0.3
    assert out["synthetic_lpiv_beats_naive"] == 1.0
    assert out["synthetic_first_stage_f_min"] > 50
    assert out["synthetic_first_stage_f_weak"] < 30
    assert out["synthetic_ar_widens_under_weak_iv"] == 1.0
    assert out["synthetic_determinism"] == 1.0
