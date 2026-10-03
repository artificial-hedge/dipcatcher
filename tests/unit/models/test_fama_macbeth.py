"""Tests for Fama-MacBeth two-pass (models/fama_macbeth.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.fama_macbeth import (
    bench_fama_macbeth,
    cross_section_r2_profile,
    estimate_betas,
    fama_macbeth,
    synth_factor_panel,
)


def _panel(**kw):
    return synth_factor_panel(seed=17, **kw)


def test_betas_shape_and_recovery():
    d = _panel()
    b = estimate_betas(np.asarray(d["returns"]), np.asarray(d["factors"]))
    beta = np.asarray(b["betas"])
    beta_true = np.asarray(d["betas"])
    assert beta.shape == beta_true.shape
    assert np.abs(beta - beta_true).mean() < 0.3


def test_lambda_recovery():
    d = _panel()
    fm = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    lam = np.asarray(fm["lambda"])
    lam_true = np.asarray(d["lambda_true"])
    assert np.abs(lam - lam_true).max() < 0.4


def test_shanken_inflates_se():
    d = _panel()
    fm = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    assert fm["shanken_inflation"] >= 1.0
    assert np.all(np.asarray(fm["se_shanken"]) >= np.asarray(fm["se_fm"]) - 1e-12)


def test_null_lambda_small():
    d = _panel(lam_true=(0.0, 0.0))
    fm = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    assert np.abs(np.asarray(fm["lambda"])).max() < 0.4


def test_r2_profile_shape():
    d = _panel()
    r2 = cross_section_r2_profile(np.asarray(d["returns"]), np.asarray(d["factors"]))
    assert r2.shape == (np.asarray(d["returns"]).shape[0],)
    assert np.all(r2 <= 1.0 + 1e-9)


def test_more_factors_harder():
    d = _panel(k=3, lam_true=(0.5, 0.3, 0.2))
    fm = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    assert np.asarray(fm["lambda"]).size == 3


def test_validation():
    d = _panel()
    r = np.asarray(d["returns"])
    f = np.asarray(d["factors"])
    with pytest.raises(ValueError):
        fama_macbeth(r[:5], f)
    with pytest.raises(ValueError):
        fama_macbeth(r, f[:3])
    with pytest.raises(ValueError):
        fama_macbeth(r[:, :3], f)  # too few assets
    r2 = r.copy()
    r2[0, 0] = np.nan
    with pytest.raises(ValueError):
        fama_macbeth(r2, f)


def test_determinism():
    d = _panel()
    a = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    b = fama_macbeth(np.asarray(d["returns"]), np.asarray(d["factors"]))
    assert np.array_equal(np.asarray(a["lambda"]), np.asarray(b["lambda"]))


def test_bench_keys():
    out = bench_fama_macbeth()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_se_shanken_gt_fm"] == 1.0
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
