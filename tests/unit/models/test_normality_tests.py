"""Normality battery tests (SW, JB, D'Agostino-Pearson)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.normality_tests import (
    dagostino_pearson,
    jarque_bera,
    shapiro_wilk,
)


def test_sw_stat_near_one_for_normal():
    rng = np.random.default_rng(0)
    out = shapiro_wilk(rng.normal(0, 1, 300))
    assert out["stat"] > 0.95
    assert out["pvalue"] > 0.05


def test_sw_rejects_lognormal():
    rng = np.random.default_rng(1)
    out = shapiro_wilk(rng.lognormal(0, 0.7, 200))
    assert out["pvalue"] < 0.01
    assert out["stat"] < 0.9


def test_jb_stat_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 200)
    out = jarque_bera(x)
    # JB = n/6 (S^2 + (K-3)^2/4)
    xc = x - x.mean()
    s2 = float((xc**2).mean())
    s = float((xc**3).mean()) / s2**1.5
    k = float((xc**4).mean()) / s2**2
    assert out["stat"] == pytest.approx(x.size / 6 * (s**2 + (k - 3) ** 2 / 4), rel=1e-9)
    assert out["pvalue"] > 0.05


def test_jb_rejects_heavy_tails():
    rng = np.random.default_rng(3)
    out = jarque_bera(rng.standard_t(3, 500))
    assert out["pvalue"] < 0.01


def test_dp_parts_and_rejection():
    rng = np.random.default_rng(4)
    out = dagostino_pearson(rng.lognormal(0, 0.8, 300))
    assert out["pvalue"] < 0.01
    assert out["z_skew"] > 3  # strong positive skew
    ok = dagostino_pearson(rng.normal(0, 1, 300))
    assert ok["pvalue"] > 0.05
    assert ok["stat"] == pytest.approx(ok["z_skew"] ** 2 + ok["z_kurt"] ** 2, rel=1e-9)


def test_validation():
    with pytest.raises(ValueError):
        jarque_bera(np.ones(5))
    with pytest.raises(ValueError):
        shapiro_wilk(np.array([np.nan] * 30))
    with pytest.raises(ValueError):
        shapiro_wilk(np.zeros(6000))
