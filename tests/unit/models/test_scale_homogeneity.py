"""Scale homogeneity tests (Levene, BF, FK, O'Brien)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.scale_homogeneity import (
    fligner_killeen,
    levene,
    obrien,
)


def _groups(seed=0, scales=(1.0, 1.0, 1.0)):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, s, 50) for s in scales]


def test_levene_accepts_equal_scale():
    out = levene(_groups(), "mean")
    assert out["pvalue"] > 0.05


def test_levene_rejects_scale_shift():
    out = levene(_groups(scales=(1.0, 1.0, 2.5)), "mean")
    assert out["pvalue"] < 0.01


def test_brown_forsythe_robust_center():
    out = levene(_groups(scales=(1.0, 1.0, 2.5)), "median")
    assert out["pvalue"] < 0.01


def test_fligner_killeen():
    rng = np.random.default_rng(5)
    # heavier-tailed group: t(3) vs normal
    g = [rng.normal(0, 1, 60), rng.normal(0, 1, 60), rng.standard_t(3, 60)]
    out = fligner_killeen(g)
    assert out["stat"] >= 0
    same = fligner_killeen(_groups())
    assert same["pvalue"] > 0.05


def test_obrien_rejects():
    out = obrien(_groups(scales=(1.0, 1.0, 2.5)))
    assert out["pvalue"] < 0.01
    same = obrien(_groups())
    assert same["pvalue"] > 0.05


def test_validation():
    with pytest.raises(ValueError):
        levene([np.ones(50)])
    with pytest.raises(ValueError):
        levene([np.ones(3), np.ones(5)])
    with pytest.raises(ValueError):
        fligner_killeen([np.ones(50), np.full(50, np.nan)])
    with pytest.raises(ValueError):
        obrien([np.ones(2), np.ones(2)])  # n_i < 3
