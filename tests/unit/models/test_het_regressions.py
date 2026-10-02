"""Heteroskedasticity tests (GQ, Park, Glejser, BP, White)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.het_regressions import (
    breusch_pagan,
    glejser,
    goldfeld_quandt,
    park,
    white,
)


def _het(seed=0, n=150, het=True):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 3, n)
    sig = 0.3 + 1.3 * x if het else np.full(n, 0.9)
    y = 1 + x + rng.normal(0, 1, n) * sig
    return x[:, None], y


def test_gq_detects_het():
    x, y = _het()
    out = goldfeld_quandt(x, y)
    assert out["pvalue"] < 0.01
    assert out["ratio"] > 1  # hi-x variance > lo-x


def test_gq_accepts_hom():
    x, y = _het(het=False)
    out = goldfeld_quandt(x, y)
    assert out["pvalue"] > 0.01


def test_park_slope_positive():
    x, y = _het()
    out = park(x, y)
    assert out["slope"] > 0
    assert out["pvalue"] < 0.01


def test_glejser_detects():
    x, y = _het()
    out = glejser(x, y)
    assert out["slope"] > 0
    assert out["pvalue"] < 0.01


def test_bp_lm():
    x, y = _het()
    out = breusch_pagan(x, y)
    assert out["pvalue"] < 0.01
    assert out["df"] == 1
    xh, yh = _het(het=False)
    assert breusch_pagan(xh, yh)["pvalue"] > 0.01


def test_white_test():
    x, y = _het()
    out = white(x, y, cross=True)
    assert out["pvalue"] < 0.01
    xh, yh = _het(het=False)
    assert white(xh, yh)["pvalue"] > 0.01


def test_validation():
    with pytest.raises(ValueError):
        breusch_pagan(np.ones(4), np.ones(4))
    with pytest.raises(ValueError):
        goldfeld_quandt(np.ones((20, 2)), np.ones(20))
