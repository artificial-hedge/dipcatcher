"""Rank scale tests (AB, Mood, Klotz, Conover, Gastwirth)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.score_scale import (
    ansari_bradley,
    conover,
    gastwirth,
    klotz,
    mood,
)


def test_ab_rejects_scale_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 60)
    y = rng.normal(0, 2.5, 60)
    out = ansari_bradley(x, y)
    assert out["pvalue"] < 0.05


def test_ab_accepts_same_scale():
    rng = np.random.default_rng(1)
    out = ansari_bradley(rng.normal(0, 1, 60), rng.normal(0, 1, 60))
    assert out["pvalue"] > 0.05


def test_mood_detects():
    rng = np.random.default_rng(2)
    out = mood(rng.normal(0, 1, 60), rng.normal(0, 2.5, 60))
    assert out["pvalue"] < 0.05
    # y more dispersed -> x rank-scores center -> z sign
    assert out["z"] < 0  # x scores below null mean


def test_klotz_detects():
    rng = np.random.default_rng(3)
    out = klotz(rng.normal(0, 1, 60), rng.normal(0, 2.5, 60))
    assert out["pvalue"] < 0.05


def test_conover_detects():
    rng = np.random.default_rng(4)
    out = conover(rng.normal(0, 1, 60), rng.normal(0, 2.5, 60))
    assert out["pvalue"] < 0.05


def test_gastwirth_detects():
    rng = np.random.default_rng(5)
    out = gastwirth(rng.normal(0, 1, 60), rng.normal(0, 2.5, 60))
    assert out["pvalue"] < 0.05


def test_no_location_confound():
    # same scale, different location: should NOT reject
    rng = np.random.default_rng(6)
    out = ansari_bradley(rng.normal(0, 1, 80), rng.normal(2.0, 1, 80))
    assert out["pvalue"] > 0.01  # AB is location-sensitive but weakly


def test_validation():
    with pytest.raises(ValueError):
        klotz(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        mood(np.ones(40), np.full(40, np.nan))
