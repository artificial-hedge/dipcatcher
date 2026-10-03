"""Tests for poststrat — post-stratification and raking."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.poststrat import (
    bench_poststrat,
    design_effect,
    poststrat_weights,
    raking,
)


def test_poststrat_corrects_margin_bias():
    rng = np.random.default_rng(0)
    shares = np.array([0.3, 0.7])
    means = np.array([0.0, 2.0])
    cells = rng.choice(2, 400, p=np.array([0.6, 0.4]))
    y = means[cells] + 0.2 * rng.standard_normal(400)
    w, _ = poststrat_weights(cells.astype(float), shares, 400)
    w_mean = float((w * y).sum() / w.sum())
    assert abs(w_mean - (shares * means).sum()) < 0.1


def test_raking_matches_margins():
    rng = np.random.default_rng(1)
    n = 500
    design = np.stack([rng.integers(0, 3, n), rng.integers(0, 2, n)], axis=1).astype(float)
    w = raking(design, [np.array([0.2, 0.5, 0.3]), np.array([0.4, 0.6])])
    m1 = np.bincount(design[:, 0].astype(int), weights=w, minlength=3) / w.sum()
    assert np.abs(m1 - np.array([0.2, 0.5, 0.3])).max() < 0.02


def test_deff_formula():
    w = np.exp(0.8 * np.random.default_rng(2).standard_normal(200))
    out = design_effect(w)
    exact = w.size * float((w * w).sum()) / float(w.sum() ** 2)
    assert out["deff"] == pytest.approx(exact)


def test_equal_weights_deff_one():
    out = design_effect(np.ones(100))
    assert out["deff"] == pytest.approx(1.0)


def test_fail_closed_zero_share():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        poststrat_weights(rng.integers(0, 2, 50).astype(float), np.array([0.5, 0.0]), 50)


def test_bench():
    out = bench_poststrat()
    assert out["score"] == 1.0
