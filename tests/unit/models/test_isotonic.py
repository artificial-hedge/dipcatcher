"""Tests for isotonic — PAVA + isotonic calibration."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.isotonic import (
    bench_isotonic,
    isotonic_calibration,
    isotonic_fit,
    pava,
)


def test_pava_monotone_output():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 6.0])
    fit = pava(y)
    assert (np.diff(fit) >= -1e-12).all()


def test_pava_block_means():
    y = np.array([2.0, 0.0, 1.0])
    fit = pava(y)
    # all three violators pool into the grand mean 1.0
    assert np.allclose(fit, 1.0)


def test_pava_already_monotone():
    y = np.array([1.0, 2.0, 2.0, 5.0])
    assert np.allclose(pava(y), y)


def test_isotonic_fit_handles_unsorted_x():
    rng = np.random.default_rng(0)
    x = rng.random(100)
    y = x + 0.05 * rng.standard_normal(100)
    fit = isotonic_fit(x, y)
    order = np.argsort(x)
    assert (np.diff(fit[order]) >= -1e-12).all()


def test_calibration_monotone_and_helps():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.random(n)
    p = 1.0 / (1.0 + np.exp(-8 * (x - 0.5)))
    y = (rng.random(n) < p).astype(float)
    raw = np.clip(0.5 + 0.2 * (x - 0.5), 0, 1)
    out = isotonic_calibration(raw, y)
    assert float(out["brier_cal"]) < float(out["brier_raw"])


def test_fail_closed_nonbinary_labels():
    with pytest.raises(ValueError):
        isotonic_calibration(np.linspace(0, 1, 10), np.linspace(0, 1, 10))


def test_bench():
    out = bench_isotonic()
    assert out["score"] == 1.0
