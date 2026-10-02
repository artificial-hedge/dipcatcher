"""Lomb-Scargle tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.lomb_scargle import (
    bench_ls,
    lomb_scargle,
    lomb_scargle_power,
)


def test_power_nonneg():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 10, 50))
    y = rng.normal(0, 1, 50)
    p = lomb_scargle_power(t, y, np.linspace(0.1, 5, 100))
    assert (p >= 0).all() and np.isfinite(p).all()


def test_recovers_frequency():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 50, 200))
    y = np.sin(2 * np.pi * 0.5 * t)
    out = lomb_scargle(t, y, n_freq=4000)
    assert out["peak_freq"] == pytest.approx(0.5, abs=0.05)


def test_noise_fap_not_tiny():
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0, 50, 200))
    out = lomb_scargle(t, rng.normal(0, 1, 200), n_freq=4000)
    # flat spectrum should not be "significant" at 1e-6
    assert out["peak_fap"] > 1e-6


def test_strong_signal_fap_tiny():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 50, 200))
    y = 3.0 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.5, 200)
    out = lomb_scargle(t, y, n_freq=4000)
    assert out["peak_fap"] < 0.01


def test_bench_ls():
    out = bench_ls()
    assert abs(out["synthetic_peak_freq"] - out["synthetic_true_freq"]) < 0.05
