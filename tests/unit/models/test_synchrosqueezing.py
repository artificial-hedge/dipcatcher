"""Unit tests for quant_fund.models.synchrosqueezing."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.synchrosqueezing import (
    bench_synchrosqueezing,
    cwt,
    phase_transform,
    ridge_frequency,
    synchrosqueeze,
)


def _chirp(n: int = 512, f0: float = 0.04, f1: float = 0.10, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if_true = f0 + (f1 - f0) * t / t[-1]
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / t[-1])
    x = np.cos(phase) + 0.1 * rng.standard_normal(n)
    return x, if_true


def test_cwt_shape_complex() -> None:
    x, _ = _chirp()
    scales = np.geomspace(5.0, 50.0, 20)
    w = cwt(x, scales)
    assert w.shape == (20, 512)
    assert np.iscomplexobj(w)
    assert np.all(np.isfinite(np.abs(w)))


def test_phase_transform_finite_on_ridge() -> None:
    x, _ = _chirp()
    scales = np.geomspace(5.0, 50.0, 20)
    w = cwt(x, scales)
    om = phase_transform(w)
    mag = np.abs(w)
    assert np.isfinite(om[mag > np.quantile(mag, 0.9)]).all()


def test_squeeze_energy_concentrated() -> None:
    x, if_true = _chirp()
    scales = np.geomspace(6.0, 60.0, 40)
    w = cwt(x, scales)
    om = phase_transform(w)
    f_grid, tf = synchrosqueeze(w, om, scales, n_freq=60)
    band = (f_grid > 0.02) & (f_grid < 0.14)
    assert tf[band].sum() / tf.sum() > 0.8


def test_ridge_tracks_chirp() -> None:
    x, if_true = _chirp()
    scales = np.geomspace(6.0, 60.0, 40)
    w = cwt(x, scales)
    om = phase_transform(w)
    f_grid, tf = synchrosqueeze(w, om, scales, n_freq=60)
    ridge = ridge_frequency(tf, f_grid)
    mid = slice(128, 384)
    assert np.median(np.abs(ridge[mid] - if_true[mid])) < 0.03


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        cwt(np.zeros(4), np.geomspace(5, 50, 20))
    with pytest.raises(ValueError):
        cwt(np.full(100, np.nan), np.geomspace(5, 50, 20))


def test_bench_synchrosqueezing_score() -> None:
    out = bench_synchrosqueezing()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_sst_ridge_err"] < 0.03
