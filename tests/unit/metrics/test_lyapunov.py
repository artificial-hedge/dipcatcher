"""Tests for Lyapunov-dynamics metrics (metrics/lyapunov.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.lyapunov import (
    bench_lyapunov,
    cao_dimension,
    embed_series,
    false_nearest_neighbors,
    largest_lyapunov,
    synth_henon,
    synth_logistic,
    synth_periodic,
    synth_white,
)


def test_embed_series_shape():
    x = np.arange(16, dtype=float)
    emb = embed_series(x, dim=3, tau=2)
    assert emb.shape == (16 - 4, 3)
    assert np.allclose(emb[0], [0.0, 2.0, 4.0])
    assert np.allclose(emb[5], [5.0, 7.0, 9.0])


def test_embed_validation():
    with pytest.raises(ValueError):
        embed_series(np.arange(10.0), dim=0)
    with pytest.raises(ValueError):
        embed_series(np.arange(10.0), dim=3, tau=0)
    with pytest.raises(ValueError):
        embed_series(np.arange(10.0), dim=10, tau=3)


def test_lyapunov_logistic():
    out = largest_lyapunov(synth_logistic(seed=1), dim=3, tau=1)
    assert abs(out["lyap"] - math.log(2.0)) < 0.15


def test_lyapunov_henon():
    out = largest_lyapunov(synth_henon(seed=2), dim=3, tau=1)
    assert abs(out["lyap"] - 0.4192) < 0.15


def test_lyapunov_periodic_nonpositive():
    out = largest_lyapunov(synth_periodic(seed=3), dim=3, tau=6)
    assert out["lyap"] < 0.05


def test_fnn_henon():
    fnn = false_nearest_neighbors(synth_henon(seed=4), max_dim=5)
    assert fnn[0] > 0.4  # dim 1 too small
    assert fnn[2] < 0.1  # dim 3 unfolds


def test_cao_henon():
    out = cao_dimension(synth_henon(seed=5), max_dim=5)
    assert 2 <= out["dim_cao"] <= 4


def test_cao_white_no_saturation():
    out = cao_dimension(synth_white(seed=6), max_dim=5)
    # noise: E1 never saturates
    assert out["dim_cao"] >= 5


def test_lyapunov_validation():
    with pytest.raises(ValueError):
        largest_lyapunov(np.arange(30.0))
    with pytest.raises(ValueError):
        largest_lyapunov(np.array([np.nan] * 100))
    with pytest.raises(ValueError):
        false_nearest_neighbors(np.ones(200))


def test_determinism():
    a = largest_lyapunov(synth_logistic(seed=9), dim=3, tau=1)["lyap"]
    b = largest_lyapunov(synth_logistic(seed=9), dim=3, tau=1)["lyap"]
    assert a == b


def test_bench_keys():
    out = bench_lyapunov()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_lyap_logistic_err"] < 0.15
    assert out["synthetic_lyap_henon_err"] < 0.15
    assert out["synthetic_lyap_periodic"] < 0.05
    assert out["synthetic_lyap_chaos_ordering"] == 1.0
    assert out["synthetic_fnn_dim1_frac"] > out["synthetic_fnn_dim2_frac"]
    assert out["synthetic_determinism"] == 1.0
