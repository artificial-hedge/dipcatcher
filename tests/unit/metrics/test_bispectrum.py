"""Tests for higher-order spectral analysis (metrics/bispectrum.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.bispectrum import (
    bench_bispectrum,
    bispectrum_direct,
    hinich_gaussianity_test,
    quadratic_phase_coupling_index,
    synth_gaussian,
    synth_quadratic,
    synth_volterra,
    third_order_cumulants,
)


def test_bispectrum_shape():
    out = bispectrum_direct(synth_gaussian(1024, seed=0), n_seg=64)
    bc = np.asarray(out["bicoh_sq"])
    assert bc.shape == (32, 32)
    assert np.all(bc >= 0)
    assert np.all(bc <= 1.0 + 1e-9)
    assert float(out["n_blocks"]) > 3


def test_bispectrum_zero_for_clean_gaussian():
    out = bispectrum_direct(np.random.default_rng(1).standard_normal(2048), n_seg=64)
    assert float(np.asarray(out["bicoh_sq"]).mean()) < 0.1


def test_hinich_gaussian_passes():
    t = hinich_gaussianity_test(synth_gaussian(2048, seed=2))
    assert t["hinich_p"] > 0.01


def test_hinich_quadratic_rejects():
    t = hinich_gaussianity_test(synth_quadratic(2048, seed=2))
    assert t["hinich_p"] < 0.001


def test_hinich_volterra_vs_gauss():
    tv = hinich_gaussianity_test(synth_volterra(2048, seed=3))
    tg = hinich_gaussianity_test(synth_gaussian(2048, seed=3))
    assert tv["hinich_p"] < tg["hinich_p"]


def test_third_order_cumulants():
    c = third_order_cumulants(synth_quadratic(1024, seed=0), max_lag=5)
    assert c.shape == (6,)
    assert np.all(np.isfinite(c))


def test_qpc_index_bounds():
    g = quadratic_phase_coupling_index(synth_gaussian(1024, seed=0))
    assert 0.0 <= g <= 1.0


def test_bispectrum_deterministic():
    a = hinich_gaussianity_test(synth_gaussian(512, seed=9))["hinich_stat"]
    b = hinich_gaussianity_test(synth_gaussian(512, seed=9))["hinich_stat"]
    assert a == b


def test_input_validation():
    with pytest.raises(ValueError):
        bispectrum_direct(np.zeros(10))
    with pytest.raises(ValueError):
        bispectrum_direct(np.ones(500))  # zero variance
    bad = np.arange(300.0)
    bad[3] = np.inf
    with pytest.raises(ValueError):
        bispectrum_direct(bad)


def test_bench_keys():
    out = bench_bispectrum()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_hinich_p_gauss"] > 0.01
    assert out["synthetic_hinich_p_quad"] < 0.001
    assert out["synthetic_mean_bicoh_quad"] > out["synthetic_mean_bicoh_gauss"]
    assert out["synthetic_determinism"] == 1.0
