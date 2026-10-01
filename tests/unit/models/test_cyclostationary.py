import numpy as np
import pytest

from quant_fund.models.cyclostationary import (
    bench_cyclostationary,
    cyclic_autocorrelation,
    scd_peak,
)


def _am(rng, n=1024, alpha=0.05, f0=0.15):
    t = np.arange(n)
    return (1.0 + 0.8 * np.cos(2 * np.pi * alpha * t)) * np.cos(
        2 * np.pi * f0 * t
    ) + 0.4 * rng.standard_normal(n)


def test_scd_peak_shape():
    rng = np.random.default_rng(0)
    alphas = np.linspace(0.01, 0.12, 40)
    out = scd_peak(_am(rng), alphas)
    assert out["scd"].shape == (40,)
    assert 0.01 <= out["alpha_peak"] <= 0.12


def test_scd_finds_cycle():
    rng = np.random.default_rng(1)
    alphas = np.linspace(0.01, 0.12, 60)
    out = scd_peak(_am(rng, n=2048, alpha=0.05), alphas, seg_len=128)
    assert abs(out["alpha_peak"] - 0.05) < 0.01


def test_scd_white_noise_floor():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(1024)
    alphas = np.linspace(0.01, 0.12, 40)
    out = scd_peak(x, alphas)
    scd = np.asarray(out["scd"])
    # white noise: profile should be flat (low contrast)
    assert scd.max() / np.median(scd) < 3.0


def test_cyclic_autocorr_shape():
    rng = np.random.default_rng(3)
    ca = cyclic_autocorrelation(_am(rng), alpha=0.05)
    assert ca.ndim == 1
    assert np.isfinite(ca).all()


def test_scd_input_validation():
    with pytest.raises(ValueError):
        scd_peak(np.ones(20), np.array([0.01, 0.02, 0.03]))
    with pytest.raises(ValueError):
        cyclic_autocorrelation(np.full(40, np.nan), 0.05)


def test_bench_cyclostationary():
    out = bench_cyclostationary()
    assert out["score"] == 1.0
    assert out["synthetic_cyclostat_alpha_err"] < 0.01
