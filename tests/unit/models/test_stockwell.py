import numpy as np
import pytest

from quant_fund.models.stockwell import bench_stockwell, s_transform, st_ridge


def _chirp(rng, n=512):
    t = np.arange(n)
    f0, f1 = 0.04, 0.14
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / t[-1])
    return np.cos(phase) + 0.1 * rng.standard_normal(n)


def test_s_transform_shapes():
    rng = np.random.default_rng(0)
    freqs, s = s_transform(_chirp(rng))
    assert s.shape == (freqs.size, 512)
    assert np.all(freqs >= 0)


def test_s_transform_complex_finite():
    rng = np.random.default_rng(1)
    _, s = s_transform(_chirp(rng))
    assert np.isfinite(s).all()


def test_ridge_tracks_chirp():
    rng = np.random.default_rng(2)
    n = 512
    t = np.arange(n)
    freqs, s = s_transform(_chirp(rng, n))
    ridge = st_ridge(s, freqs)
    if_true = 0.04 + 0.10 * t / t[-1]
    err = np.median(np.abs(ridge[n // 4 : 3 * n // 4] - if_true[n // 4 : 3 * n // 4]))
    assert err < 0.04


def test_ridge_constant_tone():
    t = np.arange(256)
    x = np.cos(2 * np.pi * 0.08 * t)
    freqs, s = s_transform(x)
    ridge = st_ridge(s, freqs)
    assert np.median(np.abs(ridge - 0.08)) < 0.02


def test_s_transform_input_validation():
    with pytest.raises(ValueError):
        s_transform(np.arange(5.0))
    with pytest.raises(ValueError):
        s_transform(np.full(32, np.nan))


def test_bench_stockwell():
    out = bench_stockwell()
    assert out["score"] == 1.0
    assert out["synthetic_stockwell_ridge_err"] < 0.04
