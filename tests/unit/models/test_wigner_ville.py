import numpy as np
import pytest

from quant_fund.models.wigner_ville import (
    bench_wigner_ville,
    ridge_curve,
    synth_wvd,
    wigner_ville,
)


def test_bench_wigner_ville_passes():
    r = bench_wigner_ville()
    assert r["synthetic_score"] == 1.0


def test_ridge_tracks_chirp():
    x, f0, f1 = synth_wvd(seed=3)
    r = wigner_ville(x)
    ridge = ridge_curve(np.asarray(r["wvd"]))
    freqs = np.asarray(r["freqs"])
    n = ridge.size
    t = np.arange(n) / n
    expected = f0 + (f1 - f0) * t
    mid = slice(n // 4, 3 * n // 4)
    err = np.median(np.abs(freqs[ridge.astype(np.int64)][mid] - expected[mid]))
    assert err < 0.08


def test_wvd_nonnegative():
    x, _, _ = synth_wvd(seed=2)
    r = wigner_ville(x)
    assert np.all(np.asarray(r["wvd"]) >= 0.0)


def test_rejects_short():
    with pytest.raises(ValueError):
        wigner_ville(np.zeros(30))


def test_rejects_bad_smooth():
    x, _, _ = synth_wvd(seed=1)
    with pytest.raises(ValueError):
        wigner_ville(x, smooth=0)


def test_high_frequency_tone_not_aliased():
    """A tone at 0.4 cycles/sample must ridge at 0.4, not wrap to a
    negative frequency — the fftshift+fftfreq/2 axis silently aliased
    every physical f > 0.25."""
    x = np.cos(2 * np.pi * 0.4 * np.arange(256))
    r = wigner_ville(x, smooth=1)
    freqs = np.asarray(r["freqs"])
    ridge = freqs[ridge_curve(np.asarray(r["wvd"])).astype(np.int64)]
    assert abs(np.median(ridge[20:-20]) - 0.4) < 0.01
    assert freqs.min() >= 0.0 and freqs.max() < 0.5
