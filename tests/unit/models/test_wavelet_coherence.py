import numpy as np
import pytest

from quant_fund.models.wavelet_coherence import (
    bench_wavelet_coherence,
    synth_coherence,
    wavelet_coherence,
)


def test_bench_wavelet_coherence_passes():
    r = bench_wavelet_coherence()
    assert r["synthetic_score"] == 1.0


def test_coherence_bounded():
    x, y, _ = synth_coherence(seed=5)
    r = wavelet_coherence(x, y)
    c = np.asarray(r["coherence"])
    assert np.all(c >= 0.0) and np.all(c <= 1.0)


def test_shared_band_exceeds_null():
    x, y, indep = synth_coherence(seed=3)
    r = wavelet_coherence(x, y)
    r_n = wavelet_coherence(x, indep)
    band = (r["scales"] >= 24.0) & (r["scales"] <= 48.0)
    assert np.mean(np.asarray(r["coherence"])[band]) > np.mean(np.asarray(r_n["coherence"])[band])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        wavelet_coherence(np.zeros(300), np.zeros(200))


def test_rejects_short_series():
    with pytest.raises(ValueError):
        wavelet_coherence(np.zeros(50), np.zeros(50))
