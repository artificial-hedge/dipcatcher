"""Tests for wavelet_modwt (wave-57)."""

import numpy as np
import pytest

from quant_fund.models.wavelet_modwt import (
    bench_wavelet_modwt,
    modwt,
    synth_wavelet,
    wavelet_correlation,
    wavelet_variance,
)


def test_mra_reconstruction_exact() -> None:
    x, _, _ = synth_wavelet(seed=1)
    r = modwt(x, level=5)
    assert "W1" in r and "S_J" in r
    assert r["recon_err"][0] < 1e-8


def test_scale_variance_peak() -> None:
    x, _, wn = synth_wavelet(seed=2)
    vx = wavelet_variance(x, level=6)
    vw = wavelet_variance(wn, level=6)
    assert np.argmax(vx[:4]) == 3
    assert vx[3] > 10 * vw[3]


def test_correlation_at_shared_scale() -> None:
    x, y, _ = synth_wavelet(seed=3)
    assert wavelet_correlation(x, y, j=4) > 0.9


def test_fail_closed() -> None:
    x, _, _ = synth_wavelet(seed=4)
    with pytest.raises(ValueError):
        modwt(x[:20])
    with pytest.raises(ValueError):
        modwt(np.full(200, np.nan))
    with pytest.raises(ValueError):
        modwt(np.ones(200))


def test_bench_schema_and_score() -> None:
    r = bench_wavelet_modwt()
    for k, v in r.items():
        assert np.isfinite(v), k
    assert r["score"] == 1.0
