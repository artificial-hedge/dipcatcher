"""MUSIC/ESPRIT subspace estimation tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.music_esprit import (
    bench_music_esprit,
    esprit,
    music_frequencies,
    music_spectrum,
)


def _mixture(seed: int = 7, n: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    f = np.array([0.08, 0.21, 0.34])
    return (
        np.sin(2 * np.pi * f[0] * t)
        + 0.7 * np.sin(2 * np.pi * f[1] * t + 1.0)
        + 0.5 * np.sin(2 * np.pi * f[2] * t + 2.0)
        + 0.4 * rng.standard_normal(n)
    )


def test_music_resolves_three_modes():
    x = _mixture()
    f = music_frequencies(x, m=48, p=6)
    truth = np.array([0.08, 0.21, 0.34])
    matched = [np.abs(f - tv).min() for tv in truth]
    assert max(matched) < 0.01


def test_esprit_resolves_three_modes():
    x = _mixture()
    f = esprit(x, m=48, p=6, tls=True)
    truth = np.array([0.08, 0.21, 0.34])
    matched = [np.abs(f - tv).min() for tv in truth]
    assert max(matched) < 0.01


def test_spectrum_peaks_at_modes():
    x = _mixture()
    w, spec = music_spectrum(x, m=48, p=6, n_grid=1024)
    for tv in (0.08, 0.21, 0.34):
        i = np.argmin(np.abs(w - tv))
        lo, hi = max(i - 3, 0), min(i + 4, w.size)
        assert spec[i] >= spec[max(i - 15, 0)] * 0.5
        assert spec[i] > np.median(spec[lo:hi])


def test_deterministic():
    x = _mixture()
    assert np.array_equal(esprit(x, 40, 6), esprit(x, 40, 6))


def test_fail_closed():
    with pytest.raises(ValueError):
        music_frequencies(np.arange(20.0), m=40, p=6)
    with pytest.raises(ValueError):
        music_frequencies(np.ones(200), m=10, p=20)
    with pytest.raises(ValueError):
        esprit(np.ones(200) * np.nan, m=40, p=6)


def test_bench_passes():
    out = bench_music_esprit()
    assert out["synthetic_music_max_f_err"] < 0.01
    assert out["synthetic_esprit_max_f_err"] < 0.01
