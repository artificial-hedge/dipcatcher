"""Vibroseis sweep correlation (Klauder wavelet) + deconvolution check.

A linear chirp sweep convolved with a reflectivity series is correlated
with the pilot sweep; the autocorrelated sweep (Klauder wavelet) peaks
at each reflector's two-way time.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 894


def linear_sweep(t: np.ndarray, f0: float, f1: float, dur: float) -> np.ndarray:
    """Unit linear chirp from f0 to f1 Hz over `dur` seconds."""
    t = np.asarray(t, dtype=np.float64)
    k = (f1 - f0) / dur
    ph = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.asarray(np.sin(ph))


def correlate(x: np.ndarray, pilot: np.ndarray) -> np.ndarray:
    """Full cross-correlation corr[k] = sum x[k+n] pilot[n]."""
    return np.correlate(
        np.asarray(x, dtype=np.float64), np.asarray(pilot, dtype=np.float64), mode="full"
    )


def bench_vibroseis_sweep(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dt = 0.001
    nt = 4000
    t = np.arange(nt) * dt
    dur = 1.0
    f0, f1 = 5.0, 80.0
    pilot = linear_sweep(t[: int(dur / dt)], f0, f1, dur)
    refl = np.zeros(nt)
    spikes = np.array([1500, 2300, 3100])
    refl[spikes] = np.array([1.0, -0.8, 0.6])
    data = np.convolve(refl, pilot)[:nt] + rng.normal(0, 0.05, nt)
    corr = correlate(data, pilot)
    # lag axis of 'full' correlation: peak for spike at s lands at s + len(pilot)-1
    lp = pilot.size
    score = 0.0
    found = 0
    for s in spikes:
        win = corr[s + lp - 1 - 8 : s + lp - 1 + 9]
        peak = s + lp - 1 - 8 + int(np.argmax(np.abs(win)))
        if abs(peak - (s + lp - 1)) <= 8:
            found += 1
    score += 1.0 if found == 3 else 0.0
    # sign of middle spike preserved (negative reflection coeff -> trough)
    s = spikes[1]
    val = corr[s + lp - 1]
    score += 1.0 if val < 0 else 0.0
    # Klauder sidelobe advantage vs white-noise pilot of same energy
    wn = rng.normal(0, 1, lp)
    wn *= np.sqrt(np.sum(pilot**2) / np.sum(wn**2))
    corr_wn = correlate(data, wn)
    s0 = spikes[0]
    main_sw = abs(corr[s0 + lp - 1])
    side_sw = np.sort(np.abs(corr[s0 + lp - 1 - 200 : s0 + lp - 1 - 30]))[-1]
    main_wn = abs(corr_wn[s0 + lp - 1])
    side_wn = np.sort(np.abs(corr_wn[s0 + lp - 1 - 200 : s0 + lp - 1 - 30]))[-1]
    score += 1.0 if (main_sw / max(side_sw, 1e-9)) > (main_wn / max(side_wn, 1e-9)) else 0.0
    return {"synthetic_vibroseis_sweep": score / 3.0}
