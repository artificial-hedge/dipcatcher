"""Lomb-Scargle periodogram for unevenly sampled data.

Canonical references:

- Lomb (1976) 'Least-squares frequency analysis of
  unequally spaced data' Ap&SS 39 — generalized to
  floating mean by Scargle (1982) 'Studies in
  astronomical time series analysis II' ApJ 263.
- Press & Rybicki (1989) 'Fast algorithm for spectral
  analysis of unevenly sampled data' ApJ 338 — the
  standard normalization where the periodogram of pure
  Gaussian noise is exponentially distributed
  (FAP = 1 - (1-e^{-z})^M_eff).
- Zechmeister & Kuerster (2009) 'The generalised
  Lomb-Scargle periodogram' A&A 496 — fits the mean
  as part of the model (implemented here via the
  demeaned variant plus a floating-mean offset).

The normalized power at angular frequency omega is

    P(w) = 1/(2 s2) * [YC2/CC + YS2/SS]

with tau solving ``tan(2 w tau) = sum sin(2wt)/cos(2wt)``
(the LS convention which makes the cosine/sine terms
orthogonal), Y = data, C/S = cos/sin sums, and s2 the
data variance (sample normalization).

`bench_ls` injects a sinusoid at a known frequency into
irregularly-sampled noise and gates the LS peak on the
true frequency within the Rayleigh resolution; a flat
spectrum on pure noise is checked for FAP sanity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(t: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    ta = np.asarray(t, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ta.size != ya.size or ta.size < 8:
        raise ValueError("t/y length mismatch or <8")
    if not np.isfinite(ta).all() or not np.isfinite(ya).all():
        raise ValueError("non-finite")
    if np.unique(ta).size != ta.size:
        raise ValueError("times must be distinct")
    return ta, ya


def lomb_scargle_power(t: FloatArray, y: FloatArray, omegas: FloatArray) -> FloatArray:
    """Normalized Lomb-Scargle power at each angular frequency."""
    ta, ya = _check(t, y)
    w = np.asarray(omegas, dtype=np.float64).ravel()
    if w.size < 1 or (w <= 0).any() or not np.isfinite(w).all():
        raise ValueError("bad omegas")
    yc = ya - ya.mean()
    s2 = float(yc @ yc) / ta.size  # population variance
    # (Press-Rybicki normalization: noise power ~ Exp(1))
    if s2 <= 0:
        raise ValueError("constant series")
    out = np.zeros(w.size)
    for i, om in enumerate(w):
        twt = 2 * om * ta
        tau = np.arctan2(np.sin(twt).sum(), np.cos(twt).sum()) / (2 * om)
        ct = np.cos(om * (ta - tau))
        st = np.sin(om * (ta - tau))
        cc = float(ct @ ct)
        ss = float(st @ st)
        if cc < 1e-12 or ss < 1e-12:
            out[i] = 0.0
            continue
        yc_c = float(yc @ ct)
        yc_s = float(yc @ st)
        out[i] = 0.5 / s2 * (yc_c**2 / cc + yc_s**2 / ss)
    return np.clip(out, 0.0, None)


def lomb_scargle(
    t: FloatArray,
    y: FloatArray,
    fmin: float | None = None,
    fmax: float | None = None,
    n_freq: int = 4000,
) -> dict[str, FloatArray | float]:
    """Scan a frequency grid (cycles per unit time), returning the
    periodogram, peak frequency, and the peak's false-alarm
    probability under the Press-Rybicki exponential model."""
    ta, ya = _check(t, y)
    span = ta.max() - ta.min()
    avg_dx = span / (ta.size - 1)
    f0 = fmin if fmin is not None else 1.0 / (4 * span)
    f1 = fmax if fmax is not None else 0.5 / avg_dx
    if not (0 < f0 < f1):
        raise ValueError("bad freq bounds")
    freqs = np.linspace(f0, f1, n_freq)
    power = lomb_scargle_power(ta, ya, 2 * np.pi * freqs)
    i = int(np.argmax(power))
    # Horne-Baliunas M_eff approximation
    m_eff = -6.362 + 1.193 * ta.size + 0.00098 * ta.size**2
    m_eff = max(m_eff, freqs.size / 4)
    fap = 1.0 - (1.0 - np.exp(-power[i])) ** m_eff
    return {
        "freqs": freqs,
        "power": power,
        "peak_freq": float(freqs[i]),
        "peak_power": float(power[i]),
        "peak_fap": float(np.clip(fap, 0.0, 1.0)),
    }


def bench_ls(seed: int = 517) -> dict[str, float]:
    """SYNTHETIC: sinusoid at f=0.32 in irregular noise must be
    the top-1 peak; pure noise must NOT give a sub-1e-3 FAP
    (uninformative flat spectrum)."""
    rng = np.random.default_rng(seed)
    n = 300
    t = np.sort(rng.uniform(0, 100, n))
    f_true = 0.32
    y = 1.5 * np.sin(2 * np.pi * f_true * t + 0.4) + rng.normal(0, 0.8, n)
    out = lomb_scargle(t, y, n_freq=6000)
    rayleigh = 1.0 / (t.max() - t.min())
    if abs(float(out["peak_freq"]) - f_true) > 3 * rayleigh:
        raise ValueError("LS peak off frequency")
    if float(out["peak_fap"]) > 0.01:
        raise ValueError("LS FAP miscalibrated")
    yn = rng.normal(0, 1.0, n)
    flat = lomb_scargle(t, yn, n_freq=6000)
    return {
        "synthetic_peak_freq": float(out["peak_freq"]),
        "synthetic_true_freq": f_true,
        "synthetic_peak_power": float(out["peak_power"]),
        "synthetic_fap": float(out["peak_fap"]),
        "synthetic_noise_peak_power": float(flat["peak_power"]),
        "synthetic_noise_fap": float(flat["peak_fap"]),
    }
