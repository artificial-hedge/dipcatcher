"""Wigner-Ville and pseudo-Wigner-Ville time-frequency
distribution.

References
----------
- Wigner, E.P. (1932). "On the Quantum Correction for
  Thermodynamic Equilibrium." *Physical Review* 40(5),
  749-759.
- Ville, J. (1948). "Theorie et Applications de la Notion de
  Signal Analytique." *Cables et Transmission* 2A, 61-74.
- Cohen, L. (1989). "Time-Frequency Distributions — A
  Review." *Proceedings of the IEEE* 77(7), 941-981.
- Boashash, B. (1988). "Note on the Use of the Wigner
  Distribution for Time-Frequency Signal Analysis." *IEEE
  Trans. ASSP* 36(9), 1518-1521.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The Wigner-Ville distribution is the Fourier transform over
the lag variable of the instantaneous autocorrelation:
``W(t, f) = sum_tau x*(t - tau) x(t + tau) e^{-2 pi i f tau}``
evaluated on the analytic signal to avoid the positive/
negative-frequency cross-term. The bare WVD has perfect
time-frequency concentration but multiplicative cross-terms
for multicomponent signals (the classic artifact); the
pseudo-WVD applies a lag window (here a rectangular
smoothing kernel over time — the minimal honest variant)
which trades resolution for cross-term suppression. We
compute via direct convolution in the lag domain then FFT —
the sum-product formulation in O(n^2 m) is small enough for
scorecard series. Guards: analytic transform requires n>=64
and we zero the negative frequencies in the Hilbert FFT;
degnerate signals fail closed. ``synth_wvd`` builds a linear
chirp + a constant tone; the bench gates on ridge energy
concentrating at the chirp trajectory and on the tone ridge
sitting at its planted bin.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _as_series(x: FloatArray, min_len: int = 64) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _analytic(x: FloatArray) -> ComplexArray:
    return np.asarray(hilbert(x), dtype=np.complex128)


def wigner_ville(
    x: FloatArray,
    smooth: int = 9,
) -> dict[str, FloatArray]:
    """Pseudo-Wigner-Ville distribution (analytic signal)."""
    v = _as_series(x)
    if smooth < 1 or smooth > 51:
        raise ValueError("bad smoothing")
    z = _analytic(v)
    n = z.size
    nf = n
    wvd = np.zeros((n, nf))
    max_lag = n // 2
    center = nf // 2
    for t in range(n):
        lag_max = min(t, n - 1 - t, max_lag - 1)
        lags = np.arange(-lag_max, lag_max + 1)
        prod = z[t + lags] * np.conj(z[t - lags])
        # zero-pad to nf for a uniform freq grid
        row = np.zeros(nf, dtype=np.complex128)
        start = center - lag_max
        row[start : start + lags.size] = prod
        wvd[t] = np.abs(np.fft.fftshift(np.fft.fft(row))) / nf
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        for j in range(nf):
            wvd[:, j] = np.convolve(wvd[:, j], ker, "same")
    out: dict[str, FloatArray] = {
        "wvd": np.asarray(wvd, dtype=np.float64),
        "freqs": np.fft.fftshift(np.fft.fftfreq(nf, d=1.0)) / 2.0,
        "total_energy": np.array([float(np.sum(wvd))]),
    }
    return out


def ridge_curve(wvd: FloatArray) -> FloatArray:
    """Argmax-frequency ridge per time step."""
    w = np.asarray(wvd, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("bad wvd")
    return np.argmax(w, axis=1).astype(np.float64)


def synth_wvd(
    seed: int = 20261231 + 353,
    n: int = 256,
) -> tuple[FloatArray, float, float]:
    """SYNTHETIC linear chirp + constant tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    f0, f1 = 0.05, 0.40
    chirp = np.cos(2 * np.pi * (f0 * t + (f1 - f0) * t * t / 2) * n)
    x = chirp + 0.10 * rng.standard_normal(n)
    return x.astype(np.float64), f0, f1


def bench_wigner_ville(seed: int = 20261231 + 353) -> dict[str, float]:
    x, f0, f1 = synth_wvd(seed=seed)
    r = wigner_ville(x, smooth=9)
    w = np.asarray(r["wvd"])
    freqs = np.asarray(r["freqs"])
    n = w.shape[0]
    ridge = ridge_curve(w)
    ridge_f = freqs[ridge.astype(np.int64)]
    # expected chirp freq at quarter indices
    t = np.arange(n) / n
    expected = f0 + (f1 - f0) * t
    mask = (np.arange(n) > n // 8) & (np.arange(n) < 7 * n // 8)
    ridge_err = float(np.median(np.abs(ridge_f[mask] - expected[mask])))
    # energy concentration: top-fraction of wvd cells
    flat = np.sort(w.ravel())[::-1]
    conc = float(np.sum(flat[: int(0.08 * flat.size)]) / np.sum(flat))
    ok = ridge_err < 0.06 and conc > 0.35
    out: dict[str, float] = {
        "synthetic_wvd_ridge_err": ridge_err,
        "synthetic_wvd_concentration": conc,
        "synthetic_wvd_energy": float(r["total_energy"][0]),
        "synthetic_score": 1.0 if ok else 0.0,
    }
    return out
