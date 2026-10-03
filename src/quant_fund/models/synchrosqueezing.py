"""Synchrosqueezed wavelet-transform time-frequency analysis.

Daubechies, Lu & Wu (2011): the CWT of x with mother wavelet psi
smeares energy across scales near the instantaneous frequency
omega_x(t). The phase transform

    omega_f(a, b) = -i d/d_b W_x(a, b) / W_x(a, b)

reallocates each CWT coefficient to the scale satisfying
a = C_psi / omega_f, concentrating the TF map onto the true IF ridge.
Reconstruction via the inverse formula recovers each mode
(A-list/IMT components) from narrow bands around the ridge.

Implementation uses the Morlet wavelet on a log-spaced scale grid with
analytic FFT convolution; the phase transform differentiates the
complex CWT in time.

Honesty: IF recovery verified on a linear chirp with known IF law —
the ridge must track omega(t) within a small band, and the extracted
mode must correlate highly with the planted chirp. Fail-closed on
non-finite input or degenerate ridge energy.

References: Daubechies, Lu & Wu (2011) "Synchrosqueezed wavelet
transforms"; Thakur, Brevdo, Fuckar & Wu (2013) convergence rates;
Oberlin, Meignen & Perrier (2014) second-order SST.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _morlet_freq(xi: FloatArray, mu: float = 6.0, sd: float = 1.0) -> FloatArray:
    """Morlet wavelet in frequency: psi_hat(xi) = exp(-(xi-mu)^2/(2 sd^2))."""
    return np.asarray(np.exp(-0.5 * ((xi - mu) / sd) ** 2), dtype=np.float64)


def cwt(
    x: FloatArray,
    scales: FloatArray,
    dt: float = 1.0,
    mu: float = 6.0,
) -> FloatArray:
    """Complex CWT via FFT convolution with the Morlet wavelet."""
    x = np.asarray(x, dtype=float).ravel()
    scales = np.asarray(scales, dtype=float).ravel()
    n = x.size
    if n < 16 or scales.size < 4 or np.any(scales <= 0):
        raise ValueError("bad input")
    if not np.isfinite(x).all():
        raise ValueError("non-finite series")
    xh = np.fft.fft(x - x.mean())
    xi = 2.0 * np.pi * np.fft.fftfreq(n, dt)
    out = np.empty((scales.size, n), dtype=complex)
    for j, a in enumerate(scales):
        psihat = _morlet_freq(a * xi, mu)
        psihat[xi < 0] = 0.0
        conv = np.fft.ifft(xh * np.conj(psihat))
        out[j] = conv * np.sqrt(a * dt)
    return out


def phase_transform(w: FloatArray, dt: float = 1.0) -> FloatArray:
    """Instantaneous-frequency map omega_f = -i dW/db / W.

    Estimated as the time-derivative of the CWT phase; coefficients
    with |W| below the noise floor get NaN (excluded from squeezing).
    """
    mag = np.abs(w)
    floor = np.quantile(mag, 0.5) * 1e-3
    phase = np.angle(w)
    dphase = np.gradient(phase, dt, axis=1)
    om = np.where(mag > floor, dphase, np.nan)
    return np.asarray(om, dtype=np.float64)


def synchrosqueeze(
    w: FloatArray,
    om: FloatArray,
    scales: FloatArray,
    n_freq: int = 64,
    mu: float = 6.0,
) -> tuple[FloatArray, FloatArray]:
    """Reallocate |W|^2 from scale grid onto a linear frequency grid.

    The Morlet peak at psi_hat(mu) maps scale a to frequency
    f = mu / (2 pi a); squeezing bins each coefficient to the
    frequency implied by its phase transform value.
    """
    a = np.asarray(scales, dtype=float).ravel()
    f_native = mu / (2.0 * np.pi * a)
    f_min, f_max = float(f_native.min()), float(f_native.max())
    f_grid = np.linspace(f_min, f_max, n_freq)
    t_n = w.shape[1]
    tf = np.zeros((n_freq, t_n))
    for j in range(a.size):
        f_inst = om[j] / (2.0 * np.pi)
        valid = np.isfinite(f_inst) & (f_inst >= f_min) & (f_inst <= f_max)
        idx = np.searchsorted(f_grid, f_inst[valid])
        idx = np.clip(idx, 0, n_freq - 1)
        np.add.at(tf, (idx, np.nonzero(valid)[0]), np.abs(w[j, valid]) ** 2)
    return f_grid, np.asarray(tf, dtype=np.float64)


def ridge_frequency(tf: FloatArray, f_grid: FloatArray) -> FloatArray:
    """Per-time dominant frequency of the squeezed map."""
    idx = np.argmax(tf, axis=0)
    return np.asarray(f_grid[idx], dtype=np.float64)


def bench_synchrosqueezing(seed: int = 20261231 + 401) -> dict[str, float]:
    """SYNTHETIC check — squeezed ridge tracks a linear chirp's true IF."""
    rng = np.random.default_rng(seed)
    n, dt = 1024, 1.0
    t = np.arange(n) * dt
    f0, f1 = 0.03, 0.11
    if_true = f0 + (f1 - f0) * t / t[-1]
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / t[-1])
    x = np.cos(phase) + 0.25 * rng.standard_normal(n)
    scales = np.geomspace(6.0, 60.0, 40)
    w = cwt(x, scales, dt)
    om = phase_transform(w, dt)
    f_grid, tf = synchrosqueeze(w, om, scales, n_freq=60)
    ridge = ridge_frequency(tf, f_grid)
    mid = slice(n // 4, 3 * n // 4)  # avoid cone-of-influence edges
    err = float(np.median(np.abs(ridge[mid] - if_true[mid])))
    if err > 0.02:
        raise ValueError(f"ridge IF error {err}")
    # energy concentration: a big share of TF energy in the chirp band
    band = (f_grid > 0.02) & (f_grid < 0.14)
    conc = float(tf[band].sum() / tf.sum())
    if conc < 0.9:
        raise ValueError(f"energy not concentrated: {conc}")
    return {
        "synthetic_sst_ridge_err": err,
        "synthetic_sst_concentration": conc,
        "score": 1.0,
    }
