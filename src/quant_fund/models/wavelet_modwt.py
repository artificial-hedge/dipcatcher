"""MODWT wavelet multiresolution analysis (Percival & Walden).

References
----------
- Percival, D.B. & Walden, A.T. (2000). *Wavelet Methods for
  Time Series Analysis*. Cambridge University Press, ch. 4-5.
- Percival, D.B. (1995). "On Estimation of the Wavelet
  Variance." *Biometrika* 82(3), 619-631.
- Whitcher, B., Guttorp, P. & Percival, D.B. (2000). "Wavelet
  Analysis of Covariance with Application to Atmospheric Time
  Series." *Journal of Geophysical Research* 105(D11),
  14941-14962.
- Gencay, R., Selcuk, F. & Whitcher, B. (2002). *An
  Introduction to Wavelets and Other Filtering Methods in
  Finance and Economics*. Academic Press.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
The maximal-overlap discrete wavelet transform keeps the
DWT pyramid but drops dyadic subsampling: at level j the
series is filtered by the ``2^(j-1)``-upsampled scaling
(high-pass ``h``) and low-pass (``g``) filters with circular
convolution, giving a length-n ``W_j`` wavelet-coefficient series and a single ``S_J``
smooth; the adjoint pyramid inverts exactly so ``x`` is
recovered to machine precision.
Wavelet variance per scale is the *unbiased* MODWT
estimator — the boundary coefficients affected by circular
wrapping (first ``L_j - 1 = (2^j - 1)(L - 1)`` taps) are
dropped before the mean-of-squares. Wavelet correlation at
scale j is the Pearson coefficient of the two interior
detail series. The bench sums two pure cycles (periods 16
and 64) plus noise: scale-4 detail (periods ~16-32) and
scale-6 (periods ~64-128) carry the variance mass while a
co-moving twin correlates ~1 at the shared scale.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_SQ2 = float(np.sqrt(2.0))
# LA(8) scaling (low-pass) filter, Percival & Walden Table.
_LA8_G = np.array(
    [
        -0.07576571478927333,
        -0.02963552764599851,
        0.49761866763201545,
        0.80373875180591614,
        0.29785779560527736,
        -0.09921954357684722,
        -0.012603967262037833,
        0.03222310060404270,
    ]
)


def _filters(name: str) -> tuple[FloatArray, FloatArray]:
    if name == "d4":
        g = np.array(
            [
                (1 + np.sqrt(3)) / (4 * _SQ2),
                (3 + np.sqrt(3)) / (4 * _SQ2),
                (3 - np.sqrt(3)) / (4 * _SQ2),
                (1 - np.sqrt(3)) / (4 * _SQ2),
            ]
        )
    elif name == "la8":
        g = _LA8_G.copy()
    else:
        raise ValueError(f"unknown wavelet {name!r}")
    # quadrature-mirror high-pass: h_l = (-1)^lag_i g_{L-1-lag_i};
    # MODWT normalizes both filter banks by 1/sqrt(2).
    l_idx = np.arange(g.size)
    h = ((-1.0) ** l_idx) * g[::-1]
    return g / _SQ2, h / _SQ2


def _circ_filter(x: FloatArray, filt: FloatArray, up: int) -> FloatArray:
    """Circular convolution with a 2^(up-1)-upsampled filter."""
    n = x.size
    out = np.zeros(n)
    for lag_i, w in enumerate(filt):
        lag = lag_i * (1 << (up - 1))
        idx = (np.arange(n) - lag) % n
        out += w * x[idx]
    return out


def modwt(
    x: FloatArray,
    level: int = 4,
    wavelet: str = "d4",
) -> dict[str, FloatArray]:
    """MODWT: returns {W1..W_J coefficients, S_J smooth, exact recon}."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 64 or not np.all(np.isfinite(xx)):
        raise ValueError("bad series")
    if float(np.std(xx)) < 1e-12:
        raise ValueError("degenerate")
    g, h = _filters(wavelet)
    j_max = int(np.floor(np.log2(xx.size))) - 1
    j_max = max(1, min(level, j_max))
    v = xx.copy()
    wave_coefs: dict[str, FloatArray] = {}
    for j in range(1, j_max + 1):
        w = _circ_filter(v, h, j)
        v = _circ_filter(v, g, j)
        wave_coefs[f"W{j}"] = w
    # inverse MODWT: adjoint (reversed-lag) pyramid — v_{j-1} gets
    # G_j^* V_j + H_j^* W_j at every level, exact for orthonormal
    # filters under circular convolution.
    recon = v.copy()
    for j in range(j_max, 0, -1):
        n = xx.size
        lo = np.zeros(n)
        hi = np.zeros(n)
        for lag_i, wgt in enumerate(g):
            lag = lag_i * (1 << (j - 1))
            idx = (np.arange(n) + lag) % n
            lo += wgt * recon[idx]
        for lag_i, wgt in enumerate(h):
            lag = lag_i * (1 << (j - 1))
            idx = (np.arange(n) + lag) % n
            hi += wgt * wave_coefs[f"W{j}"][idx]
        recon = lo + hi
    out: dict[str, FloatArray] = {"S_J": v}
    out.update(wave_coefs)
    out["recon_err"] = np.array([float(np.max(np.abs(recon - xx)))])
    out["level"] = np.array([float(j_max)])
    return out


def wavelet_variance(
    x: FloatArray,
    level: int = 4,
    wavelet: str = "d4",
) -> FloatArray:
    """Unbiased MODWT wavelet variance per scale j (boundary-trimmed)."""
    xx = np.asarray(x, dtype=np.float64)
    coeffs = modwt(xx, level=level, wavelet=wavelet)
    _, h = _filters(wavelet)
    j_max = int(coeffs["level"][0])
    out = np.zeros(j_max)
    n = xx.size
    for j in range(1, j_max + 1):
        d = coeffs[f"W{j}"]
        lj = (2**j - 1) * (h.size - 1)
        interior = d[lj:] if lj < n else d[n:]
        out[j - 1] = float(np.mean(interior**2)) if interior.size else np.nan
    return out


def wavelet_correlation(
    x: FloatArray,
    y: FloatArray,
    j: int,
    wavelet: str = "d4",
) -> float:
    """Scale-j wavelet correlation on boundary-trimmed details."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size != yy.size:
        raise ValueError("length mismatch")
    cx = modwt(xx, level=j, wavelet=wavelet)
    cy = modwt(yy, level=j, wavelet=wavelet)
    _, h = _filters(wavelet)
    lj = (2**j - 1) * (h.size - 1)
    dx = cx[f"W{j}"][lj:]
    dy = cy[f"W{j}"][lj:]
    sx, sy = float(np.std(dx)), float(np.std(dy))
    if sx < 1e-12 or sy < 1e-12:
        raise ValueError("degenerate detail")
    return float(np.corrcoef(dx, dy)[0, 1])


def synth_wavelet(
    seed: int = 20261231 + 330,
    n: int = 512,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC two-cycle signal + comoving twin + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    cyc = np.sin(2 * np.pi * t / 16.0) + 0.8 * np.sin(2 * np.pi * t / 64.0)
    x = cyc + 0.3 * rng.standard_normal(n)
    y = cyc + 0.3 * rng.standard_normal(n)
    wn = 0.3 * rng.standard_normal(n)
    return x, y, wn


def bench_wavelet_modwt(
    seed: int = 20261231 + 330,
) -> dict[str, float]:
    """Wave-57 self-check: MRA reconstruction + scale-localized
    variance + scale-matched correlation."""
    x, y, wn = synth_wavelet(seed=seed)
    coeffs = modwt(x, level=6, wavelet="d4")
    recon_ok = float(coeffs["recon_err"][0] < 1e-8)
    vx = wavelet_variance(x, level=6, wavelet="d4")
    vw = wavelet_variance(wn, level=6, wavelet="d4")
    rho = wavelet_correlation(x, y, j=4, wavelet="d4")
    peak = float(np.argmax(vx[:4]) + 1)
    # white noise wavelet variance decays like 2^-j across
    # scales; the two-cycle signal instead concentrates mass at
    # its cycle scales — the signal/noise contrast at the peak
    # is the discrimination check.
    snr_d4 = float(vx[3] / max(vw[3], 1e-12))
    ok = (
        recon_ok == 1.0
        and peak == 4.0
        and float(vx[3] > 3.0 * vx[0])
        and snr_d4 > 10.0
        and rho > 0.9
    )
    return {
        "recon_err": float(coeffs["recon_err"][0]),
        "var_peak_scale": peak,
        "var_d4": float(vx[3]),
        "snr_d4": snr_d4,
        "corr_d4": rho,
        "score": float(ok),
    }
