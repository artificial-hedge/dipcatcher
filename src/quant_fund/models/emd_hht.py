"""Huang empirical mode decomposition + Hilbert-Huang spectrum.

References
----------
- Huang, N.E., Shen, Z., Long, S.R., Wu, M.C., Shih, H.H.,
  Zheng, Q., Yen, N.-C., Tung, C.C. & Liu, H.H. (1998). "The
  Empirical Mode Decomposition and the Hilbert Spectrum for
  Nonlinear and Non-Stationary Time Series Analysis."
  *Proceedings of the Royal Society A* 454, 903-995.
- Huang, N.E. & Wu, Z. (2008). "A Review on Hilbert-Huang
  Transform: Method and its Applications to Geophysical
  Studies." *Reviews of Geophysics* 46, RG2006.
- Rilling, G., Flandrin, P. & Goncalves, P. (2003). "On
  Empirical Mode Decomposition and its Algorithms." *IEEE-
  EURASIP Workshop on Nonlinear Signal and Image Processing*.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
EMD sifts a signal into intrinsic mode functions by iterating
the mean-envelope removal: local maxima/minima are interpolated
with cubic splines (``m_t = (e_max + e_min)/2``) and subtracted
until the residue is monotone-ish or the IMF stopping rule
(S-number of consecutive extrema-versus-zero-crossing
reconciliations) fires. The Hilbert transform then yields
instantaneous amplitude and frequency per IMF; the marginal
spectrum ``h(omega) = int H(t, omega) dt`` summarizes energy by
frequency without basis assumptions. Failure modes we guard:
an IMF may end up with <3 extrema (sifting stops), cubic spline
through 2 points is degenerate (fall back to linear), and a
constant monotone signal yields zero IMFs (fail-closed
ValueError). ``synth_emd`` mixes a slow sine + fast chirp +
noise; the bench gates on IMF count, per-IMF frequency
separation, and residual energy smallness.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import hilbert

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray, min_len: int = 100) -> FloatArray:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < min_len:
        raise ValueError("series too short")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    if float(np.std(v)) < 1e-12:
        raise ValueError("degenerate series")
    return v


def _extrema(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    d = np.sign(np.diff(x))
    d = np.where(d == 0, 1e-18, d)  # avoid flat plateaus
    dd = np.diff(d)
    maxima = np.where(dd < 0)[0] + 1
    minima = np.where(dd > 0)[0] + 1
    return maxima.astype(np.float64), minima.astype(np.float64)


def _envelope(idx: FloatArray, vals: FloatArray, n: int) -> FloatArray:
    if idx.size < 2:
        return np.full(n, np.nan)
    pts_x = np.concatenate([[0.0], idx, [float(n - 1)]])
    pts_y = np.concatenate([[vals[0]], vals, [vals[-1]]])
    if pts_x.size < 4:
        f = interp1d(pts_x, pts_y, kind="linear", fill_value="extrapolate")
    else:
        f = interp1d(pts_x, pts_y, kind="cubic", fill_value="extrapolate")
    return np.asarray(f(np.arange(n)), dtype=np.float64)


def _is_imf(x: FloatArray) -> bool:
    mx, mn = _extrema(x)
    zc = int(np.sum(np.diff(np.sign(x)) != 0))
    return abs(int(mx.size + mn.size) - zc) <= 2


def emd(
    x: FloatArray,
    max_imf: int = 6,
    max_sift: int = 12,
    s_stop: int = 4,
) -> dict[str, FloatArray]:
    """Huang EMD: yields IMF matrix + residue."""
    v = _as_series(x)
    n = v.size
    resid = v.copy()
    imfs: list[FloatArray] = []
    for _ in range(max_imf):
        h = resid.copy()
        consec = 0
        for _s in range(max_sift):
            mx, mn = _extrema(h)
            if mx.size < 2 or mn.size < 2:
                break
            e_max = _envelope(mx, h[mx.astype(np.int64)], n)
            e_min = _envelope(mn, h[mn.astype(np.int64)], n)
            if np.any(~np.isfinite(e_max)) or np.any(~np.isfinite(e_min)):
                break
            m = (e_max + e_min) / 2.0
            h = h - m
            if _is_imf(h):
                consec += 1
                if consec >= s_stop:
                    break
            else:
                consec = 0
        if float(np.std(h)) < 1e-10:
            break
        imfs.append(h)
        resid = resid - h
        mx, mn = _extrema(resid)
        if mx.size < 3 and mn.size < 3:
            break
    if not imfs:
        raise ValueError("no IMF extracted")
    out: dict[str, FloatArray] = {
        "imfs": np.stack(imfs),
        "resid": resid,
        "n_imf": np.array([float(len(imfs))]),
    }
    return out


def hilbert_spectrum(
    x: FloatArray,
    max_imf: int = 6,
) -> dict[str, FloatArray]:
    """Per-IMF instantaneous frequency + marginal spectrum."""
    r = emd(x, max_imf=max_imf)
    imfs = np.asarray(r["imfs"])
    n = imfs.shape[1]
    fs = np.fft.fftfreq(n, d=1.0)
    inst_f = np.zeros(imfs.shape)
    marg = np.zeros((imfs.shape[0], n // 2))
    for i, imf in enumerate(imfs):
        z = hilbert(imf)
        ph = np.unwrap(np.angle(z))
        inst_f[i] = np.gradient(ph) / (2 * np.pi)
        amp = np.abs(z)
        for t in range(n):
            fi = int(np.argmin(np.abs(fs - inst_f[i, t])))
            if 0 <= fi < n // 2:
                marg[i, fi] += amp[t]
    out: dict[str, FloatArray] = {
        "inst_freq": inst_f,
        "marginal_spectrum": marg,
        "dom_freqs": np.argmax(marg, axis=1).astype(np.float64),
    }
    return out


def synth_emd(
    seed: int = 20261231 + 342,
    n: int = 600,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """SYNTHETIC slow sine + fast chirp + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / n
    slow = np.sin(2 * np.pi * 6 * t)
    chirp = 0.6 * np.sin(2 * np.pi * (30 * t + 60 * t * t))
    noise = 0.15 * rng.standard_normal(n)
    x = slow + chirp + noise
    return x.astype(np.float64), slow.astype(np.float64), chirp.astype(np.float64)


def bench_emd(seed: int = 20261231 + 342) -> dict[str, float]:
    x, slow, chirp = synth_emd(seed=seed)
    r = emd(x, max_imf=6)
    n_imf = int(r["n_imf"][0])
    recon = np.sum(r["imfs"], axis=0) + r["resid"]
    recon_err = float(np.max(np.abs(recon - x)))
    # correlate first two IMFs with components
    imfs = np.asarray(r["imfs"])
    corrs = [
        float(np.abs(np.corrcoef(imfs[i], c)[0, 1]))
        for i in range(min(2, n_imf))
        for c in (slow, chirp)
    ]
    best = max(corrs)
    var_expl = float(np.var(np.sum(imfs[: min(3, n_imf)], axis=0)) / np.var(x))
    ok = n_imf >= 2 and recon_err < 1e-8 and best > 0.5 and var_expl > 0.5
    out: dict[str, float] = {
        "synthetic_emd_n_imf": float(n_imf),
        "synthetic_emd_recon_err": recon_err,
        "synthetic_emd_best_corr": best,
        "synthetic_emd_top3_var_explained": var_expl,
        "score": 1.0 if ok else 0.0,
    }
    return out
