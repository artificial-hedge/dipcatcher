"""Subspace spectral estimation: MUSIC, root-MUSIC, and
LS/TLS-ESPRIT for sinusoids in noise.

Decomposes the autocorrelation matrix of a windowed signal
into signal/noise subspaces (Schmidt 1986 MUSIC; Roy &
Kailath 1989 ESPRIT), then recovers complex exponentials
via pseudospectrum peaks or generalized eigenvalues of
shifted signal subspaces.

References
----------
- Schmidt (1986) 'Multiple emitter location and signal
  parameter estimation' IEEE TAP 34(3).
- Roy & Kailath (1989) 'ESPRIT — estimation of signal
  parameters via rotational invariance techniques'
  IEEE TASSP 37(7).
- Barabell (1983) root-MUSIC for uniform linear arrays.

Honesty
-------
SYNTHETIC self-check: seeded sinusoid mixtures in AR(1)/
white noise; reports frequency recovery error vs the
known generating frequencies. No market claims.

Composition
-----------
Pure numpy (eigendecomposition of small Toeplitz
autocorrelation matrices). Input is a 1-D real/complex
series; outputs are frequency estimates and the MUSIC
pseudospectrum.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _check_signal(x: FloatArray, m: int, p: int) -> FloatArray:
    xa = np.asarray(x, dtype=np.float64).ravel()
    if xa.size < 2 * m or not np.isfinite(xa).all():
        raise ValueError("signal too short or non-finite")
    if not 1 <= p < m - 1:
        raise ValueError("p must satisfy 1 <= p < m-1")
    return xa


def _autocorr_matrix(x: FloatArray, m: int) -> ComplexArray:
    """Unbiased-forward Toeplitz autocorrelation R (m x m)."""
    n = x.size
    r = np.empty(m, dtype=np.complex128)
    for k in range(m):
        r[k] = np.dot(x[k:], x[: n - k].conj()) / (n - k)
    idx = np.abs(np.subtract.outer(np.arange(m), np.arange(m)))
    # R[i,j] = r[i-j] with conjugate for j>i
    sign = np.sign(np.subtract.outer(np.arange(m), np.arange(m)))
    rr = np.where(sign >= 0, np.conj(r[idx]), r[idx])
    return np.asarray(rr, dtype=np.complex128)


def _subspaces(x: FloatArray, m: int, p: int) -> tuple[ComplexArray, ComplexArray]:
    r = _autocorr_matrix(x, m)
    w, v = np.linalg.eigh(r)
    noise = v[:, : m - p]
    signal = v[:, m - p :]
    return signal, noise


def music_spectrum(
    x: FloatArray, m: int = 40, p: int = 2, n_grid: int = 2048
) -> tuple[FloatArray, FloatArray]:
    """MUSIC pseudospectrum on [0, pi] (normalized freq 0..0.5)."""
    xa = _check_signal(x, m, p)
    _, noise = _subspaces(xa, m, p)
    w = np.linspace(0, np.pi, n_grid)
    e = np.exp(-1j * np.outer(np.arange(m), w))  # m x G steering
    proj = noise.conj().T @ e  # (m-p) x G
    denom = (np.abs(proj) ** 2).sum(axis=0)
    spec = 1.0 / np.maximum(denom, 1e-30)
    return w / (2 * np.pi), spec


def music_frequencies(x: FloatArray, m: int = 40, p: int = 2) -> FloatArray:
    """root-MUSIC frequency estimates (cycles/sample).

    For real input, returns the positive-angle half of the
    near-unit-circle roots (one per sinusoid).
    """
    xa = _check_signal(x, m, p)
    _, noise = _subspaces(xa, m, p)
    nn = noise @ noise.conj().T
    # polynomial coefficients: sums along diagonals of NN
    coef = np.zeros(2 * m - 1, dtype=np.complex128)
    for k in range(-(m - 1), m):
        coef[k + m - 1] = np.trace(nn, offset=-k)
    roots = np.roots(coef[::-1])
    ang = np.angle(roots)
    good = np.abs(np.abs(roots) - 1.0) < 0.08
    f = np.sort(ang[(ang > 0) & good] / (2 * np.pi))
    # inside/outside root pairs share one angle -> dedupe
    if f.size:
        keep = np.concatenate([[True], np.diff(f) > 1e-3])
        f = f[keep]
    if f.size < max(1, p // 2):
        # fallback: dense grid peaks
        w, spec = music_spectrum(xa, m=m, p=p)
        idx = np.argsort(spec)[::-1]
        cand = np.sort(w[idx[: max(1, p // 2)]])
        return cand.astype(np.float64)
    return f.astype(np.float64)


def esprit(x: FloatArray, m: int = 40, p: int = 2, tls: bool = True) -> FloatArray:
    """LS/TLS-ESPRIT frequency estimates (cycles/sample)."""
    xa = _check_signal(x, m, p)
    signal, _ = _subspaces(xa, m, p)
    s1 = signal[:-1, :]
    s2 = signal[1:, :]
    if tls:
        c = np.hstack([s1, s2])
        _, _, vh = np.linalg.svd(c, full_matrices=False)
        v = vh.conj().T
        v11, v12 = v[:p, :p], v[:p, p:]
        v21, v22 = v[p:, :p], v[p:, p:]
        phi = -v12 @ np.linalg.inv(v22)
        del v11, v21
    else:
        phi = np.linalg.pinv(s1) @ s2
    eigs = np.linalg.eigvals(phi)
    ang = np.angle(eigs)
    # real input: each sinusoid emits a conjugate pair (+-f);
    # the positive-angle half indexes distinct sinusoids
    f = np.sort(ang[ang > 0] / (2 * np.pi))
    if f.size < max(1, p // 2):
        raise ValueError("ESPRIT returned too few frequencies")
    return f.astype(np.float64)


def bench_music_esprit(seed: int = 500) -> dict[str, float]:
    """SYNTHETIC: recover 3 sinusoids from noisy mixture."""
    rng = np.random.default_rng(seed)
    n = 512
    t = np.arange(n)
    f_true = np.array([0.07, 0.19, 0.33])
    x = (
        np.sin(2 * np.pi * f_true[0] * t + 0.3)
        + 0.8 * np.sin(2 * np.pi * f_true[1] * t + 1.7)
        + 0.6 * np.sin(2 * np.pi * f_true[2] * t + 4.0)
        + 0.5 * rng.standard_normal(n)
    )
    m = 48
    fm = np.sort(music_frequencies(x, m=m, p=6))
    fe = np.sort(esprit(x, m=m, p=6, tls=True))
    # keep estimates within 0.02 of a true mode
    fm3 = np.sort(fm[np.abs(fm[:, None] - f_true[None, :]).min(axis=1) < 0.02])
    fe3 = np.sort(fe[np.abs(fe[:, None] - f_true[None, :]).min(axis=1) < 0.02])
    if fm3.size != 3 or fe3.size != 3:
        raise ValueError("subspace methods failed to resolve all modes")
    err_m = float(np.max(np.abs(np.sort(fm3) - f_true)))
    err_e = float(np.max(np.abs(np.sort(fe3) - f_true)))
    return {
        "synthetic_music_max_f_err": err_m,
        "synthetic_esprit_max_f_err": err_e,
        "synthetic_music_f0": float(fm3[0]),
        "synthetic_esprit_f0": float(fe3[0]),
        "synthetic_snr_db": float(10 * np.log10(np.var(x - 0.5 * rng.standard_normal(n)) / 0.25)),
    }
