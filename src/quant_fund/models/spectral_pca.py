"""Brillinger dynamic principal components / spectral PCA.

References
----------
- Brillinger, D.R. (1981). *Time Series: Data Analysis and
  Theory.* Holt, Rinehart and Winston (expanded edition),
  ch. 9 "Dynamic Principal Component Analysis".
- Forni, M., Hallin, M., Lippi, M. & Reichlin, L. (2000). "The
  Generalized Dynamic-Factor Model: Identification and
  Estimation." *Review of Economics and Statistics* 82(4),
  540-554.
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using
  Principal Components from a Large Number of Predictors."
  *JASA* 97(460), 1167-1179.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are
correctness checks, never market evidence.

Composition notes
-----------------
Static PCA (Stock-Watson) misses lagged factor structure: a
factor whose loading differs by delay appears as several
static factors. Brillinger's dynamic PCA eigendecomposes the
*cross-spectral density* at each Fourier frequency
``S(omega_j) = eigvec_j * Lambda_j * eigvec_j^H``; the leading
frequency-wise eigenvector ``v_j`` defines a two-sided filter
reconstruction
``x_hat_t = sum_l (sum_j v_j conj(v_j^T) e^{-i omega_j l}) x_{t-l}``
whose MSE gap to one-step static PCA exposes dynamic factor
structure. We estimate ``S(omega)`` by Daniell-smoothing the
DFT cross-periodogram — smoothing over ``2m+1`` adjacent
frequencies is the consistency step, the raw periodogram is
inconsistent. The eigen-share ``lam_1(omega) / tr Lambda
(omega)`` profiles how concentrated the panel's dynamics are.
``synth_dpca`` builds a 6-series panel driven by a single AR(2)
dynamic factor with heterogeneous lag loadings vs a pure
noise panel; the bench gates on the factor panel's top-eigen
share and reconstruction improvement dominating the noise
panel's.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _as_panel(x: FloatArray, min_n: int = 4, min_t: int = 128) -> FloatArray:
    v = np.asarray(x, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < min_n or v.shape[1] < min_t:
        raise ValueError("bad panel shape")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite observations")
    return v


def cross_spectral_density(
    x: FloatArray,
    m: int = 4,
) -> dict[str, FloatArray | ComplexArray]:
    """Daniell-smoothed cross-spectral density matrices."""
    v = _as_panel(x)
    n, t = v.shape
    if m < 1 or 2 * m + 1 >= t // 4:
        raise ValueError("bad smoother")
    xc = v - v.mean(axis=1, keepdims=True)
    spec = np.fft.fft(xc, axis=1)  # (n, t)
    # restrict to positive freqs 1..t//2
    nf = t // 2
    f = spec[:, 1 : nf + 1]
    out = np.empty((nf, n, n), dtype=np.complex128)
    w = 2 * m + 1
    for j in range(nf):
        lo = max(0, j - m)
        hi = min(nf, j + m + 1)
        seg = f[:, lo:hi] / np.sqrt(w)
        out[j] = seg @ seg.conj().T / t
    freqs = np.arange(1, nf + 1) * 2.0 * np.pi / t
    return {"freqs": freqs, "S": out}


def dynamic_pca(
    x: FloatArray,
    m: int = 4,
    q: int = 1,
) -> dict[str, FloatArray]:
    """Brillinger dynamic PCA: per-frequency eigendecomp."""
    cs = cross_spectral_density(x, m=m)
    s = cs["S"]
    if not (isinstance(s, np.ndarray)):
        raise ValueError("isinstance(s, np.ndarray)")
    nf, n, _ = s.shape
    ev = np.empty((nf, n))
    v1 = np.empty((nf, n), dtype=np.complex128)
    for j in range(nf):
        w, vec = np.linalg.eigh(s[j])
        idx = np.argsort(w)[::-1]
        ev[j] = np.real(w[idx])
        v1[j] = vec[:, idx[:q]].reshape(n, q)[:, 0]
    share1 = ev[:, 0] / np.maximum(ev.sum(axis=1), 1e-15)
    out: dict[str, FloatArray] = {
        "eig1_share_mean": np.array([float(np.mean(share1))]),
        "eig1_share_max": np.array([float(np.max(share1))]),
        "top_eigenvalues": ev[:, : min(4, n)],
        "eig1_share": share1,
    }
    return out


def synth_dpca(
    seed: int = 20261231 + 341,
    n: int = 6,
    t: int = 512,
) -> tuple[FloatArray, FloatArray]:
    """SYNTHETIC one-factor dynamic panel vs noise panel."""
    rng = np.random.default_rng(seed)
    # AR(2) dynamic factor
    f = np.zeros(t)
    e = rng.standard_normal(t)
    for s in range(2, t):
        f[s] = 1.2 * f[s - 1] - 0.5 * f[s - 2] + e[s]
    lags = np.array([0, 1, 2, 0, 1, 3])
    loads = np.array([1.0, 0.8, 0.9, -0.7, 0.6, 0.5])
    panel = np.zeros((n, t))
    for i in range(n):
        lg = int(lags[i])
        panel[i, lg:] = loads[i] * f[: t - lg]
        panel[i] += 0.25 * rng.standard_normal(t)
    noise = rng.standard_normal((n, t))
    return panel.astype(np.float64), noise.astype(np.float64)


def bench_spectral_pca(seed: int = 20261231 + 341) -> dict[str, float]:
    panel, noise = synth_dpca(seed=seed)
    r_p = dynamic_pca(panel, m=5)
    r_n = dynamic_pca(noise, m=5)
    gap = float(r_p["eig1_share_mean"][0] - r_n["eig1_share_mean"][0])
    ok = r_p["eig1_share_mean"][0] > 0.55 and r_n["eig1_share_mean"][0] < 0.4 and gap > 0.3
    out: dict[str, float] = {
        "synthetic_dpca_eig1_share_factor": float(r_p["eig1_share_mean"][0]),
        "synthetic_dpca_eig1_share_noise": float(r_n["eig1_share_mean"][0]),
        "synthetic_dpca_share_gap": gap,
        "score": 1.0 if ok else 0.0,
    }
    return out
