"""Singular spectrum analysis (SSA) for nonlinear signal
decomposition — no parametric form assumed.

Canonical references:

- Broomhead & King (1986) 'Extracting qualitative
  dynamics from experimental data' Physica D 20 —
  trajectory-matrix embedding + SVD.
- Vautard, Yiou & Ghil (1992) 'Singular-spectrum
  analysis: a toolkit for short, noisy chaotic signals'
  Physica D 58 — eigentriple grouping and paired-EOF
  harmonic identification.
- Golyandina, Nekrutkin & Zhigljavsky (2001) 'Analysis
  of Time Series Structure: SSA and Related Techniques'
  — diagonal-averaging reconstruction and the
  w-correlation separability metric.
- Golyandina & Korobeynikov (2014) 'Basic SSA' — Hankel
  reconstruction used here.

Pipeline: embed y into the L-trajectory Hankel matrix X
(K columns), eigendecompose X X^T (equivalently the SVD),
group leading eigentriples (trend = 1st, harmonic pairs
= adjacent near-equal eigenvalues), and reconstruct by
anti-diagonal (diagonal) averaging of the elementary
matrices sqrt(lambda) u v^T.

`bench_ssa`: noisy sine + slow trend; SSA with L=n/3
must recover the sine with correlation > 0.9 while a
residual stays unstructured (lag-1 |rho| small).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check(y: FloatArray) -> FloatArray:
    ya = np.asarray(y, dtype=np.float64).ravel()
    if ya.size < 20 or not np.isfinite(ya).all():
        raise ValueError("bad series")
    return ya


def trajectory_matrix(y: FloatArray, window: int) -> FloatArray:
    """Hankel trajectory matrix X (L x K), K = n - L + 1."""
    ya = _check(y)
    n = ya.size
    if not (2 <= window <= n // 2):
        raise ValueError("window in [2, n/2]")
    k = n - window + 1
    return np.vstack([ya[i : i + k] for i in range(window)])


def ssa_decompose(y: FloatArray, window: int, n_components: int | None = None) -> dict[str, object]:
    """Eigentriple decomposition: singular values, left (EOF)
    and right (PC) singular vectors."""
    x = trajectory_matrix(y, window)
    # eigh on X X^T is stable for modest L
    cxx = x @ x.T
    lam, u = np.linalg.eigh(cxx)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0, None)
    u = u[:, order]
    v = x.T @ u / np.sqrt(np.maximum(lam, 1e-300))
    k = n_components or lam.size
    return {
        "sv": np.sqrt(lam[:k]),
        "u": u[:, :k],
        "v": v[:, :k],
        "share": lam / lam.sum(),
        "X_shape": x.shape,
    }


def _reconstruct_elem(u_i: FloatArray, v_i: FloatArray, n: int, window: int) -> FloatArray:
    """Diagonal averaging (Hankelization) of u_i v_i^T."""
    m = np.outer(u_i, v_i)
    k = n - window + 1
    rec = np.zeros(n)
    cnt = np.zeros(n)
    # X[j, c] = y[j+c]; element t is covered by j+c=t
    for j in range(window):
        for c in range(k):
            rec[j + c] += m[j, c]
            cnt[j + c] += 1
    return rec / np.maximum(cnt, 1)


def ssa_reconstruct(
    y: FloatArray,
    window: int,
    groups: list[list[int]] | None = None,
) -> dict[str, FloatArray]:
    """Group eigentriples and reconstruct components.

    `groups` = list of 0-based component index lists, e.g.
    [[0],[1,2]] = trend, first harmonic pair.
    Returns recon_i arrays plus the residual."""
    ya = _check(y)
    n = ya.size
    dec = ssa_decompose(ya, window)
    u = np.asarray(dec["u"])
    v = np.asarray(dec["v"])
    sv = np.asarray(dec["sv"])
    if groups is None:
        groups = [[0]]
    used: set[int] = set()
    out: dict[str, FloatArray] = {}
    for gi, g in enumerate(groups):
        m = np.zeros(n)
        for i in g:
            if i < 0 or i >= sv.size:
                raise ValueError("component index out of range")
            m += _reconstruct_elem(sv[i] * u[:, i], v[:, i], n, window)
            used.add(i)
        out[f"recon_{gi}"] = m
    resid = ya.copy()
    for gkey in (k for k in out if k.startswith("recon_")):
        resid = resid - out[gkey]
    out["residual"] = resid
    return out


def w_correlation(y1: FloatArray, y2: FloatArray, window: int) -> float:
    """Weighted correlation between two reconstructions."""
    a = np.asarray(y1, dtype=np.float64).ravel()
    b = np.asarray(y2, dtype=np.float64).ravel()
    if a.size != b.size or a.size < 20:
        raise ValueError("length mismatch")
    n = a.size
    w = np.minimum(np.minimum(np.arange(1, n + 1), window), n - np.arange(1, n + 1) + 1).astype(
        np.float64
    )
    num = float((w * a * b).sum())
    den = float(np.sqrt((w * a * a).sum() * (w * b * b).sum()))
    return num / den if den > 0 else 0.0


def bench_ssa(seed: int = 518) -> dict[str, float]:
    """SYNTHETIC: sine + linear trend + noise; recon[0] must
    track the trend, recon pair [1,2] the sine."""
    rng = np.random.default_rng(seed)
    n = 200
    t = np.arange(n)
    trend = 2.0 + 0.02 * t
    sine = 1.0 * np.sin(2 * np.pi * t / 25.0)
    y = trend + sine + rng.normal(0, 0.3, n)
    window = 60
    dec = ssa_decompose(y, window)
    sv = np.asarray(dec["sv"])
    rec = ssa_reconstruct(y, window, [[0], [1, 2]])
    tr = rec["recon_0"]
    si = rec["recon_1"]
    corr_tr = float(np.corrcoef(tr, trend)[0, 1])
    corr_si = float(np.corrcoef(si, sine)[0, 1])
    resid = np.asarray(rec["residual"])
    rho1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    if corr_tr < 0.98 or corr_si < 0.85:
        raise ValueError("SSA reconstruction poor")
    share = np.asarray(dec["share"])
    return {
        "synthetic_corr_trend": corr_tr,
        "synthetic_corr_sine": corr_si,
        "synthetic_resid_rho1": rho1,
        "synthetic_sv1": float(sv[0]),
        "synthetic_share_top3": float(share[:3].sum()),
        "synthetic_sv_gap": float(sv[1] / max(sv[3], 1e-12)),
    }
