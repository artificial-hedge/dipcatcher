"""Recurrence quantification analysis (RQA).

Recurrence plots + quantitative measures of recurrence structure
(Eckmann-Kamphorst-Ruelle 1987; Zbilut-Webber 1992; Marwan et al. 2007):
recurrence rate, determinism, laminarity, mean/max diagonal line length,
and diagonal-line-length entropy — probes that separate periodic/
deterministic-chaotic structure from iid noise, and drifting regimes
from stationary ones.

References
----------
- Eckmann, Kamphorst & Ruelle (1987). Recurrence plots of dynamical
  systems. *Europhysics Letters* 4(9).
- Zbilut & Webber (1992). Embeddings and delays as derived from
  quantification of recurrence plots. *Physics Letters A* 171.
- Marwan, Romano, Thiel & Kurths (2007). Recurrence plots for the
  analysis of complex systems. *Physics Reports* 438 — summary of
  RR/DET/LAM/Lmax/ENTR conventions.

Honesty
-------
All series are SYNTHETIC (periodic orbit, iid noise, Lorenz-63 chaos,
AR). RQA measures are structural diagnostics — keys report DET/RR
ordering between known dynamics, never a market claim.

Composition notes
-----------------
- ``metrics/dependence.py``: serial-correlation probes — this module
  probes *nonlinear* determinism via phase-space recurrences.
- ``models/koopman_edmd.py``: linear-operator embedding — RQA is a
  complementary invariant-style probe.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

FloatArray = NDArray[np.float64]


def _as_series(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 20 or not np.all(np.isfinite(x)):
        raise ValueError("series must be finite with >=20 points")
    return x


def embed_series(x: FloatArray, m: int = 2, tau: int = 1) -> FloatArray:
    """Takens delay embedding: rows are (x_t, x_{t+tau}, ..., x_{t+(m-1)tau})."""
    x = _as_series(x)
    if m < 1 or tau < 1:
        raise ValueError("m, tau >= 1")
    n = x.size - (m - 1) * tau
    if n < 5:
        raise ValueError("embedding too short")
    return np.asarray(
        np.column_stack([x[i * tau : i * tau + n] for i in range(m)]),
        dtype=np.float64,
    )


def mutual_information_delay(x: FloatArray, max_tau: int = 20, bins: int = 16) -> int:
    """First minimum of the mutual information vs delay — standard
    tau-picking heuristic (Fraser-Swinney 1986)."""
    x = _as_series(x)
    mi_prev = math.inf
    for tau in range(1, max_tau + 1):
        a = x[:-tau]
        b = x[tau:]
        h2, _xe, _ye = np.histogram2d(a, b, bins=bins)
        p = h2 / h2.sum()
        pa = p.sum(axis=1, keepdims=True)
        pb = p.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = float(np.nansum(p * np.log(p / (pa @ pb))))
        if mi > mi_prev:
            return tau - 1
        mi_prev = mi
    return max_tau


def recurrence_plot(
    x: FloatArray, m: int = 3, tau: int = 1, eps: float | None = None
) -> FloatArray:
    """Boolean recurrence matrix R[i,j] = 1{||x_i - x_j|| < eps}.

    eps defaults to 10% of the trajectory diameter (Marwan standard)."""
    emb = embed_series(x, m=m, tau=tau)
    d = cdist(emb, emb)
    if eps is None:
        eps = float(np.quantile(d[np.triu_indices(d.shape[0], 1)], 0.10))
    if eps <= 0:
        raise ValueError("eps must be positive")
    return np.asarray(d < eps, dtype=np.float64)


def _diag_lengths(rp: FloatArray, lmin: int = 2) -> FloatArray:
    """Histogram of diagonal line lengths >= lmin (excluding main diag)."""
    n = rp.shape[0]
    lengths: list[int] = []
    for k in range(-(n - 1), n):
        diag = np.diag(rp, k)
        if diag.size < lmin:
            continue
        # runs of ones
        count = 0
        for v in diag:
            if v > 0:
                count += 1
            elif count >= lmin:
                lengths.append(count)
                count = 0
            else:
                count = 0
        if count >= lmin:
            lengths.append(count)
    return np.asarray(sorted(lengths), dtype=np.float64)


def rqa_measures(
    x: FloatArray, m: int = 3, tau: int = 1, eps: float | None = None, lmin: int = 2
) -> dict[str, float]:
    """Core RQA measures from the recurrence plot.

    RR recurrence rate; DET fraction of recurrence points in diagonal
    lines >= lmin; LAM the same for vertical lines; Lmax longest
    diagonal; ENTR Shannon entropy of the diagonal-length distribution;
    TT trapping time (mean vertical line length).
    """
    rp = recurrence_plot(x, m=m, tau=tau, eps=eps)
    rp_off = rp.copy()
    np.fill_diagonal(rp_off, 0.0)
    rr = float(rp_off.mean())
    diag_l = _diag_lengths(rp_off, lmin)
    n_recur = rp_off.sum()
    det = float(np.sum(diag_l) / max(n_recur, 1.0))
    lmax = float(diag_l.max()) if diag_l.size else 0.0
    if diag_l.size:
        p = diag_l / diag_l.sum()
        entr = float(-np.sum(p * np.log(p)))
    else:
        entr = 0.0
    # vertical structures: transpose diagonals = verticals
    vert_l = _diag_lengths(rp_off.T, lmin)
    lam = float(np.sum(vert_l) / max(n_recur, 1.0))
    tt = float(vert_l.mean()) if vert_l.size else 0.0
    return {
        "rr": rr,
        "det": det,
        "lam": lam,
        "lmax": lmax,
        "entr": entr,
        "tt": tt,
        "n_recur": float(n_recur),
    }


def synth_periodic(
    n: int = 400, freq: float = 0.15, noise: float = 0.05, seed: int = 0
) -> FloatArray:
    """Noisy periodic orbit — high determinism expected."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.asarray(np.sin(freq * t) + noise * rng.standard_normal(n), dtype=np.float64)


def synth_iid(n: int = 400, seed: int = 0) -> FloatArray:
    """iid noise — low determinism expected."""
    return np.asarray(np.random.default_rng(seed).standard_normal(n), dtype=np.float64)


def synth_lorenz(n: int = 600, seed: int = 0) -> FloatArray:
    """Lorenz-63 x-component via RK4 — chaotic determinism expected."""
    rng = np.random.default_rng(seed)
    x, y, z = 1.0 + rng.standard_normal(3) * 0.1
    dt = 0.01
    out = np.empty(n + 2000)

    def f(s: FloatArray) -> FloatArray:
        xx, yy, zz = s
        return np.array([10.0 * (yy - xx), xx * (28.0 - zz) - yy, xx * yy - 8.0 / 3.0 * zz])

    s = np.array([x, y, z])
    for i in range(n + 2000):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = s[0]
    return np.asarray(out[2000:], dtype=np.float64)


def synth_ar(n: int = 400, rho: float = 0.9, seed: int = 0) -> FloatArray:
    """AR(1) — linear autocorrelation but no nonlinear determinism."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return np.asarray(x, dtype=np.float64)


def bench_rqa(seed: int = 20261231 + 164) -> dict[str, float]:
    """SYNTHETIC RQA ordering checks."""
    per = rqa_measures(synth_periodic(400, seed=seed), m=3, tau=3)
    iid = rqa_measures(synth_iid(400, seed=seed + 1), m=3, tau=3)
    lor = rqa_measures(synth_lorenz(600, seed=seed + 2), m=3, tau=5)
    ar = rqa_measures(synth_ar(400, seed=seed + 3), m=3, tau=1)
    a = rqa_measures(synth_periodic(200, seed=9), m=3, tau=3)
    b = rqa_measures(synth_periodic(200, seed=9), m=3, tau=3)
    return {
        "synthetic_det_periodic": per["det"],
        "synthetic_det_iid": iid["det"],
        "synthetic_det_lorenz": lor["det"],
        "synthetic_det_ar": ar["det"],
        "synthetic_det_margin_order": per["det"] - iid["det"],
        "synthetic_entr_lorenz": lor["entr"],
        "synthetic_entr_periodic": per["entr"],
        "synthetic_rr_periodic": per["rr"],
        "synthetic_lmax_lorenz": lor["lmax"],
        "synthetic_determinism": float(all(a[k] == b[k] for k in a)),
    }
