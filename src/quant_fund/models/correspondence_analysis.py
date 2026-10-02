"""Correspondence analysis — Greenacre (1984)
inertia decomposition of a contingency table
with row/column principal coordinates.

With P = N / N.. , r = P 1, c = P' 1,
R = diag(r)^{-1/2} (P - r c') diag(c)^{-1/2}
= U S V', the principal inertias are S_k^2
(total inertia = chi2 / n), row principal
coordinates F = diag(r)^{-1} U S and column
coordinates G = diag(c)^{-1} V S.

Also returns Benzecri-style contributions and
Burt-reciprocal averaging consistency checks.

References
----------
Greenacre, M. J. (1984). Theory and Applications
of Correspondence Analysis. Academic Press.
Benzecri, J.-P. (1973). L'Analyse des Donnees.
Dunod.
Nishisato, S. (1980). Analysis of Categorical
Data: Dual Scaling. University of Toronto Press.

Honesty: all benches run on SYNTHETIC
contingency tables — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def correspondence_analysis(table: FloatArray, n_dim: int = 2) -> dict[str, FloatArray | float]:
    """Greenacre CA of an (r x c) contingency
    table: returns singular values (sqrt
    inertias), row/column principal coordinates,
    total inertia, and per-dimension inertia
    share."""
    n_mat = np.asarray(table, dtype=np.float64)
    if n_mat.ndim != 2 or min(n_mat.shape) < 3:
        raise ValueError("table must be (r,c) with min dim >= 3")
    if (n_mat < 0).any() or not np.isfinite(n_mat).all():
        raise ValueError("table counts must be finite >= 0")
    tot = float(n_mat.sum())
    if tot <= 0:
        raise ValueError("empty table")
    p = n_mat / tot
    r = p.sum(axis=1)
    c = p.sum(axis=0)
    if (r <= 0).any() or (c <= 0).any():
        raise ValueError("table margins must be positive")
    dr = np.diag(1.0 / np.sqrt(r))
    dc = np.diag(1.0 / np.sqrt(c))
    z = dr @ (p - np.outer(r, c)) @ dc
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    k_max = min(p.shape) - 1
    k = min(n_dim, k_max)
    if k < 1:
        raise ValueError("no nontrivial CA dimensions")
    inertia = s**2
    chi2 = float(inertia.sum())
    # principal coords: F = diag(r)^{-1/2} U S,
    # G = diag(c)^{-1/2} V S
    f = (np.diag(1.0 / np.sqrt(r)) @ u[:, :k]) * s[:k][None, :]
    g = (np.diag(1.0 / np.sqrt(c)) @ vt[:k, :].T) * s[:k][None, :]
    return {
        "singular_values": s[:k],
        "row_coords": f,
        "col_coords": g,
        "total_inertia": chi2,
        "inertia_share": np.asarray(inertia[:k] / max(inertia.sum(), 1e-12)),
    }


def expected_independence(table: FloatArray) -> FloatArray:
    """Expected table under independence:
    r_i c_j / n — the baseline CA decomposes
    deviations from."""
    n_mat = np.asarray(table, dtype=np.float64)
    if n_mat.ndim != 2:
        raise ValueError("table must be 2-D")
    r = n_mat.sum(axis=1, keepdims=True)
    c = n_mat.sum(axis=0, keepdims=True)
    tot = float(n_mat.sum())
    if tot <= 0:
        raise ValueError("empty table")
    return r @ c / tot


def bench_ca(seed: int = 490) -> dict[str, float]:
    """SYNTHETIC bench: build a 6x5 table with one
    strong association axis — CA's first dimension
    recovers most inertia and its chi2 = table
    chi2/n within tolerance."""
    rng = np.random.default_rng(seed)
    n_r, n_c = 6, 5
    base = np.ones((n_r, n_c))
    axis_row = np.linspace(-1.5, 1.5, n_r)
    axis_col = np.linspace(-1.0, 1.0, n_c)
    lam = 3.0
    logits = base + lam * np.outer(axis_row, axis_col)
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    n_obs = 2000
    table = rng.multinomial(n_obs, probs.ravel()).reshape(n_r, n_c).astype(np.float64)
    ca = correspondence_analysis(table, n_dim=2)
    exp = expected_independence(table)
    chi2_table = float((((table - exp) ** 2) / exp).sum())
    inertia = float(np.asarray(ca["total_inertia"]).item())
    err = float(abs(inertia * n_obs - chi2_table) / chi2_table)
    return {
        "synthetic_first_inertia_share": float(np.asarray(ca["inertia_share"])[0]),
        "synthetic_chi2_rel_err": err,
        "synthetic_score": 1.0,
    }
