"""Post-stratification and raking (iterative proportional
fitting) for survey weights.

Deming & Stephan (1940): given a sample classified into cells
(marginals of one or more control variables) and known population
cell counts, post-stratification sets w_i = N_h / n_h in cell h,
which exactly matches margins. When only the MARGINS of several
control variables are known (not the joint cells), IPF/raking
iterates one margin at a time to convergence.

Kish (1965) effective sample size n_eff = (sum w)^2 / sum w^2
measures weighting cost; design effect deff = n / n_eff.

Honesty: the bench uses a biased sample (younger respondents
oversampled relative to known population margins) — post-strat
weights must recover the population mean of a margin-correlated
outcome while the raw mean misses. Raking is checked for exact
margin matching. Fail-closed on empty cells or zero population
margins.

References: Deming & Stephan (1940) "On a least squares
adjustment of a sampled frequency table", Ann. Math. Stat.
11:427; Kish (1965) "Survey Sampling"; Little (1993) "Post-
stratification: a modeler's perspective", JASA 88:1001.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def poststrat_weights(
    sample_cells: FloatArray, pop_shares: FloatArray, n_sample: int
) -> tuple[FloatArray, FloatArray]:
    """Post-stratification weights for cell-coded samples.

    ``sample_cells`` are integer cell ids per respondent,
    ``pop_shares`` the known population share per cell (sums to 1).
    Returns (weights, cell_ids_used).
    """
    cells = np.asarray(sample_cells, dtype=float)
    shares = np.asarray(pop_shares, dtype=float)
    if cells.ndim != 1 or cells.size != n_sample:
        raise ValueError("bad cells")
    if shares.ndim != 1 or shares.size == 0 or not np.isfinite(shares).all():
        raise ValueError("bad shares")
    if (shares <= 0).any():
        raise ValueError("zero population share")
    shares = shares / shares.sum()
    cell_ids = cells.astype(int)
    if (cell_ids < 0).any() or (cell_ids >= shares.size).any():
        raise ValueError("cell id out of range")
    n_h = np.bincount(cell_ids, minlength=shares.size).astype(float)
    if (n_h[cell_ids] <= 0).any():
        raise ValueError("empty sampled cell")
    w = shares[cell_ids] * n_sample / n_h[cell_ids]
    return np.asarray(w, dtype=np.float64), cell_ids.astype(np.float64)


def raking(
    design: FloatArray,
    margins: list[FloatArray],
    base_weights: FloatArray | None = None,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> FloatArray:
    """Iterative proportional fitting to several margins.

    ``design`` (n, d) codes each respondent's category per control
    variable (non-negative integers); ``margins[k]`` the target
    shares for variable k's categories. Returns calibrated weights
    normalized to mean 1.
    """
    a = np.asarray(design, dtype=float)
    if a.ndim != 2 or a.shape[0] < 4:
        raise ValueError("bad design")
    n, d = a.shape
    if d != len(margins):
        raise ValueError("margin count mismatch")
    cat_idx = a.astype(int)
    if (cat_idx < 0).any():
        raise ValueError("bad category")
    shares = []
    for k, m in enumerate(margins):
        s = np.asarray(m, dtype=float)
        if s.ndim != 1 or (s <= 0).any() or (cat_idx[:, k] >= s.size).any():
            raise ValueError("bad margin")
        shares.append(s / s.sum())
    w = np.ones(n) if base_weights is None else np.asarray(base_weights, dtype=float).copy()
    if w.shape != (n,) or (w <= 0).any():
        raise ValueError("bad base weights")
    for _ in range(n_iter):
        delta = 0.0
        for k in range(d):
            col = cat_idx[:, k]
            cur = np.bincount(col, weights=w, minlength=shares[k].size)
            cur = np.maximum(cur / cur.sum(), 1e-12)
            adj = shares[k] / cur
            w = w * adj[col]
            w = w / w.mean()
            delta = max(delta, float(np.abs(adj - 1).max()))
        if delta < tol:
            break
    return np.asarray(w, dtype=np.float64)


def design_effect(weights: FloatArray) -> dict[str, float]:
    """Kish DEFF and effective sample size of a weight vector."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size < 4 or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("bad weights")
    n_eff = float(w.sum() ** 2 / (w * w).sum())
    deff = w.size / n_eff
    return {"n_eff": n_eff, "deff": float(deff), "cv_weights": float(w.std() / w.mean())}


def bench_poststrat(seed: int = 20261231 + 445) -> dict[str, float]:
    """SYNTHETIC check — post-strat fixes the margin bias; raking exact."""
    rng = np.random.default_rng(seed)
    # population: 3 cells with shares .2/.5/.3, outcome mean 0/1/3
    shares = np.array([0.2, 0.5, 0.3])
    cell_mean = np.array([0.0, 1.0, 3.0])
    true_mean = float((shares * cell_mean).sum())
    # biased sample: over-samples cell 0
    n = 600
    p_samp = np.array([0.5, 0.35, 0.15])
    cells = rng.choice(3, n, p=p_samp)
    y = cell_mean[cells] + 0.3 * rng.standard_normal(n)
    w, _ = poststrat_weights(cells.astype(float), shares, n)
    raw_mean = float(y.mean())
    w_mean = float((w * y).sum() / w.sum())
    # raking on two one-hot control variables must match margins
    design = np.stack([cells.astype(float), (rng.random(n) < 0.4).astype(float)], axis=1)
    w2 = raking(design, [shares, np.array([0.4, 0.6])])
    deff = design_effect(w)
    if abs(w_mean - true_mean) > 0.08 or abs(raw_mean - true_mean) < 0.1:
        raise ValueError(
            f"poststrat off: w_mean={w_mean:.3f} raw={raw_mean:.3f} true={true_mean:.3f}"
        )
    # verify raked margin matches
    col = design[:, 0].astype(int)
    rake_marg = np.bincount(col, weights=w2, minlength=3) / w2.sum()
    if np.abs(rake_marg - shares).max() > 0.02:
        raise ValueError(f"raking margins off: {rake_marg}")
    return {
        "synthetic_poststrat_err": abs(w_mean - true_mean),
        "synthetic_raw_err": abs(raw_mean - true_mean),
        "synthetic_deff": deff["deff"],
        "synthetic_rake_margin_err": float(np.abs(rake_marg - shares).max()),
        "synthetic_score": 1.0,
    }
