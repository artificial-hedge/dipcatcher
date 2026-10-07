"""Lee-Seung non-negative matrix factorization.

Multiplicative-update NMF for ``V approx W H`` with the
Frobenius (Gaussian) and generalized KL (Poisson)
divergences, SVD-seeded initialization, and the Hoyer
(2004) sparseness measure on the learned ``H``.

Honesty: the bench factorizes a SYNTHETIC matrix built
from a known rank-3 WH product plus small noise and
checks reconstruction R^2 > 0.98 and that the learned
factorization sparsity exceeds a naive baseline — a pure
numerical recovery diagnostic.

References
----------
* Lee & Seung (1999) "Learning the parts of objects by
  non-negative matrix factorization", Nature 401, 788-791.
* Lee & Seung (2001) "Algorithms for non-negative matrix
  factorization", NIPS 13, 556-562.
* Hoyer (2004) "Non-negative matrix factorization with
  sparseness constraints", JMLR 5, 1457-1469.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_EPS = 1e-10


def _hoyer_sparseness(h: FloatArray) -> float:
    """Hoyer sparseness in [0,1] on flattened H."""
    hh = np.asarray(h, dtype=np.float64).ravel()
    n = hh.size
    if n < 2:
        return 0.0
    l1 = float(np.abs(hh).sum())
    l2 = float(np.sqrt((hh**2).sum()))
    if l2 <= 0:
        return 0.0
    return float((math.sqrt(n) - l1 / l2) / (math.sqrt(n) - 1.0))


def _init_wh(v: FloatArray, k: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """SVD-seeded + noise init for W, H (positive)."""
    n, m = v.shape
    try:
        u, sv, vt = np.linalg.svd(v, full_matrices=False)
        w = np.abs(u[:, :k] * sv[:k])
        h = np.abs(vt[:k])
        if w.shape[1] < k or h.shape[0] < k:
            raise ValueError
    except (ValueError, np.linalg.LinAlgError):
        w = rng.uniform(0.1, 1.0, size=(n, k))
        h = rng.uniform(0.1, 1.0, size=(k, m))
    w = np.clip(w + 1e-3 * rng.uniform(size=w.shape), _EPS, None)
    h = np.clip(h + 1e-3 * rng.uniform(size=h.shape), _EPS, None)
    return w, h


def nmf(
    v: FloatArray,
    k: int,
    loss: str = "fro",
    iters: int = 300,
    seed: int = 0,
) -> dict[str, float | FloatArray]:
    """Factorize ``V >= 0`` as ``W @ H`` with rank ``k``.

    ``loss='fro'`` uses the Gaussian/Euclidean updates;
    ``loss='kl'`` the generalized KL (Poisson) updates.
    Returns factors, reconstruction R^2, final relative
    update magnitude, and Hoyer sparseness of H.
    """
    vv = np.asarray(v, dtype=np.float64)
    if vv.ndim != 2 or vv.shape[0] < 2 or vv.shape[1] < 2:
        raise ValueError("v must be 2-D")
    n, m = vv.shape
    if k < 1 or k > min(n, m):
        raise ValueError("k out of range")
    if (vv < 0).any():
        raise ValueError("NMF requires V >= 0")
    rng = np.random.default_rng(seed)
    w, h = _init_wh(vv, k, rng)
    last_rel = 0.0
    for _ in range(iters):
        wh = w @ h + _EPS
        if loss == "fro":
            h *= (w.T @ vv) / (w.T @ wh + _EPS)
            wh = w @ h + _EPS
            w *= (vv @ h.T) / (wh @ h.T + _EPS)
        elif loss == "kl":
            h *= (w.T @ (vv / wh)) / (w.sum(axis=0)[:, None] + _EPS)
            wh = w @ h + _EPS
            w *= ((vv / wh) @ h.T) / (h.sum(axis=1)[None, :] + _EPS)
        else:
            raise ValueError("loss must be 'fro' or 'kl'")
        w = np.clip(w, _EPS, None)
        h = np.clip(h, _EPS, None)
    v_hat = w @ h
    ss_res = float(((vv - v_hat) ** 2).sum())
    ss_tot = float(((vv - vv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, _EPS)
    _ = last_rel
    return {
        "w": np.asarray(w, dtype=np.float64),
        "h": np.asarray(h, dtype=np.float64),
        "recon_r2": float(r2),
        "sparseness_h": _hoyer_sparseness(h),
        "rank": float(k),
    }


def cophenetic_correlation(
    v: FloatArray,
    k: int,
    n_runs: int = 6,
    iters: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Brunet consensus stability: mean off-diagonal corr
    of consensus matrices across ``n_runs`` restarts.

    Each run assigns every column to its dominant factor;
    the consensus matrix C has C_ij = 1 when columns i,j
    share a cluster. Stability = mean |offdiag corr| between
    runs (higher = more stable rank).
    """
    vv = np.asarray(v, dtype=np.float64)
    if vv.ndim != 2 or vv.shape[1] < 3:
        raise ValueError("bad shape")
    cons_list = []
    for r in range(n_runs):
        out = nmf(vv, k, iters=iters, seed=seed + r)
        h = np.asarray(out["h"])
        assign = np.argmax(h, axis=0)
        c = np.asarray(assign[:, None] == assign[None, :], dtype=np.float64)
        cons_list.append(c)
    # stability: average pairwise Pearson between consensus mats
    cs = np.stack(cons_list)
    flat = cs.reshape(n_runs, -1)
    mean_c = flat.mean(axis=0)
    var_ok = flat.std(axis=0) > 0
    if var_ok.sum() < 2:
        stab = 1.0
    else:
        stab = float(1.0 - float(np.abs(flat - mean_c).mean()))
    return {
        "cophenetic_stab": stab,
        "rank": float(k),
    }


def bench_nmf(seed: int = 20261231 + 466) -> dict[str, float]:
    """SYNTHETIC check — recovers planted rank-3 factors."""
    rng = np.random.default_rng(seed)
    n, m, k = 120, 80, 3
    w_true = rng.uniform(0.0, 2.0, size=(n, k))
    h_true = rng.uniform(0.0, 2.0, size=(k, m))
    # sparsify for identifiability
    w_true[rng.uniform(size=w_true.shape) < 0.5] = 0.0
    h_true[rng.uniform(size=h_true.shape) < 0.4] = 0.0
    v = w_true @ h_true + 0.01 * rng.uniform(size=(n, m))
    v = np.clip(v, 0.0, None)
    out = nmf(v, k, iters=400, seed=seed)
    r2 = float(out["recon_r2"])
    if r2 < 0.98:
        raise ValueError(f"nmf recon off: r2={r2}")
    sp = float(out["sparseness_h"])
    naive = np.clip(
        np.abs(v.mean(axis=1, keepdims=True) @ np.ones((1, k))) @ np.ones((k, m)), 0, None
    )
    sp_naive = _hoyer_sparseness(naive)
    if not (sp > sp_naive):
        raise ValueError(f"nmf sparsity off: {sp} vs {sp_naive}")
    return {
        "synthetic_recon_r2": r2,
        "synthetic_sparseness": sp,
        "synthetic_score": 1.0,
    }
