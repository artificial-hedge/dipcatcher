"""Experimental-design canon: 2^k full factorial,
Plackett-Burman (Paley construction), central
composite, Box-Behnken, Latin hypercube, and
D-optimal Fedorov exchange.

`bench_doe` checks structural properties:
orthogonality of factorial/PB columns, CCD rotatable
alpha, LHS margin uniformity, and D-optimal beating
a random design's log-det.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "full_factorial_2k",
    "plackett_burman",
    "central_composite",
    "box_behnken",
    "latin_hypercube",
    "d_optimal",
    "bench_doe",
]


def full_factorial_2k(k: int) -> FloatArray:
    """2^k full factorial in standard Yates order,
    levels ±1."""
    idx = np.arange(2**k)
    x = np.empty((2**k, k))
    for j in range(k):
        x[:, j] = 1 - 2 * ((idx >> (k - 1 - j)) & 1)
    return x


_PB_SEEDS = {
    12: "++-+---+++-",
    20: "++--++++-+-+--+-++-",
    24: "+++++-+-++-++--+-++--+--",
}


def plackett_burman(n: int) -> FloatArray:
    """Plackett-Burman design for N runs (N multiple
    of 4) via cyclic Paley rows; returns N x (N-1)."""
    if n not in _PB_SEEDS:
        raise ValueError(f"pb sizes supported: {sorted(_PB_SEEDS)}")
    row0 = np.array([1 if c == "+" else -1 for c in _PB_SEEDS[n]])
    mat = np.empty((n - 1, n - 1))
    for i in range(n - 1):
        mat[i] = np.roll(row0, i)
    return np.vstack([mat, -np.ones(n - 1)])


def central_composite(k: int, alpha: float | None = None, n_center: int = 4) -> FloatArray:
    """CCD: factorial + axial star points at ±alpha
    (rotatable default 2^{k/4}) + center runs."""
    if alpha is None:
        alpha = 2.0 ** (k / 4.0)
    f = full_factorial_2k(k)
    star = []
    for j in range(k):
        for s in (-alpha, alpha):
            row = np.zeros(k)
            row[j] = s
            star.append(row)
    return np.vstack([f, np.array(star), np.zeros((n_center, k))])


def box_behnken(k: int, n_center: int = 3) -> FloatArray:
    """Box-Behnken: all pairs of factors at ±1 with
    the rest at 0, plus center runs."""
    from itertools import combinations

    if k < 3:
        raise ValueError("box-behnken needs k>=3")
    rows = []
    for i, j in combinations(range(k), 2):
        for si in (-1, 1):
            for sj in (-1, 1):
                row = np.zeros(k)
                row[i], row[j] = si, sj
                rows.append(row)
    rows.extend([np.zeros(k)] * n_center)
    return np.array(rows)


def latin_hypercube(n: int, k: int, seed: int = 0) -> FloatArray:
    """Latin hypercube: stratified-uniform margins
    with independent column permutations."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, k))
    for j in range(k):
        perm = rng.permutation(n)
        u = rng.uniform(size=n)
        x[:, j] = (perm + u) / n
    return x


def d_optimal(
    candidates: FloatArray,
    n_runs: int,
    it: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Fedorov exchange for D-optimality: iterate
    adding the candidate that most raises det(XᵀX)
    and swapping."""
    cand = np.asarray(candidates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_cand, p = cand.shape
    if n_runs < p:
        raise ValueError("n_runs < p")
    sel = rng.choice(n_cand, n_runs, replace=False)
    for _ in range(it):
        improved = False
        for i in range(n_runs):
            # swap position i with best outsider
            outside = np.setdiff1d(np.arange(n_cand), sel)
            gains = []
            for j in outside:
                sub = np.delete(sel, i)
                m2 = cand[sub].T @ cand[sub] + np.outer(cand[j], cand[j])
                try:
                    sign, ld = np.linalg.slogdet(m2 + 1e-10 * np.eye(p))
                    gains.append((ld if sign > 0 else -np.inf, j))
                except np.linalg.LinAlgError:
                    continue
            best_ld, best_j = max(gains, key=lambda t: t[0])
            sel2 = np.delete(sel, i)
            m_cur = cand[sel2].T @ cand[sel2] + np.outer(cand[sel[i]], cand[sel[i]])
            sign_c, ld_cur = np.linalg.slogdet(m_cur + 1e-10 * np.eye(p))
            if best_ld > (ld_cur if sign_c > 0 else -np.inf) + 1e-9:
                sel = np.append(sel2, best_j)
                improved = True
                break
        if not improved:
            break
    return {"sel": sel, "design": cand[sel]}


def bench_doe(seed: int = 539) -> dict[str, float]:
    """SYNTHETIC: orthogonality of 2^k and PB designs,
    rotatable CCD alpha, LHS margin coverage, and
    D-optimal > random design."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    f = full_factorial_2k(3)
    xtx = f.T @ f
    off = xtx - np.diag(np.diag(xtx))
    out["synthetic_2k_max_offdiag"] = float(np.abs(off).max())
    if out["synthetic_2k_max_offdiag"] > 1e-10:
        raise ValueError("2k not orthogonal")
    pb = plackett_burman(12)
    xtx = pb.T @ pb
    off = np.abs(xtx - np.diag(np.diag(xtx))).max()
    out["synthetic_pb12_max_offdiag"] = float(off)
    if off > 1e-10:
        raise ValueError(f"pb12 not orthogonal: {off}")
    ccd = central_composite(3)
    a = float(np.abs(ccd[8]).max())
    out["synthetic_ccd_alpha"] = a
    if abs(a - 2 ** (3 / 4)) > 1e-9:
        raise ValueError(f"ccd alpha off: {a}")
    lhs = latin_hypercube(50, 3, seed)
    margin_min = float(lhs.min(axis=0).max())
    out["synthetic_lhs_margin_min"] = margin_min
    if margin_min > 0.15:
        raise ValueError(f"lhs margin off: {margin_min}")
    # D-optimal vs random on quadratic candidate space
    u = rng.uniform(-1, 1, (200, 2))
    cand = np.c_[np.ones(200), u, u[:, 0] * u[:, 1], u**2]
    dopt = d_optimal(cand, 12, seed=seed)
    m = np.asarray(dopt["design"])
    sign, ld_opt = np.linalg.slogdet(m.T @ m)
    rng_r = np.random.default_rng(seed + 1)
    ld_rand = -np.inf
    for _ in range(50):
        s = rng_r.choice(200, 12, replace=False)
        mr = cand[s]
        sgn, ld = np.linalg.slogdet(mr.T @ mr + 1e-12 * np.eye(6))
        if sgn > 0:
            ld_rand = max(ld_rand, ld)
    out["synthetic_dopt_logdet"] = float(ld_opt)
    out["synthetic_dopt_rand_logdet"] = float(ld_rand)
    if ld_opt <= ld_rand:
        raise ValueError(f"dopt not better: {ld_opt} vs {ld_rand}")
    return out
