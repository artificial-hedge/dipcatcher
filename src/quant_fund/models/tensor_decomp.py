"""Multilinear tensor factor analysis: CP/ALS, Tucker/HOOI, core consistency (SYNTHETIC).

Implements the classical polyadic (CP) decomposition via alternating
least squares (Harshman 1970) and the Tucker model via higher-order
orthogonal iteration (De Lathauwer, De Moor & Vandewalle 2000), with
rank selection by core consistency diagnostics (Bro & Kiers 2003) and
an EM step for missing entries (Tomasi & Bro 2005). These recover
planted low-rank factor structure in firm × characteristic × time
panels — the multilinear analogue of PCA.

References
----------
- Harshman (1970). Foundations of the PARAFAC procedure.
  *UCLA Working Papers in Phonetics* 16.
- De Lathauwer, De Moor & Vandewalle (2000). On the best rank-1 and
  rank-(R1,...,RN) approximation of higher-order tensors.
  *SIAM J. Matrix Anal. Appl.* 21(4).
- Bro & Kiers (2003). A new efficient method for determining the
  number of components in PARAFAC models. *J. Chemometrics* 17(5).
- Tomasi & Bro (2005). PARAFAC and missing values.
  *Chemometrics and Intelligent Laboratory Systems* 75.

Honesty
-------
Synth tensors carry planted factor structure; keys report
reconstruction error, Tucker-congruence factor recovery, missing-data
imputation error, and determinism — never claims about real asset
panels.

Composition notes
-----------------
- ``models/factor_nowcast.py``: two-way dynamic factors — this module
  is the three-way (and N-way) multilinear extension.
- ``metrics/marchenko_pastur.py``: spectral rank screening — core
  consistency is the CP-analogue rank diagnostic.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
TensorDict = dict[str, FloatArray | np.float64 | list[FloatArray]]


def _as_tensor(x: FloatArray, min_mode: int = 2) -> FloatArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < min_mode or min(x.shape) < 2:
        raise ValueError("tensor must have ndim>=min_mode with all modes >=2")
    if not np.all(np.isfinite(x)):
        raise ValueError("tensor contains non-finite values")
    return x


def unfold(x: FloatArray, mode: int) -> FloatArray:
    """Mode-`mode` matricization: rows are mode fibers, columns index the
    remaining modes (cyclical ordering)."""
    x = _as_tensor(x)
    n_modes = x.ndim
    if not 0 <= mode < n_modes:
        raise ValueError("mode out of range")
    order = [mode] + [m for m in range(n_modes) if m != mode]
    xt = np.transpose(x, order)
    return np.asarray(xt.reshape(x.shape[mode], -1), dtype=np.float64)


def fold(a: FloatArray, mode: int, shape: tuple[int, ...]) -> FloatArray:
    """Inverse of ``unfold``."""
    if not 0 <= mode < len(shape):
        raise ValueError("mode out of range")
    order = [mode] + [m for m in range(len(shape)) if m != mode]
    inv = np.argsort(order)
    tmp = np.asarray(a).reshape([shape[i] for i in order])
    return np.asarray(np.transpose(tmp, inv), dtype=np.float64)


def _khatri_rao(a: FloatArray, b: FloatArray) -> FloatArray:
    """Column-wise Kronecker product (a: n×R, b: m×R) → (nm × R)."""
    n, r = a.shape
    m, r2 = b.shape
    if r != r2:
        raise ValueError("Khatri-Rao factor rank mismatch")
    out = np.empty((n * m, r))
    for j in range(r):
        out[:, j] = np.kron(a[:, j], b[:, j])
    return out


def cp_als(
    x: FloatArray,
    rank: int,
    n_iter: int = 100,
    seed: int = 0,
    normalize: bool = True,
) -> TensorDict:
    """CP decomposition via alternating least squares.

    Returns factors (list of (mode_size, rank) arrays), weights
    (rank,) — column norms absorbed out — fitted tensor, and the final
    relative reconstruction error.
    """
    x = _as_tensor(x)
    if rank < 1 or rank > min(x.shape):
        raise ValueError("rank must be in [1, min(dim)]")
    rng = np.random.default_rng(seed)
    n_modes = x.ndim
    factors = [rng.standard_normal((s, rank)) + 0.5 for s in x.shape]
    norm_x = np.linalg.norm(x)
    prev_err = np.inf
    for _it in range(n_iter):
        for mode in range(n_modes):
            others = [factors[m] for m in range(n_modes) if m != mode]
            # MTTKRP: X_(mode) @ Khatri-Rao(others in reverse-cyclical)
            kr = others[0]
            for o in others[1:]:
                kr = _khatri_rao(kr, o)
            mttkrp = unfold(x, mode) @ kr
            v = np.ones((rank, rank))
            for o in others:
                v *= o.T @ o
            factors[mode] = mttkrp @ np.linalg.inv(v + 1e-8 * np.eye(rank))
            if normalize:
                for j in range(rank):
                    nrm = np.linalg.norm(factors[mode][:, j])
                    if nrm > 1e-12:
                        factors[mode][:, j] /= nrm
        fit = _cp_reconstruct(factors)
        err = float(np.linalg.norm(x - fit) / norm_x)
        if abs(prev_err - err) < 1e-8:
            prev_err = err
            break
        prev_err = err
    weights = np.ones(rank)
    fit = _cp_reconstruct(factors)
    rel_err = float(np.linalg.norm(x - fit) / norm_x)
    return {
        "factors": factors,
        "weights": np.asarray(weights),
        "fit": np.asarray(fit),
        "rel_err": np.float64(rel_err),
        "rank": np.float64(rank),
    }


def _cp_reconstruct(factors: list[FloatArray]) -> FloatArray:
    rank = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)
    out = np.zeros(shape)
    for j in range(rank):
        term = factors[0][:, j]
        for f in factors[1:]:
            term = np.multiply.outer(term, f[:, j])
        out += term
    return out


def tucker_hooi(
    x: FloatArray,
    ranks: tuple[int, ...],
    n_iter: int = 30,
) -> TensorDict:
    """Tucker decomposition via higher-order orthogonal iteration.

    Returns core tensor, factor matrices (orthonormal columns), fitted
    tensor, relative error.
    """
    x = _as_tensor(x, min_mode=3)
    n_modes = x.ndim
    if len(ranks) != n_modes or any(r < 1 or r > s for r, s in zip(ranks, x.shape, strict=True)):
        raise ValueError("ranks must match ndim and be <= mode sizes")
    # init factors from leading left-singular vectors of each unfolding
    factors = []
    for m in range(n_modes):
        u, _s, _vt = np.linalg.svd(unfold(x, m), full_matrices=False)
        factors.append(np.asarray(u[:, : ranks[m]]))
    norm_x = np.linalg.norm(x)
    for _it in range(n_iter):
        for m in range(n_modes):
            # project x onto all other factors then SVD the mode-m unfold
            y = x.copy()
            for k in range(n_modes):
                if k == m:
                    continue
                fm = factors[k]
                y_un = unfold(y, k)
                y = fold(
                    fm.T @ y_un,
                    k,
                    tuple(fm.shape[1] if i == k else y.shape[i] for i in range(n_modes)),
                )
            u, _s, _vt = np.linalg.svd(unfold(y, m), full_matrices=False)
            factors[m] = np.asarray(u[:, : ranks[m]])
    # core: project all modes
    core = x.copy()
    for m in range(n_modes):
        cu = unfold(core, m)
        core = fold(
            factors[m].T @ cu,
            m,
            tuple(ranks[m] if i == m else core.shape[i] for i in range(n_modes)),
        )
    fit = core.copy()
    for m in range(n_modes):
        fu = unfold(fit, m)
        fit = fold(
            factors[m] @ fu,
            m,
            tuple(x.shape[m] if i == m else fit.shape[i] for i in range(n_modes)),
        )
    rel_err = float(np.linalg.norm(x - fit) / norm_x)
    return {
        "core": np.asarray(core),
        "factors": factors,
        "fit": np.asarray(fit),
        "rel_err": np.float64(rel_err),
    }


def core_consistency(x: FloatArray, factors: list[FloatArray]) -> float:
    """CORCONDIA diagnostic (Bro & Kiers 2003): compares the Tucker core
    implied by CP factors to the ideal superdiagonal core. ~100 = clean
    CP; low/negative values flag the wrong rank.
    """
    x = _as_tensor(x)
    rank = factors[0].shape[1]
    # build core via pseudo-inverse projections
    core = x.copy()
    for m, f in enumerate(factors):
        pinv = np.linalg.pinv(f)
        cu = unfold(core, m)
        core = fold(
            pinv @ cu,
            m,
            tuple(rank if i == m else core.shape[i] for i in range(x.ndim)),
        )
    # ideal core: superdiagonal ones (norm of each column 1)
    ideal = np.zeros((rank,) * x.ndim)
    for j in range(rank):
        ideal[(j,) * x.ndim] = 1.0
    num = float(np.sum((core - ideal) ** 2))
    den = float(np.sum(ideal**2))
    return float(100.0 * (1.0 - num / den))


def cp_missing_em(
    x: FloatArray, mask: FloatArray, rank: int, n_iter: int = 60, seed: int = 0
) -> TensorDict:
    """CP-ALS with EM imputation for missing entries (Tomasi & Bro 2005).

    mask: 1 where observed, 0 where missing. Each iteration fills
    missing cells with the current model estimate before the ALS sweep.
    """
    x = _as_tensor(x)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != x.shape:
        raise ValueError("mask must match tensor shape")
    x_fill = np.where(mask > 0.5, x, 0.0)
    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((s, rank)) for s in x.shape]
    n_modes = x.ndim
    for _it in range(n_iter):
        fit = _cp_reconstruct(factors)
        x_fill = np.where(mask > 0.5, x, fit)
        for mode in range(n_modes):
            others = [factors[m] for m in range(n_modes) if m != mode]
            kr = others[0]
            for o in others[1:]:
                kr = _khatri_rao(kr, o)
            mttkrp = unfold(x_fill, mode) @ kr
            v = np.ones((rank, rank))
            for o in others:
                v *= o.T @ o
            factors[mode] = mttkrp @ np.linalg.inv(v + 1e-8 * np.eye(rank))
            for j in range(rank):
                nrm = np.linalg.norm(factors[mode][:, j])
                if nrm > 1e-12:
                    factors[mode][:, j] /= nrm
    fit = _cp_reconstruct(factors)
    miss = mask < 0.5
    miss_err = float(np.linalg.norm((x - fit)[miss]) / np.sqrt(miss.sum())) if miss.any() else 0.0
    obs_err = float(np.linalg.norm((x - fit)[mask > 0.5]) / np.linalg.norm(x[mask > 0.5]))
    return {
        "factors": factors,
        "fit": np.asarray(fit),
        "miss_rmse": np.float64(miss_err),
        "obs_rel_err": np.float64(obs_err),
    }


def tucker_congruence(a: FloatArray, b: FloatArray) -> float:
    """Tucker's congruence coefficient between two factor vectors:
    |<a,b>| / (||a|| ||b||) — the standard factor-recovery score."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return 0.0
    return float(abs(a @ b) / den)


def synth_tensor(
    shape: tuple[int, int, int] = (15, 10, 20),
    rank: int = 3,
    noise: float = 0.05,
    seed: int = 0,
) -> TensorDict:
    """Planted low-rank 3-way tensor: firm × characteristic × time."""
    rng = np.random.default_rng(seed)
    factors = [
        rng.standard_normal((s, rank)) * rng.choice([0.5, 1.0], size=(s, rank)) for s in shape
    ]
    # smooth the time factor to look like factor time series
    t_ar = np.cumsum(rng.standard_normal((shape[2], rank)), axis=0)
    factors[2] = t_ar / np.linalg.norm(t_ar, axis=0)[None, :]
    for f in factors[:2]:
        f /= np.linalg.norm(f, axis=0)[None, :]
    clean = _cp_reconstruct(factors)
    noisy = clean + noise * float(np.std(clean)) * rng.standard_normal(shape)
    return {
        "X": np.asarray(noisy, dtype=np.float64),
        "X_clean": np.asarray(clean),
        "true_factors": factors,
    }


def bench_tensor_decomp(seed: int = 20261231 + 166) -> dict[str, float]:
    """SYNTHETIC CP/Tucker recovery: relerr, congruence, EM imputation."""
    data = synth_tensor(rank=3, seed=seed)
    x = cast(FloatArray, data["X"])
    true_f = cast(list[FloatArray], data["true_factors"])
    cp = cp_als(x, rank=3, n_iter=150, seed=seed)
    tk = tucker_hooi(x, ranks=(3, 3, 3), n_iter=25)
    # factor recovery: best congruence per planted factor (mode 0)
    est_f = cast(list[FloatArray], cp["factors"])
    congs = []
    for j in range(3):
        best = max(tucker_congruence(true_f[0][:, j], est_f[0][:, r]) for r in range(3))
        congs.append(best)
    cc = core_consistency(x, est_f)
    cc_over = core_consistency(
        x, cast(list[FloatArray], cp_als(x, rank=6, n_iter=80, seed=seed + 1)["factors"])
    )
    # missing data EM: drop 15% cells
    rng = np.random.default_rng(seed + 7)
    mask = (rng.random(x.shape) > 0.15).astype(np.float64)
    em = cp_missing_em(x, mask, rank=3, n_iter=50, seed=seed + 2)
    d1 = float(cast(np.float64, cp["rel_err"]))
    d2 = float(cast(np.float64, cp_als(x, rank=3, n_iter=150, seed=seed)["rel_err"]))
    return {
        "synthetic_cp_relerr": d1,
        "synthetic_tucker_relerr": float(cast(np.float64, tk["rel_err"])),
        "synthetic_factor_congruence": float(np.mean(congs)),
        "synthetic_core_consistency": float(cc),
        "synthetic_core_consistency_overfit": float(cc_over),
        "synthetic_em_miss_rmse": float(cast(np.float64, em["miss_rmse"])),
        "synthetic_em_obs_relerr": float(cast(np.float64, em["obs_rel_err"])),
        "synthetic_determinism": float(d1 == d2),
    }
