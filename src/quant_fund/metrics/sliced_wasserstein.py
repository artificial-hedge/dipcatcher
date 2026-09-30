"""Sliced Wasserstein distances and OT two-sample goodness-of-fit testing.

Sliced Wasserstein (SW) distances for predictive distributions: project both
measures onto directions of the unit sphere, evaluate the exact 1-D
Wasserstein distance on each projection via
``quant_fund.metrics.wasserstein.wasserstein_1d`` (optimal monotone quantile
coupling in 1-D, Villani 2003, Ch. 2), and aggregate over directions:

    SW_p^p(P, Q) = int_{S^{d-1}} W_p^p(theta#P, theta#Q) dtheta,

introduced by Rabin, Flamary, Cuturi & Villani (2010, "Water and Waves",
AISTATS 2010, JMLR W&CP 9:579-586), with the deterministic equally-spaced
direction scheme and the fixed-point barycenter of Bonneel, Rabin, Peyre &
Pfister (2015, "Sliced and Radon Wasserstein barycenters of planar
distributions", J. Math. Imaging Vis. 51(1):22-45,
doi:10.1007/s10851-014-0506-3).

1. ``sliced_wasserstein_distance``: SW_p with random (seeded Gaussian,
   uniform on the sphere) or deterministic (equally-spaced in 2-D; Sobol'
   quasi-uniform + inverse-normal map in dim >= 3) directions. Known
   properties — closed-form evaluations and SW/W bounds are studied by
   Gao & Xiu (2021) and Nadjahi, Durmus, Chizat, Kolouri, Shahrampour &
   Simsekli (2021, "Statistical and Topological Properties of Sliced
   Probability Divergences", AISTATS 2021, arXiv:2003.07817): projections
   are 1-Lipschitz, so SW_p <= W_p with constant 1; in the reverse direction
   W_2 <= C_d * SW_2 with C_d growing like sqrt(d), and the bound is sharp
   for isotropic Gaussian scale shifts where W_2 = sqrt(d) * SW_2 exactly;
   for d = 1, SW_p = W_p exactly (every direction is +/- the identity, and
   W_p is invariant under the antipodal sign).
2. ``max_sliced_wasserstein_distance``: the max-over-slices variant (the
   sup-statistic is discussed already in Rabin et al. 2010 and used for
   projection-based detection in Kolouri, Zou & Rohde 2018, "Detection of
   near-anomalies using sliced Wasserstein projections", Proc. SPIE; see
   also Kolouri et al. 2019, "Generalized Sliced Wasserstein Distances",
   NeurIPS 32, arXiv:1905.12958). On a common direction set,
   SW_p <= MSW_p <= W_p by construction (max >= mean^(1/p); each projected
   W_p is dominated by the ambient W_p).
3. ``sliced_wasserstein_test``: permutation two-sample test with SW or MSW
   statistic and the exactly calibrated p-value (1 + #{perm >= obs}) /
   (n_perm + 1) (Phipson & Smyth 2010, "Permutation P-values Should Never
   Be Zero", Brief. Bioinform. 11(6):611-616, doi:10.1093/bib/bbq042).
   Directions are fixed across permutations, and the permutation loop uses a
   column-batched replica of the ``wasserstein_1d`` quantile-grid coupling,
   so the observed statistic agrees with the public distance function. The
   natural comparator is the energy-distance permutation test (Rizzo &
   Székely 2016, "Energy distance", WIREs Comput. Stat. 8(1):27-38,
   doi:10.1002/wics.1375; cf. ``quant_fund.metrics.energy_score`` and
   ``wasserstein.energy_distance_from_ot``): the d-dimensional energy
   distance is likewise an average over sphere directions of a 1-D
   discrepancy, so SW tests are the optimal-transport analogues of the
   energy test, and both are consistent against fixed alternatives (Ramdas,
   Garcia Trillos & Cuturi 2017, Ann. Inst. Statist. Math. 69(2):679-694,
   arXiv:1509.02237).
4. ``sliced_wasserstein_barycenter``: the fixed-point iteration of Bonneel
   et al. (2015) for the SW barycenter of equal-cardinality empirical
   clouds — per direction, rank-match sorted projections of the current
   barycenter against each input cloud and take the weighted mean of the
   matched points. Documented convergence check: relative sup-norm
   displacement of the barycenter per sweep (``converged`` /
   ``final_delta``). The iteration is a POCS-style heuristic — for a finite
   direction set in dim >= 2 the per-direction rank constraints are
   generally mutually inconsistent (the averaged quantile slices need not
   be the slices of any single n-atom cloud), so it can enter a limit cycle
   instead of a fixed point; ``converged=False`` after ``max_iter`` sweeps
   must then be treated as a failure of evidence, never silently accepted.
   Where the constraints are consistent — dim = 1, identical inputs, single
   cloud, degenerate (point-mass) inputs — it converges exactly, typically
   in one or two sweeps.

Honesty: outputs are distributional distances, calibrated test p-values and
barycenter point clouds — proper diagnostic quantities for SYNTHETIC or
research data. No Sharpe/Sortino/P&L content; no live-trading claims
(AGENTS.md honesty contract).

Conventions: numpy/scipy core, functional API with a frozen-dataclass result
for the barycenter, fail-closed edges (ValueError on empty/non-finite
clouds, dimension mismatch, n_projections <= 0, p < 1, n_perm < 1, unknown
projection mode or statistic kind, bad weights; nothing is silently
dropped), seeded determinism (same seed -> bit-identical output).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri
from scipy.stats import qmc

from quant_fund.metrics.wasserstein import wasserstein_1d

Array = NDArray[np.float64]

__all__ = [
    "SlicedBarycenter",
    "deterministic_projections",
    "max_sliced_wasserstein_distance",
    "random_projections",
    "sliced_wasserstein_barycenter",
    "sliced_wasserstein_distance",
    "sliced_wasserstein_test",
]

_PROJECTION_MODES = ("random", "deterministic")
_STATISTIC_KINDS = ("sw", "msw")


# ------------------------------------------------------------- validation


def _check_point_cloud(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D (n_samples, dim) array")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_dim_match(a: Array, b: Array) -> None:
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: A has dim {a.shape[1]}, B has dim {b.shape[1]}")


def _check_p(p: float) -> float:
    q = float(p)
    if not np.isfinite(q) or q < 1.0:
        raise ValueError("p must be finite and >= 1")
    return q


def _check_n_projections(n_projections: int) -> int:
    lp = int(n_projections)
    if lp < 1:
        raise ValueError("n_projections must be >= 1")
    return lp


def _check_seed(seed: int) -> int:
    s = int(seed)
    if s < 0:
        raise ValueError("seed must be non-negative")
    return s


def _check_projection_mode(projection: str) -> str:
    mode = str(projection)
    if mode not in _PROJECTION_MODES:
        raise ValueError(f"projection must be one of {_PROJECTION_MODES}, got {projection!r}")
    return mode


def _check_statistic_kind(statistic: str) -> str:
    kind = str(statistic)
    if kind not in _STATISTIC_KINDS:
        raise ValueError(f"statistic must be one of {_STATISTIC_KINDS}, got {statistic!r}")
    return kind


def _check_n_perm(n_perm: int) -> int:
    nper = int(n_perm)
    if nper < 1:
        raise ValueError("n_perm must be >= 1")
    return nper


def _check_weights(name: str, weights: Array | None, n: int) -> Array:
    if weights is None:
        return np.full(n, 1.0 / float(n), dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"{name} must have one entry per input ({n}), got {w.size}")
    if not bool(np.all(np.isfinite(w))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if bool(np.any(w < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return np.asarray(w / total, dtype=float)


# ------------------------------------------------------------- directions


def _normalize_directions(v: Array, name: str) -> Array:
    norms = np.linalg.norm(v, axis=1)
    if bool(np.any(norms <= 0.0)) or not bool(np.all(np.isfinite(norms))):
        raise ValueError(f"{name} must all be nonzero and finite")
    return np.asarray(v / norms[:, None], dtype=float)


def random_projections(dim: int, n_proj: int, seed: int) -> Array:
    """Seeded Gaussian directions, i.i.d. uniform on the sphere S^{dim-1}.

    Standard-normal rows normalized to unit length are exactly uniform on
    the sphere (rotation invariance of the Gaussian); the seed pins the draw
    so repeated calls are bit-identical (Rabin et al. 2010 use this Monte
    Carlo scheme for the sphere integral in SW_p).
    """
    d = int(dim)
    lp = _check_n_projections(n_proj)
    s = _check_seed(seed)
    if d < 1:
        raise ValueError("dim must be >= 1")
    rng = np.random.default_rng(s)
    return _normalize_directions(
        np.asarray(rng.standard_normal((lp, d)), dtype=float), "random directions"
    )


def deterministic_projections(dim: int, n_proj: int) -> Array:
    """Deterministic equally-spaced / quasi-uniform directions.

    dim == 1: the +1 direction repeated (antipodal symmetry — theta = -1
    gives the same W_p, so one direction spans S^0 fully).
    dim == 2: equally-spaced angles on [0, pi), the scheme of Bonneel et al.
    (2015) for planar slices (theta and -theta give identical W_p).
    dim >= 3: Sobol' quasi-random points (Owen-scrambled with pinned
    internal seed 0, so the scheme stays deterministic) mapped through the
    inverse normal CDF and normalized — a QMC analogue of the equally-spaced
    scheme, quasi-uniform on the sphere. Draws are padded to a power of two
    for the Sobol' balance property.
    """
    d = int(dim)
    lp = _check_n_projections(n_proj)
    if d < 1:
        raise ValueError("dim must be >= 1")
    if d == 1:
        return np.ones((lp, 1), dtype=float)
    if d == 2:
        ang = np.pi * np.arange(lp, dtype=float) / float(lp)
        return np.asarray(np.column_stack((np.cos(ang), np.sin(ang))), dtype=float)
    m = max(1, int(np.ceil(np.log2(float(lp)))))
    u = np.asarray(qmc.Sobol(d=d, scramble=True, seed=0).random(2**m)[:lp], dtype=float)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return _normalize_directions(np.asarray(ndtri(u), dtype=float), "Sobol directions")


def _projection_matrix(dim: int, n_proj: int, projection: str, seed: int) -> Array:
    mode = _check_projection_mode(projection)
    if mode == "random":
        return random_projections(dim, n_proj, seed)
    return deterministic_projections(dim, n_proj)


# ------------------------------------------------------------- aggregation


def _wp_columns(pa: Array, pb: Array, p: float) -> Array:
    """Per-column W_p between two projected clouds, batched over columns.

    Exact column-wise replica of ``wasserstein_1d``: the same
    mid-probability quantile-grid monotone coupling (Villani 2003, Ch. 2),
    evaluated for all projection directions at once via ``np.quantile`` on
    axis 0. Used by the permutation loop so that the observed statistic and
    the null statistics are computed identically.
    """
    n = int(max(pa.shape[0], pb.shape[0]))
    probs = (np.arange(n, dtype=float) + 0.5) / float(n)
    qa = np.asarray(np.quantile(pa, probs[:, None], axis=0), dtype=float)
    qb = np.asarray(np.quantile(pb, probs[:, None], axis=0), dtype=float)
    return np.asarray(np.mean(np.abs(qa - qb) ** p, axis=0) ** (1.0 / p), dtype=float)


def _aggregate_w(w: Array, kind: str, p: float) -> float:
    """SW_p = (mean_l w_l^p)^(1/p) for kind="sw"; MSW_p = max_l w_l for "msw"."""
    if kind == "msw":
        return float(np.max(w))
    return float(np.mean(w**p) ** (1.0 / p))


# ------------------------------------------------------------- distances


def sliced_wasserstein_distance(
    A: Array,
    B: Array,
    p: float = 2.0,
    n_projections: int = 128,
    projection: str = "random",
    seed: int = 0,
) -> float:
    """Sliced Wasserstein distance SW_p between two empirical point clouds.

        SW_p = ( (1/L) sum_l W_p^p(theta_l#A, theta_l#B) )^{1/p},

    with directions theta_l from :func:`random_projections` (seeded Monte
    Carlo over the sphere, Rabin et al. 2010) or
    :func:`deterministic_projections` (equally-spaced / Sobol', Bonneel et
    al. 2015). Each projected 1-D W_p uses the exact monotone quantile
    coupling of ``wasserstein_1d``; unequal sample sizes are handled by the
    common mid-probability quantile grid of that routine.

    Known properties (Gao & Xiu 2021; Nadjahi et al. 2021,
    arXiv:2003.07817): SW_p(P, P) = 0; SW_p <= W_p (projections are
    1-Lipschitz); W_2 <= C_d * SW_2 with C_d ~ sqrt(d), sharp for isotropic
    Gaussian scale shifts; for d = 1, SW_p = W_p exactly.

    Fail-closed: empty/non-finite clouds, dimension mismatch, p < 1,
    n_projections < 1, unknown projection mode or negative seed raise
    ValueError.
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    _check_dim_match(a, b)
    q = _check_p(p)
    lp = _check_n_projections(n_projections)
    _check_projection_mode(projection)
    _check_seed(seed)
    theta = _projection_matrix(int(a.shape[1]), lp, projection, seed)
    pa = np.asarray(a @ theta.T, dtype=float)
    pb = np.asarray(b @ theta.T, dtype=float)
    w = np.array([wasserstein_1d(pa[:, l_idx], pb[:, l_idx], p=q) for l_idx in range(lp)])
    return _aggregate_w(w, "sw", q)


def max_sliced_wasserstein_distance(
    A: Array,
    B: Array,
    p: float = 2.0,
    n_projections: int = 128,
    projection: str = "random",
    seed: int = 0,
) -> float:
    """Max-sliced Wasserstein distance MSW_p over the direction set.

        MSW_p = max_l W_p(theta_l#A, theta_l#B),

    the finite-direction approximation of the sup over slices (Rabin et al.
    2010; Kolouri, Zou & Rohde 2018 for its use as a detection statistic;
    Kolouri et al. 2019, arXiv:1905.12958, for the generalized-sliced view).
    On a common direction set, SW_p <= MSW_p <= W_p: the max dominates the
    p-mean, and each projected W_p is dominated by the ambient W_p since
    theta# is 1-Lipschitz. For d = 1, MSW_p = W_p exactly.

    Parameters and fail-closed edges as in :func:`sliced_wasserstein_distance`.
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    _check_dim_match(a, b)
    q = _check_p(p)
    lp = _check_n_projections(n_projections)
    _check_projection_mode(projection)
    _check_seed(seed)
    theta = _projection_matrix(int(a.shape[1]), lp, projection, seed)
    pa = np.asarray(a @ theta.T, dtype=float)
    pb = np.asarray(b @ theta.T, dtype=float)
    w = np.array([wasserstein_1d(pa[:, l_idx], pb[:, l_idx], p=q) for l_idx in range(lp)])
    return _aggregate_w(w, "msw", q)


# ------------------------------------------------------------- test


def sliced_wasserstein_test(
    A: Array,
    B: Array,
    statistic: str = "sw",
    p: float = 2.0,
    n_projections: int = 64,
    projection: str = "random",
    seed: int = 0,
    n_perm: int = 999,
) -> dict[str, float]:
    """Permutation two-sample goodness-of-fit test with an SW/MSW statistic.

    Observed statistic: SW_p (``statistic="sw"``) or MSW_p
    (``statistic="msw"``) between the clouds on a fixed direction set. Null
    distribution: the same statistic recomputed on ``n_perm`` random splits
    of the pooled projections (label permutations), so the directions and the
    quantile-grid coupling are identical under observed and null. The
    p-value is exactly calibrated and never zero,

        pvalue = (1 + #{perm stat >= observed}) / (n_perm + 1),

    counting the observed split itself (Phipson & Smyth 2010). Under H0
    (exchangeable labels) this is a valid level-alpha test; against fixed
    alternatives SW-based tests are consistent (Ramdas, Garcia Trillos &
    Cuturi 2017), like the energy-distance permutation test they mirror
    (Rizzo & Székely 2016) — the energy distance is also a sphere-average of
    a 1-D discrepancy, but of the all-pairs energy form rather than the
    optimal-coupling W_p.

    Returns a dict with ``statistic``, ``pvalue``, ``n_perm``, ``n_a``,
    ``n_b`` and ``n_projections`` (all floats). Fail-closed: bad clouds,
    dimension mismatch, unknown statistic kind, p < 1, n_projections < 1,
    n_perm < 1 raise ValueError.
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    _check_dim_match(a, b)
    kind = _check_statistic_kind(statistic)
    q = _check_p(p)
    lp = _check_n_projections(n_projections)
    _check_projection_mode(projection)
    s = _check_seed(seed)
    nper = _check_n_perm(n_perm)
    theta = _projection_matrix(int(a.shape[1]), lp, projection, s)
    pa = np.asarray(a @ theta.T, dtype=float)
    pb = np.asarray(b @ theta.T, dtype=float)
    observed = _aggregate_w(_wp_columns(pa, pb, q), kind, q)
    pooled = np.vstack((pa, pb))
    n_a = int(pa.shape[0])
    n_tot = int(pooled.shape[0])
    rng = np.random.default_rng(s)
    count_ge = 1  # the observed split counts as >= itself (Phipson & Smyth 2010)
    for _ in range(nper):
        perm = rng.permutation(n_tot)
        stat = _aggregate_w(_wp_columns(pooled[perm[:n_a]], pooled[perm[n_a:]], q), kind, q)
        if stat >= observed:
            count_ge += 1
    return {
        "statistic": float(observed),
        "pvalue": float(count_ge) / float(nper + 1),
        "n_perm": float(nper),
        "n_a": float(n_a),
        "n_b": float(b.shape[0]),
        "n_projections": float(lp),
    }


# ------------------------------------------------------------- barycenter


@dataclass(frozen=True)
class SlicedBarycenter:
    """SW barycenter fixed-point result with convergence diagnostics.

    ``points`` is the (n, dim) barycenter cloud; ``converged`` is True iff
    the relative sup-norm change dropped to ``<= tol`` within ``max_iter``
    sweeps; ``final_delta`` is that last relative change (0.0 on exact
    fixed points); ``n_iter`` counts executed sweeps. A ``converged=False``
    result is a failed convergence check, not a barycenter to trust.
    """

    points: Array
    n_iter: int
    converged: bool
    final_delta: float
    n_projections: int


def _check_clouds(clouds: Sequence[Array]) -> tuple[list[Array], int, int]:
    if len(clouds) == 0:
        raise ValueError("clouds must contain at least one point cloud")
    checked = [
        _check_point_cloud(f"clouds[{k}]", np.asarray(c, dtype=float)) for k, c in enumerate(clouds)
    ]
    n = int(checked[0].shape[0])
    dim = int(checked[0].shape[1])
    for k, c in enumerate(checked):
        if int(c.shape[1]) != dim:
            raise ValueError(
                f"dimension mismatch: clouds[0] has dim {dim}, clouds[{k}] has dim {c.shape[1]}"
            )
        if int(c.shape[0]) != n:
            raise ValueError(
                f"cardinality mismatch: clouds[0] has {n} points, clouds[{k}] has {c.shape[0]}"
            )
    return checked, n, dim


def sliced_wasserstein_barycenter(
    clouds: Sequence[Array],
    weights: Array | None = None,
    n_projections: int = 64,
    projection: str = "random",
    seed: int = 0,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> SlicedBarycenter:
    """Sliced-Wasserstein barycenter of equal-cardinality point clouds.

    Fixed-point iteration of Bonneel et al. (2015): initialize Z at the
    weighted pointwise mean of the inputs; then, per sweep and per direction
    theta_l, sort the projections of Z and of every input cloud, rank-match
    the j-th smallest barycenter point with the j-th smallest point of each
    input, and move Z to the weighted mean of its matched points. The
    iteration preserves the weighted mean of the inputs exactly at every
    sweep (rank matching only permutes points within each cloud).

    Documented convergence check: ``converged`` is True iff the relative
    sup-norm displacement ``final_delta`` of the barycenter drops to
    ``<= tol`` within ``max_iter`` sweeps. This is a POCS-style heuristic,
    not a certified descent method: for a finite direction set in dim >= 2
    the per-direction rank constraints are generally mutually inconsistent
    (the averaged quantile slices need not be realizable by a single
    n-atom cloud), so the iteration can cycle indefinitely — a
    ``converged=False`` return is honest evidence of non-convergence and
    the output must not be treated as a certified barycenter. Where the
    constraints are consistent — dim = 1 (all directions are +/- the
    identity), identical inputs, a single cloud, point-mass inputs — the
    iteration converges exactly, typically in one or two sweeps.

    Requires equal cardinality across inputs (the rank coupling matches the
    j-th order statistics; unequal sizes would need mass splitting, which
    this routine deliberately does not fake). Works for 1-D clouds (n, 1)
    and any d >= 1; for identical inputs the barycenter is exactly that
    input, reached in one sweep with ``final_delta = 0``.

    Fail-closed: empty cloud list, empty/non-finite clouds, dimension or
    cardinality mismatch, bad weights (wrong length, negative, non-finite,
    zero total), n_projections < 1, max_iter < 1, tol <= 0 raise ValueError.
    """
    checked, n, dim = _check_clouds(clouds)
    w = _check_weights("weights", weights, len(checked))
    lp = _check_n_projections(n_projections)
    _check_projection_mode(projection)
    s = _check_seed(seed)
    mi = int(max_iter)
    if mi < 1:
        raise ValueError("max_iter must be >= 1")
    t = float(tol)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("tol must be finite and > 0")
    theta = _projection_matrix(dim, lp, projection, s)
    stack = np.asarray(np.stack(checked), dtype=float)  # (K, n, dim)
    # sorted_stack[k, j, l, :] = k-th cloud's j-th smallest point along theta_l.
    proj = np.asarray(stack @ theta.T, dtype=float)  # (K, n, L)
    orders = np.argsort(proj, axis=1, kind="stable")  # (K, n, L)
    n_clouds = len(checked)
    sorted_stack = np.asarray(
        np.stack([stack[k][orders[k]] for k in range(n_clouds)]), dtype=float
    )  # (K, n, L, dim): stack[k][orders[k]] indexes (n, L) -> (n, L, dim)
    z = np.asarray(np.tensordot(w, stack, axes=(0, 0)), dtype=float)  # (n, dim)
    n_iter = 0
    delta_rel = float("inf")
    converged = False
    for _ in range(mi):
        n_iter += 1
        z_new = np.zeros_like(z)
        for l_idx in range(lp):
            order_z = np.argsort(np.asarray(z @ theta[l_idx], dtype=float), kind="stable")
            matched = np.asarray(
                np.tensordot(w, sorted_stack[:, :, l_idx, :], axes=(0, 0)), dtype=float
            )  # (n, dim)
            z_new[order_z] += matched
        z_new /= float(lp)
        delta = float(np.max(np.abs(z_new - z)))
        scale = max(1.0, float(np.max(np.abs(z_new))))
        delta_rel = delta / scale
        z = z_new
        if delta_rel <= t:
            converged = True
            break
    return SlicedBarycenter(
        points=z,
        n_iter=n_iter,
        converged=converged,
        final_delta=delta_rel,
        n_projections=lp,
    )
