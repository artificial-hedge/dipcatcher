"""Wasserstein / optimal-transport metrics for predictive distributions.

Distances between predictive distributions built on optimal transport
(Villani, 2003, *Topics in Optimal Transportation*, Graduate Studies in
Math. 58, AMS, doi:10.1090/gsm/058; Villani, 2009, *Optimal Transport:
Old and New*, Springer, doi:10.1007/978-3-540-71050-9). The Wasserstein
(p-OT) distance metrizes weak convergence plus convergence of p-th moments,
so it is a natural "calibration + sharpness" summary for forecast ensembles:
a forecast can rank well yet sit far in Wasserstein from the realized
distribution if its dispersion is wrong.

1. ``wasserstein_1d``: exact p-Wasserstein between two empirical measures on
   the line via the monotone (quantile) coupling. In 1-D the sorted-sample
   (quantile) coupling is optimal (Villani 2003, Ch. 2; the monotone
   rearrangement of one distribution into the other minimizes every convex
   transport cost), so for equal-size samples the mean of |a_(i) - b_(i)|^p
   over the joint rank coupling, raised to 1/p, is exact. Unequal sizes are
   handled by interpolating both empirical quantile functions on a common
   mid-probability grid (a standard, monotone coupling approximation).
2. ``wasserstein_gaussian``: closed-form W2 between Gaussians,
       W2^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2}),
   with the matrix square root via ``scipy.linalg.sqrtm`` and PSD
   symmetrization (Gelbrich, 1990, Math. Oper. Res. 15(3):592-608,
   doi:10.1287/moor.15.3.592; the Bures-Wasserstein form and its geometry:
   Bhatia, Jain & Lim, 2019, "On the Bures-Wasserstein distance: computation
   for matrices", Expositiones Math. 37(3):292-304, arXiv:1801.09287).
3. ``sinkhorn_distance``: entropic OT between empirical measures with
   squared-Euclidean cost, computed with the log-domain stabilized Sinkhorn
   iteration (Cuturi, 2013, "Sinkhorn Distances: Lightspeed Computation of
   Optimal Transport", NeurIPS 26, arXiv:1306.0895; the log-domain
   formulation with running maxima — equivalently logsumexp centering — is
   the standard fix for the overflow/underflow of the bare matrix-scaling
   iteration when entries of -C/reg leave the floating-point range, cf.
   Schmitzer, 2019, "Stabilized Sparse Scaling Algorithms for Entropic
   Transport Problems", SIAM J. Sci. Comput. 41(3):A1443-A1481,
   arXiv:1610.06519). Returns the primal transport cost <P, C> of the
   entropic plan P, which converges to the exact OT cost as reg -> 0+.
4. ``sinkhorn_divergence``: the debiased divergence
       SD(A, B) = S(A, B) - (S(A, A) + S(B, B)) / 2,
   which removes the entropic self-bias of S and yields a symmetric,
   positive-definite divergence usable on samples of different sizes
   (Genevay, Peyre & Cuturi, 2018, "Learning Generative Models with Sinkhorn
   Divergences", AISTATS, PMLR 84:2348-2357, arXiv:1706.00292; Ramdas,
   Garcia Trillos & Cuturi, 2017, "On Wasserstein two-sample testing and
   related families of nonparametric tests", Ann. Inst. Statist. Math.
   69(2):679-694, arXiv:1509.02237).
5. ``energy_distance_from_ot``: bridge to the energy score of
   ``quant_fund.metrics.energy_score`` — the two-sample energy distance
   D^2(X, Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'|| (Székely, 2003,
   "Statistics on the energy distance", InterStat; Székely & Rizzo, 2013,
   J. Statist. Plann. Inference 143(8):1249-1272, arXiv:1210.3927).
   Relation to the other quantities in this module: the energy score of a
   forecast ensemble x_1..x_n against an observation y is the energy
   distance between the forecast and a point mass at y (Gneiting & Raftery,
   2007, JASA 102(477):359-378, doi:10.1198/016214506000001437, kernel-score
   view), and energy distance is dominated by W1 in the sense
   D^2(X, Y) <= 2 W1(X, Y) (each of the three expectations in D^2 is taken
   under SOME coupling, while W1 minimizes the first-moment cost over all
   couplings), so energy distance is the "average-coupling" counterpart of
   the "best-coupling" W1.

Honesty: outputs are distributional distances and transport costs — proper
scoring/diagnostic quantities. No Sharpe/Sortino/P&L content; no
live-trading claims (AGENTS.md honesty contract).

Conventions: numpy core, frozen-dataclass-free functional API (stateless
distances), fail-closed edges (ValueError on empty, non-finite, mismatched
dimension, negative weights, reg <= 0, non-PSD covariance; nothing is
silently dropped), exact in the equal-size 1-D case.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

Array = NDArray[np.float64]

__all__ = [
    "energy_distance_from_ot",
    "sinkhorn_divergence",
    "sinkhorn_distance",
    "wasserstein_1d",
    "wasserstein_gaussian",
]

_PSD_TOL = 1e-10
_EPS = 1e-300


def _check_1d_sample(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_p(p: float) -> float:
    q = float(p)
    if not np.isfinite(q) or q < 1.0:
        raise ValueError("p must be finite and >= 1")
    return q


def wasserstein_1d(a: Array, b: Array, p: float = 1.0) -> float:
    """Exact p-Wasserstein between two 1-D empirical samples.

    W_p^p(P, Q) = inf over couplings pi of E_pi[|X - Y|^p]. On the line the
    infimum is attained by the monotone (quantile) coupling: for equal-size
    samples the rank coupling (a_(i), b_(i)) is optimal for every convex cost
    (Villani 2003, Ch. 2 — monotone rearrangement minimizes all convex
    transport costs in 1-D), so

        W_p = (mean_i |a_(i) - b_(i)|^p)^{1/p}

    is exact. For unequal sizes both empirical quantile functions are
    evaluated on a common mid-probability grid (the same monotone coupling
    read at higher resolution); the result is monotone in each sample and
    exact when the sizes coincide.

    Parameters
    ----------
    a, b : non-empty finite 1-D samples.
    p : moment order, finite and >= 1 (default 1 gives the earth-mover/W1).

    Returns
    -------
    float
        The p-th root of the coupled p-th moment (not the p-th power).
    """
    x = _check_1d_sample("a", a)
    y = _check_1d_sample("b", b)
    q = _check_p(p)
    n = int(max(x.size, y.size))
    probs = (np.arange(n, dtype=float) + 0.5) / float(n)
    qx = np.quantile(x, probs)
    qy = np.quantile(y, probs)
    return float(np.mean(np.abs(qx - qy) ** q) ** (1.0 / q))


def _check_mean(name: str, mu: Array) -> Array:
    arr = np.asarray(mu, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_cov(name: str, cov: Array, dim: int) -> Array:
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if not bool(np.allclose(arr, arr.T, atol=1e-8, rtol=1e-8)):
        raise ValueError(f"{name} must be symmetric")
    sym: Array = 0.5 * (arr + arr.T)
    min_eig = float(np.min(np.linalg.eigvalsh(sym)))
    if min_eig < -_PSD_TOL:
        raise ValueError(f"{name} must be positive semidefinite")
    return sym


def _matrix_sqrt_psd(mat: Array, name: str) -> Array:
    """PSD matrix square root via scipy.linalg.sqrtm with symmetrization.

    Numerical noise can make ``sqrtm`` return a complex result or a mildly
    asymmetric one even for PSD input; the imaginary part is dropped when it
    is at noise level (1e-8 relative) and the result is re-symmetrized.
    """
    root = sqrtm(mat)
    if np.iscomplexobj(root):
        max_imag = float(np.max(np.abs(np.imag(root)))) if root.size else 0.0
        scale = float(max(np.max(np.abs(root)), 1.0))
        if max_imag > 1e-8 * scale:
            raise ValueError(f"matrix square root of {name} is not real")
        root = np.real(root)
    root_arr = np.asarray(root, dtype=float)
    return 0.5 * (root_arr + root_arr.T)


def wasserstein_gaussian(
    mu1: Array,
    cov1: Array,
    mu2: Array,
    cov2: Array,
) -> float:
    """Closed-form W2 between two Gaussians (squared, then square root).

    For P = N(mu1, S1), Q = N(mu2, S2) the 2-Wasserstein distance has the
    closed form (Gelbrich 1990, Thm. 2.1; Bhatia, Jain & Lim 2019 for the
    Bures-Wasserstein matrix geometry):

        W2^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2}).

    The inner matrix S1^{1/2} S2 S1^{1/2} is PSD whenever S1, S2 are, so its
    square root is real; both roots use ``scipy.linalg.sqrtm`` with PSD
    symmetrization. The trace term is zero iff S1 = S2, so equal covariances
    give exactly ||mu1 - mu2||.

    Fail-closed: means must be non-empty and finite, covariances square,
    symmetric, finite, and positive semidefinite (eigenvalue floor at
    -1e-10); mismatched dimensions raise ValueError.
    """
    m1 = _check_mean("mu1", mu1)
    m2 = _check_mean("mu2", mu2)
    if m1.shape != m2.shape:
        raise ValueError(f"mean dimension mismatch: mu1 has dim {m1.size}, mu2 has dim {m2.size}")
    s1 = _check_cov("cov1", cov1, int(m1.size))
    s2 = _check_cov("cov2", cov2, int(m1.size))
    delta = m1 - m2
    mean_term = float(delta @ delta)
    root1 = _matrix_sqrt_psd(s1, "cov1")
    middle = root1 @ s2 @ root1
    middle = 0.5 * (middle + middle.T)
    root_middle = _matrix_sqrt_psd(middle, "S1^{1/2} S2 S1^{1/2}")
    trace_term = float(np.trace(s1 + s2 - 2.0 * root_middle))
    # Symmetric PSD arithmetic can leave a few ulps of negative slack.
    trace_term = max(trace_term, 0.0)
    return float(mean_term + trace_term)


def _check_point_cloud(name: str, x: Array) -> Array:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D (n_samples, dim) array")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _check_weights(name: str, weights: Array | None, n: int) -> Array:
    if weights is None:
        return np.full(n, 1.0 / float(n), dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != n:
        raise ValueError(f"{name} must have one entry per sample ({n}), got {w.size}")
    if not bool(np.all(np.isfinite(w))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    if bool(np.any(w < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return w / total


def _check_reg(reg: float) -> float:
    r = float(reg)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("reg must be finite and > 0")
    return r


def _sinkhorn_cost(
    a: Array,
    b: Array,
    weights_a: Array | None,
    weights_b: Array | None,
    reg: float,
    max_iter: int,
    tol: float,
) -> float:
    """Log-domain stabilized Sinkhorn primal cost <P, C>, C = squared L2."""
    wa = _check_weights("weights_a", weights_a, int(a.shape[0]))
    wb = _check_weights("weights_b", weights_b, int(b.shape[0]))
    cost = cdist(a, b, metric="sqeuclidean")
    log_kernel = -cost / reg
    log_wa = np.log(np.maximum(wa, _EPS))
    log_wb = np.log(np.maximum(wb, _EPS))
    log_u = np.zeros(a.shape[0], dtype=float)
    log_v = np.zeros(b.shape[0], dtype=float)
    for _ in range(int(max_iter)):
        log_u_new = log_wa - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v_new = log_wb - logsumexp(log_kernel + log_u_new[:, None], axis=0)
        delta = max(
            float(np.max(np.abs(log_u_new - log_u))),
            float(np.max(np.abs(log_v_new - log_v))),
        )
        log_u, log_v = log_u_new, log_v_new
        if delta < tol:
            break
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    # logsumexp over the plan gives log of row-summed masses; row sums equal
    # wa by construction, so only the explicit <P, C> contraction is needed.
    plan = np.exp(log_plan)
    return float(np.sum(plan * cost))


def sinkhorn_distance(
    A: Array,
    B: Array,
    weights_a: Array | None = None,
    weights_b: Array | None = None,
    reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> float:
    """Entropic OT transport cost between two empirical measures.

    Solves min_{P in Pi(w_a, w_b)} <P, C> + reg * sum_ij P_ij (log P_ij - 1)
    with the squared-Euclidean cost C_ij = ||a_i - b_j||^2, via the
    log-domain stabilized Sinkhorn matrix-scaling iteration (Cuturi 2013;
    Schmitzer 2019 for the stabilization). The returned value is the primal
    transport cost <P, C> of the entropic plan, which converges to the exact
    OT cost as reg -> 0+ (for equal-size uniform weights this is exactly the
    assignment problem, cf. ``scipy.optimize.linear_sum_assignment``).

    Parameters
    ----------
    A, B : non-empty finite 2-D point clouds (n, dim) / (m, dim) with a
        shared ambient dimension.
    weights_a, weights_b : optional nonnegative probability weights
        (normalized internally); None gives uniform weights.
    reg : entropic regularization, finite and > 0 (smaller reg -> closer to
        exact OT but slower, less stable iteration).
    max_iter : cap on Sinkhorn sweeps.
    tol : stopping tolerance on the sup-norm change of the log-scalings.

    Fail-closed: empty/non-finite point clouds, dimension mismatch,
    negative or non-finite weights, reg <= 0 raise ValueError.
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: A has dim {a.shape[1]}, B has dim {b.shape[1]}")
    r = _check_reg(reg)
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and > 0")
    return _sinkhorn_cost(a, b, weights_a, weights_b, r, int(max_iter), float(tol))


def sinkhorn_divergence(
    A: Array,
    B: Array,
    reg: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> float:
    """Debiased Sinkhorn divergence between two empirical measures.

        SD(A, B) = S(A, B) - (S(A, A) + S(B, B)) / 2,

    where S(X, Y) is the entropic OT cost at regularization ``reg`` (Cuturi
    2013). The raw entropic cost carries a reg-dependent self-bias — S(X, X)
    is not zero — that inflates cross-comparisons; subtracting the
    within-sample terms removes it (Genevay, Peyre & Cuturi 2018). The
    result is symmetric, vanishes exactly when A == B, and is positive
    definite between distinct measures in the reg -> 0 limit, where it
    approaches (a monotone transform of) the squared W2 (Ramdas, Garcia
    Trillos & Cuturi 2017). Uniform weights on each cloud.

    Parameters as in :func:`sinkhorn_distance` (no weights: the divergence
    needs the within-cloud terms, which only make sense for the uniform
    empirical measure of each cloud).
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: A has dim {a.shape[1]}, B has dim {b.shape[1]}")
    r = _check_reg(reg)
    if int(max_iter) < 1:
        raise ValueError("max_iter must be >= 1")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and > 0")
    sab = _sinkhorn_cost(a, b, None, None, r, int(max_iter), float(tol))
    saa = _sinkhorn_cost(a, a, None, None, r, int(max_iter), float(tol))
    sbb = _sinkhorn_cost(b, b, None, None, r, int(max_iter), float(tol))
    return float(sab - 0.5 * (saa + sbb))


def energy_distance_from_ot(A: Array, B: Array) -> float:
    """Two-sample energy distance between empirical point clouds.

        D^2(A, B) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||,

    with X, X' iid uniform over A and Y, Y' iid uniform over B (Euclidean
    norm; Székely 2003; Székely & Rizzo 2013). This is the bridge to
    ``quant_fund.metrics.energy_score``: the energy score of an ensemble
    against an observation is the energy distance between the ensemble and a
    point mass at the observation (the kernel-score view of Gneiting &
    Raftery 2007), and D^2 <= 2 W1 pointwise-in-coupling, so energy distance
    is the all-pairs-coupling counterpart of the optimal-coupling W1.

    Fail-closed: empty/non-finite clouds or dimension mismatch raise
    ValueError.
    """
    a = _check_point_cloud("A", A)
    b = _check_point_cloud("B", B)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dimension mismatch: A has dim {a.shape[1]}, B has dim {b.shape[1]}")
    cross = float(np.mean(cdist(a, b, metric="euclidean")))
    within_a = float(np.mean(cdist(a, a, metric="euclidean")))
    within_b = float(np.mean(cdist(b, b, metric="euclidean")))
    return float(2.0 * cross - within_a - within_b)
