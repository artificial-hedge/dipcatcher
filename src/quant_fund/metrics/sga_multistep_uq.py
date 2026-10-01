"""SGA: DAG-structured multi-step forecast uncertainty quantification.

This lane implements the uncertainty-quantification machinery of the SGA
(Slicing-Graphing-Alignment) method — a directed-acyclic-graph scheme that
characterizes the *forecasting space* of multi-step forecasts and bounds
its uncertainty by a graph complexity — plus a linear-Gaussian analytic
counterpart that propagates per-step forecast-error variance through a
horizon-lag DAG exactly. Implements:

    Hu, X.-Y., Liang, S., Feng, C., Zhang, S.-Q. (2026). "SGA: Uncertainty
    Quantification for Multi-Step Forecasting in Time Series Foundation
    Models." arXiv:2609.28582 [cs.LG] (23 Sep 2026).
    Citation verified against https://arxiv.org/abs/2609.28582 and the
    full HTML text (fetched 2026-09-30); section/algorithm references
    below are to the fetched text.

Paper machinery implemented (§3.2 unless noted)
-----------------------------------------------
1. **HUQ formulation** (§3.1): attach an uncertainty estimate to the whole
   h-step forecast trajectory; step-wise error accumulation is exactly what
   the DAG propagation quantifies.
2. **Slicing** (Alg. 1): K sampled trajectories segmented into
   ``n = ceil(h / l_s)`` consecutive slices of length ``l_s`` (default 4,
   Appendix A); the last slice takes the remainder when ``l_s`` does not
   divide ``h``. Slices at the same index j are candidate forecast
   branches over the same temporal span.
3. **Graphing** (Alg. 1): the historical series is the root; each slice is
   a node; edges encode temporal dependency (root -> first slice, slice_j
   -> slice_{j+1} within each trajectory).
4. **Alignment** (Alg. 1): within each slice index j, merge node pairs
   whose DTW distance (Sakoe & Chiba 1978) is at most the seasonal-scaled
   threshold ``tau = lam/(t-S) * sum_i |x_i - x_{i+S}|`` (Hyndman &
   Athanasopoulos 2018 seasonal-naive scaling; ``lam = 0.25`` default).
   Merged nodes union their slice-uncertainty multisets and inherit both
   endpoints' edges — revealing the intrinsic topology of the forecasting
   space. The paper iterates an unordered node set; this implementation
   uses creation order (trajectory index ascending) so results are
   deterministic and bit-identical across calls.
5. **Step- and slice-level uncertainty** (§3.2): for trajectory-based
   models the per-step density is a Gaussian KDE over the K samples with
   bandwidth ``eta_s = K^{-1/5} * sigma_s`` (the paper's form of Scott's
   rule — Scott 1992). Step uncertainty is the plug-in cross-entropy
   ``U_s = -(1/K) * sum_k log P_s(x_k_s)``; slice uncertainty averages its
   steps; a merged node's uncertainty averages its multiset. Honest
   caveat: the plug-in estimator is biased relative to the true
   differential entropy — it is a consistent branch-dispersion score, not
   an exact entropy.
6. **Graph complexity** (Alg. 2): Bonacich-Lloyd alpha-centrality
   ``B(v) = U(v) + alpha * sum_{(w,v) in E} B(w)`` over a topological
   order (``alpha = 0.1``, Bucur & Holme 2020), ``GC(G) = sum_v B(v)`` —
   integrating topological information with model-inherent stochasticity.

Lane extension — the analytic linear-Gaussian DAG (spec-mandated)
-----------------------------------------------------------------
The paper quantifies a sampled forecasting space; this lane additionally
equips the same DAG/centrality machinery with an exact covariance
propagation on a *forecast-error* DAG whose edges are horizon-lag linear
coefficients:

    e_s = sum_{r<s} A[r,s] * e_r + eps_s,   eps_s ~ (0, sigma^2_s) iid.

``propagate_uncertainty`` walks the DAG in topological order, filling the
error covariance by the trek-rule recursion

    Cov(e_r, e_s) = sum_i A[i,s] Cov(e_r, e_i)          (r < s)
    Var(e_s)      = sum_{i,j} A[i,s] A[j,s] C[i,j] + sigma^2_s,

which equals the closed form ``Sigma = (I - A^T)^{-1} D (I - A)^{-1}``
exactly (verified at 1e-12 in tests). ``estimate_error_dag`` fits the
edges by per-step OLS of each residual-panel column on its up-to
``max_lag`` predecessors (columns are demeaned — the DAG models the
dispersion structure, not forecast bias); ``dag_from_covariance`` is the
banded ordered factorization of an error covariance matrix (modified
Cholesky; ``max_lag = h - 1`` reproduces the input covariance exactly).

The paper's statement that "graph complexity bounds the uncertainty" is
made a concrete certificate in variance units: taking
``U(v) := Var(e_v)`` in the centrality recursion gives
``B(v) >= U(v)`` for every node (induction on the topological order —
``alpha >= 0``, ``U >= 0``, edges only add non-negative ancestor mass),
and ``GC(G) >= sum_s Var(e_s)``. ``variance_bound_certificate`` returns
both per-step bounds and the total bound with their margins; the
topological amplification ``B(v_s) - Var(e_s)`` measures how much
predecessor branching inflates a step's certified uncertainty.

Outputs / honesty
-----------------
* Uncertainty outputs are per-horizon ``std``, Gaussian interval
  half-widths ``z * std``, the propagated covariance, per-node centrality,
  and the complexity bounds. These are proper distribution diagnostics
  (variance/entropy/coverage) — never Sharpe/Sortino/P&L; no sim-internal
  PnL exists here at all.
* ``simulate_synthetic_dag_errors`` / ``var1_error_dag`` /
  ``make_synthetic_ar1_forecaster`` are seeded SYNTHETIC harnesses for
  correctness tests only — never market evidence.
* Fail-closed: cyclic or backward edge sets, ragged/short/non-finite
  panels, non-PD covariances, degenerate step distributions, and invalid
  hyperparameters all raise ``ValueError``.
* Deterministic: every stochastic path flows through a caller-supplied
  ``numpy.random.Generator``; merging iterates a fixed creation order, so
  repeated calls are bit-identical.

Composition: the Gaussian KDE evaluation is imported from
``metrics.kde.gaussian_kde`` (never reimplemented); 1-d guards come from
``utils.series``. DTW has no existing repo implementation, so the
textbook DP is provided here.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from quant_fund.metrics.kde import gaussian_kde
from quant_fund.utils.series import as_named_1d

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]

STUDY_SCHEMA = "sga_multistep_uq.v1"

#: Paper defaults (Appendix A): slicing length l_s = 4, threshold
#: coefficient lambda = 0.25, attenuation factor alpha = 0.1.
DEFAULT_SLICE_LEN = 4
DEFAULT_LAMBDA = 0.25
DEFAULT_ALPHA = 0.1
#: KDE step-entropy needs enough trajectory samples to be meaningful
#: (matches gaussian_kde's minimum-observation contract).
MIN_SAMPLES = 10
#: Numerical floor for estimated innovation variances; a step whose own
#: variance is at/below this is degenerate and rejected instead.
MIN_INNOVATION_VAR = 1e-12


def _as_square_matrix(values: object, name: str) -> Array:
    m = np.asarray(values, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not bool(np.isfinite(m).all()):
        raise ValueError(f"{name} must be finite")
    return m


def _as_positive_1d(values: object, name: str, length: int) -> Array:
    v = as_named_1d(name, np.asarray(values, dtype=float))
    if v.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    if not bool(np.isfinite(v).all()):
        raise ValueError(f"{name} must be finite")
    if bool(np.any(v <= 0.0)):
        raise ValueError(f"{name} must be strictly positive")
    return v


def _require_residual_panel(residuals: object, max_lag: int) -> Array:
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2:
        raise ValueError("residuals must be a 2d (n_panels, horizon) panel")
    n, h = r.shape
    if h < 1:
        raise ValueError("residuals must have at least one horizon step")
    min_n = max(8, max_lag + 2)
    if n < min_n:
        raise ValueError(f"residuals need >= {min_n} panels (got {n})")
    if not bool(np.isfinite(r).all()):
        raise ValueError("residuals must be finite")
    return r


def _edges_from_coefficients(coefficients: Array) -> IntArray:
    src, dst = np.nonzero(coefficients)
    order = np.lexsort((dst, src))
    return np.stack([src[order], dst[order]], axis=1).astype(np.intp)


def _topological_order(n_nodes: int, edges: IntArray) -> IntArray:
    """Kahn's algorithm; deterministic (smallest id first). Raises on cycles."""
    indeg = np.zeros(n_nodes, dtype=np.intp)
    succ: list[list[int]] = [[] for _ in range(n_nodes)]
    for p, c in edges:
        indeg[c] += 1
        succ[int(p)].append(int(c))
    heap = [int(v) for v in range(n_nodes) if indeg[v] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        v = heapq.heappop(heap)
        order.append(v)
        for c in succ[v]:
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, int(c))
    if len(order) != n_nodes:
        raise ValueError("edge set contains a directed cycle")
    return np.asarray(order, dtype=np.intp)


# ---------------------------------------------------------------------------
# 1. Error DAG: horizon-lag linear-Gaussian structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorDAG:
    """Linear-Gaussian structural equation model on horizon steps.

    Node ``s`` is the forecast error ``e_s`` at horizon step ``s``; edge
    ``r -> s`` with weight ``coefficients[r, s]`` means ``e_s`` loads on
    ``e_r`` — an autoregressive error-propagation edge. The model is

        e_s = sum_{r<s} A[r,s] e_r + eps_s,  eps_s ~ (0, innovation_var[s]),

    with independent innovations. Time order is a topological order, so
    valid coefficient matrices are strictly upper triangular (checked by
    ``build_error_dag``; cycles are rejected before orientation).
    """

    coefficients: Array  # (h, h) strictly upper triangular edge weights
    innovation_var: Array  # (h,) strictly positive

    @property
    def horizon(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def edges(self) -> IntArray:
        """(m, 2) edge list ``(r, s)`` over nonzero coefficients, sorted."""
        return _edges_from_coefficients(self.coefficients)

    def topo_order(self) -> IntArray:
        """Time order is the canonical topological order of this DAG."""
        return np.arange(self.horizon, dtype=np.intp)


def build_error_dag(coefficients: object, innovation_var: object) -> ErrorDAG:
    """Validate edge weights and innovation variances into an ``ErrorDAG``.

    Fail-closed: non-square/non-finite coefficients, self-loops, any
    directed cycle (checked via Kahn before the orientation rule so a
    cyclic input reports a cycle), edges that run backward in horizon
    order (``r >= s`` nonzero entries), and non-positive innovation
    variances all raise ``ValueError``.
    """
    a = _as_square_matrix(coefficients, "coefficients")
    h = a.shape[0]
    if bool(np.any(np.diag(a) != 0.0)):
        raise ValueError("coefficients must have a zero diagonal (no self-loops)")
    # Cycle check first: any nonzero cycle trips the DAG guard even though
    # the orientation rule below would also reject it.
    nonzero = (
        np.stack(np.nonzero(a), axis=1).astype(np.intp)
        if np.any(a)
        else (np.zeros((0, 2), dtype=np.intp))
    )
    _topological_order(h, nonzero)
    if bool(np.any(np.tril(a) != 0.0)):
        raise ValueError(
            "coefficients must be strictly upper triangular — edges point "
            "forward in horizon order (r < s)"
        )
    d = _as_positive_1d(innovation_var, "innovation_var", h)
    return ErrorDAG(coefficients=np.asarray(a, dtype=float), innovation_var=d)


def var1_error_dag(horizon: int, phi: float, sigma: float) -> ErrorDAG:
    """Planted VAR(1)-style error DAG: ``e_s = phi * e_{s-1} + eps_s``.

    Cold start: ``Var(e_1) = sigma^2`` (node 1 has no predecessor); the
    stationary-limit variance is ``sigma^2 / (1 - phi^2)`` for ``|phi| < 1``.
    SYNTHETIC construction helper — correctness scaffolding, not market
    evidence.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    p = float(phi)
    s = float(sigma)
    if not math.isfinite(p) or not math.isfinite(s) or s <= 0.0:
        raise ValueError("phi must be finite and sigma positive")
    a = np.zeros((h, h))
    if h > 1:
        a[np.arange(h - 1), np.arange(1, h)] = p
    d = np.full(h, s * s)
    return build_error_dag(a, d)


def dag_from_covariance(
    cov: object,
    max_lag: int | None = None,
    *,
    min_innovation_var: float = MIN_INNOVATION_VAR,
) -> ErrorDAG:
    """Factor an error covariance matrix into a horizon-lag DAG.

    Ordered modified-Cholesky factorization: each step ``s`` is regressed
    on its up-to-``max_lag`` predecessors via the covariance entries,
    ``A[L, s] = Sigma[L, L]^{-1} Sigma[L, s]`` with innovation variance
    the Schur complement ``Sigma[s,s] - c^T a``. With ``max_lag = h - 1``
    this is the unique ordered DAG factorization — propagation reproduces
    ``Sigma`` exactly. Banding yields the closest horizon-lag DAG in
    conditional-expectation terms.

    Fail-closed: non-symmetric input, non-finite entries, and any
    non-positive conditional variance (degenerate direction) raise
    ``ValueError``.
    """
    s_cov = _as_square_matrix(cov, "cov")
    h = s_cov.shape[0]
    if not bool(np.allclose(s_cov, s_cov.T, rtol=1e-10, atol=1e-12)):
        raise ValueError("cov must be symmetric")
    p = h - 1 if max_lag is None else int(max_lag)
    if p < 0 or p > h - 1:
        raise ValueError(f"max_lag must be in [0, {h - 1}]")
    a = np.zeros((h, h))
    d = np.empty(h)
    for s in range(h):
        lo = max(0, s - p)
        lags = np.arange(lo, s)
        if lags.size == 0:
            d[s] = s_cov[s, s]
        else:
            g = s_cov[np.ix_(lags, lags)]
            c = s_cov[lags, s]
            try:
                coef = np.linalg.solve(g, c)
            except np.linalg.LinAlgError as exc:
                raise ValueError("covariance block is singular — degenerate") from exc
            a[lags, s] = coef
            d[s] = s_cov[s, s] - float(c @ coef)
        if not math.isfinite(d[s]) or d[s] <= 0.0:
            raise ValueError(
                f"non-positive conditional variance at step {s} — "
                "covariance is degenerate along horizon order"
            )
    return build_error_dag(a, np.maximum(d, float(min_innovation_var)))


@dataclass(frozen=True)
class EstimatedErrorDAG:
    """Fitted horizon-lag error DAG plus estimation diagnostics."""

    dag: ErrorDAG
    bias: Array  # (h,) column means removed before fitting
    r_squared: Array  # (h,) per-step OLS fit quality
    innovation_var: Array  # (h,) fitted innovation variances
    max_lag: int
    n_panels: int


def estimate_error_dag(
    residuals: object,
    max_lag: int = 1,
    *,
    min_innovation_var: float = MIN_INNOVATION_VAR,
) -> EstimatedErrorDAG:
    """Estimate a horizon-lag error DAG from a residual panel.

    ``residuals`` is an ``(n_panels, horizon)`` array of realized
    multi-step forecast errors (one forecast origin per row). Columns are
    demeaned (forecast bias is reported separately — the DAG models the
    dispersion structure); each column ``e_s`` is then regressed on its up
    to ``max_lag`` predecessors by no-intercept OLS, giving edge weights
    ``A[r, s]`` and the innovation variance ``s2_s = SSE_s / (n - p_s)``.

    Fail-closed: ragged/non-2d panels, fewer than
    ``max(8, max_lag + 2)`` panels, non-finite entries, degenerate columns
    (zero dispersion), and ``max_lag`` outside ``[1, h - 1]`` (``h >= 2``)
    all raise ``ValueError``. ``h == 1`` is allowed — the DAG is a single
    node with the sample variance.

    Returns an ``EstimatedErrorDAG``: the fitted ``ErrorDAG``, per-step
    ``bias``, ``r_squared`` (per regression), ``innovation_var`` and the
    ``max_lag`` used.
    """
    r = _require_residual_panel(residuals, int(max_lag))
    n, h = r.shape
    p_max = int(max_lag)
    if p_max < 1:
        raise ValueError("max_lag must be >= 1")
    if h >= 2 and p_max > h - 1:
        raise ValueError(f"max_lag must be <= {h - 1} for horizon {h}")
    bias = r.mean(axis=0)
    e = r - bias
    col_var = e.var(axis=0, ddof=1)
    if bool(np.any(col_var <= 0.0)):
        raise ValueError("residual panel has a degenerate (constant) column")
    a = np.zeros((h, h))
    d = np.empty(h)
    r2 = np.zeros(h)
    d[0] = float(col_var[0])
    for s in range(1, h):
        p = min(p_max, s)
        x = e[:, s - p : s]
        y = e[:, s]
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        sse = float(resid @ resid)
        dof = n - p
        s2 = sse / dof if dof > 0 else float("nan")
        sst = float(y @ y)
        r2[s] = 1.0 - sse / sst if sst > 0 else float("nan")
        if not math.isfinite(s2) or s2 < 0.0:
            raise ValueError(f"innovation variance fit failed at step {s}")
        a[s - p : s, s] = beta
        d[s] = s2
    d = np.maximum(d, float(min_innovation_var))
    return EstimatedErrorDAG(
        dag=build_error_dag(a, d),
        bias=bias,
        r_squared=r2,
        innovation_var=d,
        max_lag=p_max,
        n_panels=n,
    )


# ---------------------------------------------------------------------------
# 2. Topological propagation (exact under the linear-Gaussian assumption)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropagatedUncertainty:
    """Exact propagated forecast-error covariance and marginal scales."""

    covariance: Array  # (h, h)
    variance: Array  # (h,) diagonal
    std: Array  # (h,)

    def interval_half_widths(self, z: float) -> Array:
        """Gaussian interval half-widths ``z * std`` per horizon step."""
        zz = float(z)
        if not math.isfinite(zz) or zz <= 0.0:
            raise ValueError("z must be a positive finite multiplier")
        return zz * self.std

    def intervals(self, center: object, z: float) -> tuple[Array, Array]:
        """``center +/- z * std`` per step (center broadcastable to (h,))."""
        c = as_named_1d("center", np.asarray(center, dtype=float))
        if c.shape[0] != self.std.shape[0]:
            raise ValueError("center must match the horizon")
        if not bool(np.isfinite(c).all()):
            raise ValueError("center must be finite")
        hw = self.interval_half_widths(z)
        return c - hw, c + hw


def propagate_uncertainty(dag: ErrorDAG) -> PropagatedUncertainty:
    """Propagate per-step variance through the DAG in topological order.

    Trek-rule recursion (exact under the linear-Gaussian SEM): processing
    steps in time order, each new step's covariances with earlier steps
    and its own variance are linear functions of the already-propagated
    block — identical to ``(I - A^T)^{-1} D (I - A)^{-1}`` up to floating
    point.
    """
    a = dag.coefficients
    d = dag.innovation_var
    h = dag.horizon
    cov = np.zeros((h, h))
    for s in range(h):
        w = a[:s, s]
        cross = cov[:s, :s] @ w if s > 0 else np.zeros(0)
        cov[:s, s] = cross
        cov[s, :s] = cross
        cov[s, s] = float(w @ cov[:s, :s] @ w) + d[s] if s > 0 else d[s]
    var = np.diag(cov).copy()
    return PropagatedUncertainty(covariance=cov, variance=var, std=np.sqrt(var))


def dag_covariance_closed_form(dag: ErrorDAG) -> Array:
    """Closed-form propagated covariance ``(I - A^T)^{-1} D (I - A)^{-1}``.

    Analytical reference for the topological recursion — kept as a public
    cross-check rather than a private helper because it is the textbook
    SEM expression (``e = (I - A^T)^{-1} eps``).
    """
    a = dag.coefficients
    h = dag.horizon
    eye = np.eye(h)
    m = eye - a.T
    w = np.linalg.solve(m, eye)
    return np.asarray(w @ np.diag(dag.innovation_var) @ w.T, dtype=float)


# ---------------------------------------------------------------------------
# 3. Bonacich-Lloyd alpha-centrality and the graph-complexity bound
# ---------------------------------------------------------------------------


def alpha_centrality(
    edges: object,
    n_nodes: int,
    node_uncertainty: object,
    alpha: float = DEFAULT_ALPHA,
    topo_order: IntArray | None = None,
) -> Array:
    r"""Bonacich & Lloyd (2001) alpha-centrality over a DAG (paper Alg. 2).

    ``B(v) = U(v) + alpha * sum_{(w,v) in E} B(w)`` evaluated in
    topological order. ``edges`` is an ``(m, 2)`` integer array of
    ``(predecessor, successor)`` pairs. With ``topo_order = None`` the
    order is computed by Kahn's algorithm (fail-closed on cycles).
    """
    e = np.asarray(edges, dtype=np.intp)
    if e.size == 0:
        e = np.zeros((0, 2), dtype=np.intp)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must be an (m, 2) integer array")
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be >= 1")
    if bool(np.any(e < 0)) or bool(np.any(e >= n)):
        raise ValueError("edge endpoints must be node ids in [0, n_nodes)")
    u = as_named_1d("node_uncertainty", np.asarray(node_uncertainty, dtype=float))
    if u.shape[0] != n:
        raise ValueError("node_uncertainty must have length n_nodes")
    if not bool(np.isfinite(u).all()):
        raise ValueError("node_uncertainty must be finite")
    al = float(alpha)
    if not math.isfinite(al) or al < 0.0:
        raise ValueError("alpha must be a non-negative attenuation factor")
    if topo_order is None:
        order = _topological_order(n, e)
    else:
        order = np.asarray(topo_order, dtype=np.intp).reshape(-1)
        if order.shape[0] != n or set(order.tolist()) != set(range(n)):
            raise ValueError("topo_order must be a permutation of the nodes")
    b = u.copy()
    preds: list[list[int]] = [[] for _ in range(n)]
    for p, c in e:
        preds[int(c)].append(int(p))
    for v in order:
        acc = float(b[v])
        for w in preds[int(v)]:
            acc += al * float(b[w])
        b[v] = acc
    return b


def graph_complexity(
    edges: object,
    node_uncertainty: object,
    alpha: float = DEFAULT_ALPHA,
    topo_order: IntArray | None = None,
) -> float:
    """``GC(G) = sum_v B(v)`` — the paper's graph complexity (Alg. 2)."""
    n = int(np.asarray(node_uncertainty, dtype=float).reshape(-1).shape[0])
    b = alpha_centrality(edges, n, node_uncertainty, alpha, topo_order)
    return float(b.sum())


def variance_bound_certificate(
    dag: ErrorDAG,
    alpha: float = DEFAULT_ALPHA,
    *,
    tol: float = 1e-9,
) -> dict[str, object]:
    """Certify the paper's bound: propagated variance <= graph complexity.

    Node uncertainty in *variance units*: ``U(v) := Var(e_v)`` propagated
    by ``propagate_uncertainty``. Then ``B(v_s) >= Var(e_s)`` for every
    step (induction on the topological order — ``alpha >= 0``, all
    ``B(w) >= U(w) >= 0``, so the edge term only adds non-negative
    ancestor mass) and ``GC(G) >= sum_s Var(e_s)``. The certificate is
    structural, not fitted; ``certified`` reports the numerical check.
    """
    pu = propagate_uncertainty(dag)
    var = pu.variance
    edges = dag.edges
    b = alpha_centrality(edges, dag.horizon, var, alpha, dag.topo_order())
    gc = float(b.sum())
    margins = b - var
    total_margin = gc - float(var.sum())
    certified = bool(np.all(margins >= -tol) and total_margin >= -tol)
    return {
        "schema": STUDY_SCHEMA,
        "per_step_variance": var,
        "per_step_centrality": b,
        "per_step_bound_margin": margins,
        "total_variance": float(var.sum()),
        "graph_complexity": gc,
        "total_bound_margin": total_margin,
        "alpha": float(alpha),
        "certified": certified,
    }


# ---------------------------------------------------------------------------
# 4. Sampled SGA pipeline: slice -> graph -> align -> complexity
# ---------------------------------------------------------------------------

#: Forecaster stub contract: ``f(history, horizon, n_samples, rng) ->
#: (n_samples, horizon)`` finite trajectory samples. All randomness flows
#: through ``rng`` so seeded harnesses are bit-identical.
ForecasterStub = Callable[[Array, int, int, np.random.Generator], Array]


def sample_trajectories(
    forecaster: ForecasterStub,
    history: object,
    horizon: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Array:
    """Draw ``n_samples`` multi-step forecast trajectories from a stub.

    Fail-closed: bad history/horizon/sample counts, non-Generator ``rng``,
    and stub outputs that are not finite ``(n_samples, horizon)`` arrays
    all raise ``ValueError``.
    """
    hist = as_named_1d("history", np.asarray(history, dtype=float))
    if hist.size < 1 or not bool(np.isfinite(hist).all()):
        raise ValueError("history must be a non-empty finite 1d series")
    h = int(horizon)
    k = int(n_samples)
    if h < 1 or k < MIN_SAMPLES:
        raise ValueError(f"need horizon >= 1 and n_samples >= {MIN_SAMPLES}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator (seeded)")
    out = np.asarray(forecaster(hist, h, k, rng), dtype=float)
    if out.shape != (k, h):
        raise ValueError(
            f"forecaster stub must return (n_samples, horizon) = {(k, h)}; got {out.shape}"
        )
    if not bool(np.isfinite(out).all()):
        raise ValueError("forecaster stub returned non-finite trajectories")
    return out


def slice_spans(horizon: int, slice_len: int) -> tuple[tuple[int, int], ...]:
    """Slice boundaries ``[(start, stop)]`` covering the horizon.

    ``n = ceil(h / l_s)``; the last slice takes the remainder when
    ``l_s`` does not divide ``h`` (paper Appendix A). ``l_s > h`` yields a
    single whole-horizon slice.
    """
    h = int(horizon)
    ls = int(slice_len)
    if h < 1 or ls < 1:
        raise ValueError("horizon and slice_len must be >= 1")
    n = int(math.ceil(h / ls))
    return tuple((j * ls, min((j + 1) * ls, h)) for j in range(n))


def dtw_distance(a: object, b: object) -> float:
    """Dynamic Time Warping distance with absolute local cost.

    ``d_DTW(a, b) = min over warping paths sum |a_i - b_j|`` (Sakoe &
    Chiba 1978) via the textbook O(len(a) * len(b)) DP. No existing repo
    implementation — provided here for the alignment stage.
    """
    x = as_named_1d("a", np.asarray(a, dtype=float))
    y = as_named_1d("b", np.asarray(b, dtype=float))
    if x.size == 0 or y.size == 0:
        raise ValueError("dtw inputs must be non-empty")
    if not bool(np.isfinite(x).all()) or not bool(np.isfinite(y).all()):
        raise ValueError("dtw inputs must be finite")
    n, m = x.size, y.size
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(n):
        cur = np.full(m + 1, np.inf)
        xi = x[i]
        for j in range(1, m + 1):
            cur[j] = abs(xi - y[j - 1]) + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m])


def seasonal_scaled_threshold(
    history: object,
    seasonality: int,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    r"""DTW merge threshold ``tau = lam/(t-S) * sum_i |x_i - x_{i+S}|``.

    Seasonal-naive scaling (Hyndman & Athanasopoulos 2018) roughly matches
    the merge threshold's scale to the series' MASE scale, so ``lam``
    transfers across series (paper default 0.25). Fail-closed: history
    length ``t <= S``, ``S < 1``, non-positive ``lam``, non-finite input.
    """
    hist = as_named_1d("history", np.asarray(history, dtype=float))
    t = hist.size
    s = int(seasonality)
    la = float(lam)
    if s < 1 or t <= s:
        raise ValueError(f"need seasonality >= 1 and history length > S (t={t}, S={s})")
    if not bool(np.isfinite(hist).all()):
        raise ValueError("history must be finite")
    if not math.isfinite(la) or la <= 0.0:
        raise ValueError("lam must be positive")
    return float(la * np.abs(hist[:-s] - hist[s:]).mean())


def step_level_entropy(trajectories: object) -> Array:
    r"""Per-step uncertainty ``U_s`` by Gaussian-KDE cross-entropy.

    Paper's trajectory-based branch: ``P_s`` is the Gaussian KDE over the
    ``K`` samples at step ``s`` with bandwidth
    ``eta_s = K^{-1/5} * sigma_s`` (the paper's Scott-rule form; the repo's
    ``scott_bandwidth`` carries an extra 1.059 constant and is not used),
    and ``U_s = -(1/K) * sum_k log P_s(x_k_s)`` — a plug-in cross-entropy
    estimator of the step distribution's differential entropy. Density
    evaluation reuses ``metrics.kde.gaussian_kde``.

    Fail-closed: fewer than ``MIN_SAMPLES`` trajectories, non-finite
    values, or a step with zero sample dispersion (point mass — entropy
    undefined, degenerate input per the honesty contract).
    """
    tr = _require_trajectories(trajectories)
    k, h = tr.shape
    ent = np.empty(h)
    for s in range(h):
        vals = tr[:, s]
        sd = float(vals.std(ddof=1))
        if sd <= 0.0:
            raise ValueError(f"zero-dispersion step distribution at step {s}")
        eta = (k ** (-1.0 / 5.0)) * sd
        dens = gaussian_kde(vals, vals, eta)
        ent[s] = float(-np.mean(np.log(dens)))
    return ent


def _require_trajectories(trajectories: object) -> Array:
    tr = np.asarray(trajectories, dtype=float)
    if tr.ndim != 2 or tr.shape[0] < MIN_SAMPLES or tr.shape[1] < 1:
        raise ValueError(
            f"trajectories must be a 2d (n_samples >= {MIN_SAMPLES}, horizon >= 1) array"
        )
    if not bool(np.isfinite(tr).all()):
        raise ValueError("trajectories must be finite")
    return tr


@dataclass(frozen=True)
class SgaGraph:
    """The aligned DAG over merged forecast-branch nodes.

    Node ``0`` is the root (the historical series, ``level = -1``); slice
    nodes carry ``level`` in ``[0, n_levels)`` and ``members`` — the
    trajectory indices merged into that branch. ``edges`` is an
    ``(m, 2)`` ``(predecessor, successor)`` integer array; node ids are
    level-major so id order is a topological order.
    """

    edges: IntArray
    node_uncertainty: Array
    levels: IntArray  # -1 = root, otherwise slice index j
    members: tuple[tuple[int, ...], ...]
    slice_spans: tuple[tuple[int, int], ...]
    n_levels: int
    n_samples: int

    @property
    def n_nodes(self) -> int:
        return int(self.node_uncertainty.shape[0])

    def nodes_at_level(self, level: int) -> IntArray:
        return cast(IntArray, np.flatnonzero(self.levels == level))


def build_sga_graph(
    trajectories: object,
    history: object | None = None,
    slice_len: int = DEFAULT_SLICE_LEN,
    tau: float | None = None,
    *,
    lam: float = DEFAULT_LAMBDA,
    seasonality: int = 1,
) -> SgaGraph:
    """Slicing -> graphing -> alignment on sampled trajectories (Alg. 1).

    ``tau`` may be given directly; otherwise it is derived from
    ``history`` by ``seasonal_scaled_threshold`` (``lam``, ``seasonality``).
    Passing neither raises ``ValueError`` (fail-closed). Alignment is
    greedy in creation order (trajectory index ascending) — deterministic;
    the paper leaves the iteration order unspecified.
    """
    tr = _require_trajectories(trajectories)
    k, h = tr.shape
    spans = slice_spans(h, int(slice_len))
    n_levels = len(spans)
    if tau is None:
        if history is None:
            raise ValueError("pass tau or history for the seasonal-scaled threshold")
        taus = seasonal_scaled_threshold(history, seasonality, lam)
    else:
        taus = float(tau)
        if not math.isfinite(taus) or taus < 0.0:
            raise ValueError("tau must be a non-negative finite threshold")
    ent = step_level_entropy(tr)
    # Node ids: 0 = root; 1 + j*K + k' = slice node of trajectory k' at level j.
    n_nodes = 1 + n_levels * k
    preds: list[set[int]] = [set() for _ in range(n_nodes)]
    succs: list[set[int]] = [set() for _ in range(n_nodes)]
    unc: list[list[float]] = [[] for _ in range(n_nodes)]

    def nid(kk: int, j: int) -> int:
        return int(1 + j * k + kk)

    for kk in range(k):
        prev = 0
        for j, (lo, hi) in enumerate(spans):
            v = nid(kk, j)
            preds[v].add(prev)
            succs[prev].add(v)
            unc[v].append(float(ent[lo:hi].mean()))
            prev = v
    alive = np.ones(n_nodes, dtype=bool)
    members: list[list[int]] = [[] for _ in range(n_nodes)]
    for v in range(1, n_nodes):
        members[v] = [(v - 1) % k]  # trajectory index of slice node v
    slices = [[tr[kk, lo:hi] for kk in range(k)] for (lo, hi) in spans]  # slices[j][kk]
    for j in range(n_levels):
        for vi in range(k):
            v = nid(vi, j)
            if not alive[v]:
                continue
            for wi in range(vi + 1, k):
                w = nid(wi, j)
                if not alive[w]:
                    continue
                if dtw_distance(slices[j][vi], slices[j][wi]) <= taus:
                    # merge w into v: union uncertainty, inherit edges.
                    unc[v].extend(unc[w])
                    members[v].extend(members[w])
                    for p in preds[w]:
                        if p != v:
                            preds[v].add(p)
                            succs[p].discard(w)
                            succs[p].add(v)
                    for c in succs[w]:
                        preds[c].discard(w)
                        preds[c].add(v)
                        succs[v].add(c)
                    preds[w].clear()
                    succs[w].clear()
                    alive[w] = False
    live_ids = [int(v) for v in range(n_nodes) if alive[v]]
    remap = {old: new for new, old in enumerate(live_ids)}
    new_edges: list[tuple[int, int]] = []
    for old in live_ids:
        for c in succs[old]:
            new_edges.append((remap[old], remap[c]))
    new_edges.sort()
    new_unc = np.array([float(np.mean(unc[v])) if unc[v] else 0.0 for v in live_ids])
    new_levels = np.array([-1 if v == 0 else (v - 1) // k for v in live_ids], dtype=np.intp)
    new_members = tuple(tuple(sorted(members[v])) for v in live_ids)
    return SgaGraph(
        edges=np.asarray(new_edges, dtype=np.intp).reshape(-1, 2),
        node_uncertainty=new_unc,
        levels=new_levels,
        members=new_members,
        slice_spans=spans,
        n_levels=n_levels,
        n_samples=k,
    )


def sga_complexity(graph: SgaGraph, alpha: float = DEFAULT_ALPHA) -> dict[str, object]:
    """Alpha-centrality and graph complexity of an aligned ``SgaGraph``."""
    b = alpha_centrality(graph.edges, graph.n_nodes, graph.node_uncertainty, alpha)
    return {
        "schema": STUDY_SCHEMA,
        "centrality": b,
        "graph_complexity": float(b.sum()),
        "alpha": float(alpha),
        "n_nodes": graph.n_nodes,
    }


def uq_from_residuals(
    residuals: object,
    max_lag: int = 1,
    alpha: float = DEFAULT_ALPHA,
    z: float = 1.96,
) -> dict[str, object]:
    """Interface (i): residual panel -> estimate DAG -> propagate -> bound."""
    fit = estimate_error_dag(residuals, max_lag)
    pu = propagate_uncertainty(fit.dag)
    cert = variance_bound_certificate(fit.dag, alpha)
    return {
        "schema": STUDY_SCHEMA,
        "dag": fit.dag,
        "bias": fit.bias,
        "r_squared": fit.r_squared,
        "propagated": pu,
        "half_widths": pu.interval_half_widths(z),
        "certificate": cert,
        "monotonicity": horizon_monotonicity(pu.variance),
    }


def uq_from_forecaster(
    forecaster: ForecasterStub,
    history: object,
    horizon: int,
    n_samples: int = 64,
    *,
    rng: np.random.Generator,
    slice_len: int = DEFAULT_SLICE_LEN,
    lam: float = DEFAULT_LAMBDA,
    seasonality: int = 1,
    alpha: float = DEFAULT_ALPHA,
    z: float = 1.96,
) -> dict[str, object]:
    """Interface (ii): forecaster stub -> sample -> SGA -> complexity.

    Per-horizon ``std`` is the trajectory dispersion at each step
    (empirical, non-graph); ``graph_complexity`` is the paper's
    uncertainty estimate for the whole trajectory; ``half_widths`` are
    ``z * std`` Gaussian interval half-widths.
    """
    tr = sample_trajectories(forecaster, history, horizon, n_samples, rng)
    graph = build_sga_graph(tr, history, slice_len, None, lam=lam, seasonality=seasonality)
    comp = sga_complexity(graph, alpha)
    std = tr.std(axis=0, ddof=1)
    return {
        "schema": STUDY_SCHEMA,
        "trajectories": tr,
        "graph": graph,
        "step_entropy": step_level_entropy(tr),
        "per_step_std": std,
        "half_widths": float(z) * std,
        "centrality": comp["centrality"],
        "graph_complexity": comp["graph_complexity"],
        "monotonicity": horizon_monotonicity(std**2),
    }


# ---------------------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------------------


def horizon_monotonicity(variance: object) -> dict[str, object]:
    """Non-decreasing-variance diagnostic over the horizon.

    Error accumulation should make per-step variance non-decreasing for
    well-posed error DAGs (e.g. stable AR(1) approaches its stationary
    variance from below). Returns the monotone flag, the first violating
    step (``-1`` if none), the increment vector, and terminal variance.
    """
    v = as_named_1d("variance", np.asarray(variance, dtype=float))
    if v.size < 1 or not bool(np.isfinite(v).all()):
        raise ValueError("variance must be a non-empty finite 1d array")
    if bool(np.any(v <= 0.0)):
        raise ValueError("variance must be strictly positive")
    inc = np.diff(v)
    bad = np.nonzero(inc < -1e-12)[0]
    return {
        "monotone": bool(bad.size == 0),
        "first_decrease": int(bad[0] + 1) if bad.size else -1,
        "increments": inc,
        "terminal_variance": float(v[-1]),
    }


def interval_coverage(errors: object, std: object, z: float) -> dict[str, object]:
    """Empirical ``|e_s| <= z * std_s`` coverage per horizon step."""
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2 or e.shape[0] < 2 or e.shape[1] < 1:
        raise ValueError("errors must be a 2d (n, horizon) panel with n >= 2")
    if not bool(np.isfinite(e).all()):
        raise ValueError("errors must be finite")
    s = as_named_1d("std", np.asarray(std, dtype=float))
    if s.shape[0] != e.shape[1] or bool(np.any(s <= 0.0)) or not bool(np.isfinite(s).all()):
        raise ValueError("std must be a positive finite array matching the horizon")
    zz = float(z)
    if not math.isfinite(zz) or zz <= 0.0:
        raise ValueError("z must be a positive finite multiplier")
    cov = (np.abs(e) <= zz * s[None, :]).mean(axis=0)
    return {"per_step": cov, "mean": float(cov.mean()), "z": zz}


def complexity_scale_diagnostic(model_sizes: object, complexities: object) -> dict[str, object]:
    """Scaling-law diagnostic: model size vs graph complexity.

    The paper's empirical scaling law — larger TSFM scales correlate with
    lower multi-step uncertainty — reported as Pearson on
    ``(log size, GC)`` and Spearman rank correlation, plus the monotone-
    decreasing flag. Diagnostic only; ``n >= 3`` required.
    """
    sz = as_named_1d("model_sizes", np.asarray(model_sizes, dtype=float))
    gc = as_named_1d("complexities", np.asarray(complexities, dtype=float))
    if sz.shape != gc.shape or sz.size < 3:
        raise ValueError("model_sizes and complexities need >= 3 matched entries")
    if not bool(np.isfinite(sz).all()) or not bool(np.isfinite(gc).all()):
        raise ValueError("inputs must be finite")
    if bool(np.any(sz <= 0.0)):
        raise ValueError("model_sizes must be positive")
    lx = np.log(sz)
    if float(lx.std()) == 0.0 or float(gc.std()) == 0.0:
        raise ValueError("degenerate (constant) inputs — correlation undefined")
    pear = float(np.corrcoef(lx, gc)[0, 1])
    spear = float(spearmanr(sz, gc).statistic)
    return {
        "pearson_log": pear,
        "spearman": spear,
        "decreasing": bool(np.all(np.diff(gc) <= 0.0)),
        "n": int(sz.size),
    }


# ---------------------------------------------------------------------------
# 6. Seeded SYNTHETIC harnesses (correctness scaffolding, not evidence)
# ---------------------------------------------------------------------------


def simulate_synthetic_dag_errors(
    dag: ErrorDAG,
    n_panels: int,
    rng: np.random.Generator,
) -> Array:
    """Simulate error panels from a DAG's linear-Gaussian SEM.

    SYNTHETIC: forward recursion ``e_s = sum_r A[r,s] e_r + eps_s`` with
    Gaussian innovations, in topological order. Correctness scaffolding
    for coverage/envelope tests — never market evidence.
    """
    if not isinstance(dag, ErrorDAG):
        raise ValueError("dag must be an ErrorDAG")
    n = int(n_panels)
    if n < 1:
        raise ValueError("n_panels must be >= 1")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator (seeded)")
    a = dag.coefficients
    h = dag.horizon
    out = np.zeros((n, h))
    for s in range(h):
        innov = rng.normal(0.0, math.sqrt(float(dag.innovation_var[s])), n)
        out[:, s] = (out[:, :s] @ a[:s, s] if s > 0 else 0.0) + innov
    return out


def make_synthetic_ar1_forecaster(phi: float, sigma: float) -> ForecasterStub:
    """Build a seeded SYNTHETIC AR(1) trajectory forecaster stub.

    Returns a ``ForecasterStub`` emitting ``(n_samples, horizon)``
    Gaussian AR(1) trajectories continued from ``history[-1]`` —
    correctness scaffolding, never market evidence.
    """
    p = float(phi)
    s = float(sigma)
    if not math.isfinite(p) or not math.isfinite(s) or s <= 0.0:
        raise ValueError("phi must be finite and sigma positive")

    def _stub(history: Array, horizon: int, n_samples: int, rng: np.random.Generator) -> Array:
        out = np.empty((n_samples, horizon))
        x = np.full(n_samples, float(history[-1]))
        for t in range(horizon):
            x = p * x + rng.normal(0.0, s, n_samples)
            out[:, t] = x
        return out

    return _stub
