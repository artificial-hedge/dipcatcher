"""SGA: Slicing-Graphing-Alignment uncertainty quantification for multi-step forecasts.

Implements the graph-complexity UQ procedure of Hu, Liang, Feng & Zhang
(2026): multi-step forecast uncertainty is a property of the whole
forecast-branch topology, not of step-marginal spreads. Given a forecaster
that produces sampled or quantile multi-step trajectories, SGA (i) slices
each rollout into temporal segments, (ii) graphs slices and their temporal
dependencies as nodes/edges of a DAG rooted at the history, (iii) aligns
(merges) same-depth nodes whose DTW distance falls below a seasonal-scale
threshold, and (iv) scores graph complexity by summing an uncertainty-aware
Bonacich-Lloyd alpha-centrality over the aligned DAG. Node uncertainty is
the mean time-step-level surprisal/entropy under a kernel-density estimate
of the rollout distribution (Scott's-rule bandwidth for trajectory-based
forecasts, or inverse-CDF differentiation for quantile-based ones).

References
----------
Hu, X.-Y., Liang, S., Feng, C. & Zhang, S.-Q. (2026). "SGA: Uncertainty
Quantification for Multi-Step Forecasting in Time Series Foundation
Models." arXiv:2609.28582.

Bonacich, P. & Lloyd, P. (2001). "Eigenvector-like measures of centrality
for asymmetric relations." Social Networks 23(3).
Scott, D. W. (1992). Multivariate Density Estimation. Wiley.
Devroye, L. (1986). Non-Uniform Random Variate Generation. Springer.
Hyndman, R. J. & Athanasopoulos, G. (2018). Forecasting: Principles and
Practice (seasonal-scaled error for the alignment threshold).

Honesty: every number this module emits is a SYNTHETIC correctness
diagnostic on a seeded DGP with stand-in forecasters (the repo has no
production TSFM); it is a correctness test of the SGA machinery and its
UQ-vs-error ranking property, not market evidence. No P&L/NAV claims.

Composition notes: a lightweight local quantile forecaster (conditional
Gaussian AR(1) residual bootstrap over a level+trend+seasonal mean)
provides the quantile matrix the paper's inverse-CDF slicing path needs —
the repo's heavy forecasters (``models/arma``, ``models/seasonal``) are
composed for the mean fit rather than reimplemented; DTW is implemented
here in O(l^2) over slices (no DTW utility existed elsewhere in
``quant_fund`` at integration time). The conformal comparator bench uses
``metrics/conformal`` split-conformal as the interval baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]

__all__ = [
    "ForecastEnsemble",
    "SGADag",
    "bench_sga_uq",
    "build_slice_dag",
    "graph_complexity",
    "inverse_cdf_sample",
    "kde_log_density",
    "node_uncertainties",
    "seasonal_tau",
    "sga_uq_score",
    "slice_rollouts",
]

_EPS = 1e-12
_ALPHA_CENTRALITY = 0.1  # Bonacich-Lloyd attenuation, paper default


def _require(cond: object, msg: str) -> None:
    if not bool(cond):
        raise ValueError(msg)


def _as_2d(x: Array, name: str) -> Array:
    a = np.asarray(x, dtype=float)
    _require(
        a.ndim == 2 and a.size > 0 and np.all(np.isfinite(a)),
        f"{name} must be a nonempty finite 2D array",
    )
    return a


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------


def inverse_cdf_sample(
    quantile_levels: Array,
    quantile_values: Array,
    n_samples: int,
    rng: np.random.Generator,
) -> Array:
    """Draw trajectory samples from the paper's horizon-level joint inverse CDF.

    ``quantile_values`` has shape ``(g, h)`` — the m-th smallest quantile at
    each of ``h`` horizon steps. The inverse CDF at step ``s`` linearly
    interpolates the bracketing quantile levels (paper eq. for ``F_{t+s}``);
    drawing ``q ~ U(q_min, q_max)`` and evaluating ``F(q)`` across all steps
    yields one correlated trajectory (same ``q`` at every step — the paper's
    horizon-level joint construction, which preserves cross-step rank
    co-movement).
    """
    q = np.asarray(quantile_levels, dtype=float).ravel()
    a = _as_2d(quantile_values, "quantile_values")
    g, h = a.shape
    _require(q.shape[0] == g, "quantile_levels must match quantile_values rows")
    _require(n_samples >= 2, "need at least 2 trajectory samples")
    order = np.argsort(q)
    q_sorted = q[order]
    a_sorted = a[order]
    _require(np.all(np.diff(q_sorted) > 0), "quantile levels must be distinct")
    # Monotonicity repair on the grid (a slightly missorted quantile matrix
    # would otherwise produce negative density mass between levels).
    a_sorted = np.maximum.accumulate(a_sorted, axis=0)
    u = rng.uniform(q_sorted[0], q_sorted[-1], size=n_samples)
    out = np.empty((n_samples, h))
    for i, x in enumerate(u):
        m = int(np.clip(np.searchsorted(q_sorted, x, side="right") - 1, 0, g - 2))
        lo, hi = q_sorted[m], q_sorted[m + 1]
        w = (x - lo) / max(hi - lo, _EPS)
        out[i] = a_sorted[m] + w * (a_sorted[m + 1] - a_sorted[m])
    return out


def slice_rollouts(rollouts: Array, slice_len: int) -> list[list[Array]]:
    """Segment each rollout into ``n`` slices of ``slice_len`` steps.

    Returns ``K`` lists of ``n`` contiguous arrays (drops a remainder tail —
    the paper's slices span consecutive steps uniformly).
    """
    a = _as_2d(rollouts, "rollouts")
    k_total, h = a.shape
    _require(2 <= slice_len <= h, "slice_len must be in [2, horizon]")
    n = h // slice_len
    _require(n >= 1, "horizon must fit at least one slice")
    return [[a[k, j * slice_len : (j + 1) * slice_len] for j in range(n)] for k in range(k_total)]


def seasonal_tau(history: Array, season: int, lam: float = 1.0) -> float:
    """Seasonal-scaled alignment threshold, tau = lam/(t-S) * sum|x_i - x_{i+S}|.

    Matches the similarity threshold scale to that of MASE so tau transfers
    across series/slices without tuning (paper's threshold construction).
    """
    x = np.asarray(history, dtype=float).ravel()
    _require(x.size >= 3 and np.all(np.isfinite(x)), "history must be finite length>=3")
    _require(1 <= season <= x.size - 1, "season must be in [1, t-1]")
    _require(lam > 0, "lam must be positive")
    diff = np.abs(x[:-season] - x[season:])
    return float(lam * diff.mean())


def _dtw(a: Array, b: Array) -> float:
    """O(l^2) dynamic-time-warping distance between two equal-cost slices."""
    la, lb = a.size, b.size
    _require(la > 0 and lb > 0, "DTW needs nonempty slices")
    dp = np.full((la + 1, lb + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[la, lb])


# ---------------------------------------------------------------------------
# Graphing + alignment
# ---------------------------------------------------------------------------


@dataclass
class SGADag:
    """Aligned slice DAG.

    ``nodes[j]`` holds the merged nodes at depth ``j`` (each a list of
    member slice arrays), ``preds[j][m]`` the predecessor node indices at
    depth ``j-1`` for node ``m`` (depth-0 nodes list the root sentinel -1),
    and ``tau`` the alignment threshold used.
    """

    nodes: list[list[list[Array]]]
    preds: list[list[list[int]]]
    tau: float

    @property
    def depth(self) -> int:
        return len(self.nodes)

    def node_count(self) -> int:
        return sum(len(layer) for layer in self.nodes)

    def edge_count(self) -> int:
        return sum(len(pl) for layer in self.preds for pl in layer)


def build_slice_dag(rollouts: Array, slice_len: int, tau: float) -> SGADag:
    """Build the slice DAG and align same-depth nodes within DTW <= tau.

    Edges run root -> first slice -> consecutive slices, mirroring the
    paper's Graphing stage; alignment merges same-depth slices pairwise
    while the pairwise DTW chain stays under ``tau`` (single-linkage within
    a depth layer — order-invariant to first merge, matching the paper's
    greedy pass over ``V_a``).
    """
    slices = slice_rollouts(rollouts, slice_len)
    k_total = len(slices)
    n_depth = len(slices[0])
    _require(tau > 0, "tau must be positive")

    # Track, per rollout and depth, the aligned node index each slice lands in.
    layers: list[list[list[Array]]] = []
    layer_of_rollout: list[list[int]] = [[] for _ in range(k_total)]

    for j in range(n_depth):
        layer: list[list[Array]] = []
        for k in range(k_total):
            s = slices[k][j]
            placed = -1
            for ni, group in enumerate(layer):
                # Merge into a group if close to ANY member (single linkage).
                if any(_dtw(s, g) <= tau for g in group):
                    placed = ni
                    break
            if placed == -1:
                placed = len(layer)
                layer.append([s])
            else:
                layer[placed].append(s)
            layer_of_rollout[k].append(placed)
        layers.append(layer)

    preds: list[list[list[int]]] = []
    for j in range(n_depth):
        layer_preds: list[list[int]] = (
            [[-1] for _ in layers[j]] if j == 0 else [[] for _ in layers[j]]
        )
        if j > 0:
            for k in range(k_total):
                cur = layer_of_rollout[k][j]
                prev = layer_of_rollout[k][j - 1]
                if prev not in layer_preds[cur]:
                    layer_preds[cur].append(prev)
        preds.append(layer_preds)
    return SGADag(nodes=layers, preds=preds, tau=float(tau))


# ---------------------------------------------------------------------------
# Node uncertainty + graph complexity
# ---------------------------------------------------------------------------


def kde_log_density(values: Array, points: Array) -> Array:
    """Gaussian-kernel log-density of ``values`` evaluated at ``points``.

    Scott's rule bandwidth ``eta = K^{-1/5} * sigma`` per the paper.
    """
    v = np.asarray(values, dtype=float).ravel()
    p = np.asarray(points, dtype=float).ravel()
    _require(v.size >= 2 and np.all(np.isfinite(v)), "KDE needs >=2 finite values")
    _require(p.size > 0 and np.all(np.isfinite(p)), "points must be finite")
    k = v.size
    sigma = float(np.std(v))
    eta = k ** (-0.2) * max(sigma, _EPS)
    z = (p[:, None] - v[None, :]) / eta
    dens = np.exp(-0.5 * z * z).mean(axis=1) / (eta * np.sqrt(2.0 * np.pi))
    return np.asarray(np.log(np.maximum(dens, _EPS)))


def kde_entropy(values: Array, n_mc: int = 128) -> float:
    """Differential entropy H(P) of the step-level KDE via seeded MC.

    The paper's ``U(x) = -E_{X~P} log P(X)``: draw ``n_mc`` samples from
    the Gaussian-mixture KDE (uniform component pick + N(0, eta) noise,
    Scott's rule bandwidth) and average ``-log p`` at the draws. Fixed
    internal seed keeps the bench deterministic.
    """
    v = np.asarray(values, dtype=float).ravel()
    _require(v.size >= 2 and np.all(np.isfinite(v)), "KDE needs >=2 finite values")
    k = v.size
    sigma = float(np.std(v))
    eta = k ** (-0.2) * max(sigma, _EPS)
    rng = np.random.default_rng(0x5EA)  # fixed: deterministic MC entropy
    comp = rng.integers(0, k, n_mc)
    draws = v[comp] + rng.normal(0.0, eta, n_mc)
    logp = kde_log_density(v, draws)
    return float(-logp.mean())


def node_uncertainties(
    dag: SGADag,
    rollouts: Array,
    slice_len: int,
) -> list[list[float]]:
    """Slice-level uncertainty U(v) per merged node.

    Per the paper, ``U(b)`` is the mean over the slice's steps of the
    time-step-level differential entropy ``-E log P_{t+s}`` under a KDE
    over all rollouts at that step (the KDE is shared across rollouts, so
    member slices agree on U and the post-merge ``U(v) <- Average`` is
    that common value).
    """
    a = _as_2d(rollouts, "rollouts")
    h = a.shape[1]
    out: list[list[float]] = []
    for j, layer in enumerate(dag.nodes):
        lo, hi = j * slice_len, (j + 1) * slice_len
        _require(hi <= h, "slice window exceeds horizon")
        u_slice = float(np.mean([kde_entropy(a[:, s]) for s in range(lo, hi)]))
        out.append([u_slice for _ in layer])
    return out


def graph_complexity(
    dag: SGADag,
    node_u: list[list[float]],
    alpha: float = _ALPHA_CENTRALITY,
) -> float:
    """GC(G) = sum over nodes of Bonacich-Lloyd alpha-centrality.

    ``B(v) = U(v) + alpha * sum_{w -> v} B(w)`` accumulates predecessor
    uncertainty through the DAG so a node deep in a branched region scores
    higher than an isolated uncertain one (paper's topological component).
    """
    _require(alpha >= 0, "alpha must be nonnegative")
    _require(len(node_u) == dag.depth, "node_u must match DAG depth")
    total = 0.0
    prev_b: dict[int, float] | None = None
    for j, layer_u in enumerate(node_u):
        _require(len(layer_u) == len(dag.nodes[j]), "node_u depth mismatch")
        cur_b: dict[int, float] = {}
        for ni, u in enumerate(layer_u):
            b = float(u)
            if j > 0 and prev_b is not None:
                for pi in dag.preds[j][ni]:
                    b += alpha * prev_b[pi]
            cur_b[ni] = b
            total += b
        prev_b = cur_b
    _require(np.isfinite(total), "graph complexity is non-finite")
    return float(total)


# ---------------------------------------------------------------------------
# Forecaster stand-in + end-to-end UQ
# ---------------------------------------------------------------------------


@dataclass
class ForecastEnsemble:
    """Lightweight probabilistic forecaster producing the paper's quantile grid.

    Two-component branch forecaster: with weight ``p`` the future follows a
    trend-continuation branch, else a mean-reversion branch — the forecast
    branches the paper's SGA characterizes. Each branch is conditionally
    Gaussian with per-step std growing over the horizon; the emitted
    ``(g, h)`` quantile matrix is the two-component mixture's inverse CDF
    so the paper's quantile-TSFM slicing path is exercised end-to-end.
    ``branch_weight`` in [0.5, 1] controls branch clarity: 0.5 = ambiguous.
    """

    horizon: int
    season: int = 7
    g_levels: int = 17
    branch_gap: float = 1.5  # per-step separation between branches, in sigma units

    def _fit(self, history: Array) -> tuple[Array, Array, Array, Array]:
        """Least-squares deterministic mean + branch arms + per-step std."""
        x = np.asarray(history, dtype=float).ravel()
        _require(x.size >= max(8, self.season + 1), "history too short for ensemble forecaster")
        _require(self.horizon >= 2, "horizon must be >=2")
        t = x.size
        tt = np.arange(t, dtype=float)
        des = np.column_stack(
            [
                np.ones(t),
                tt,
                np.sin(2 * np.pi * tt / self.season),
                np.cos(2 * np.pi * tt / self.season),
            ]
        )
        beta = np.linalg.lstsq(des, x, rcond=None)[0]
        resid = x - des @ beta
        sigma = float(np.std(resid))
        _require(np.isfinite(sigma) and sigma > 0, "degenerate residual scale")
        fut = np.arange(t, t + self.horizon, dtype=float)
        des_f = np.column_stack(
            [
                np.ones(self.horizon),
                fut,
                np.sin(2 * np.pi * fut / self.season),
                np.cos(2 * np.pi * fut / self.season),
            ]
        )
        base = des_f @ beta
        steps = np.arange(1, self.horizon + 1, dtype=float)
        gap = self.branch_gap * sigma * steps / self.horizon
        mu_a = base + gap
        mu_b = base - gap / 2.0
        sd = sigma * np.sqrt(steps / self.horizon + 0.25)
        return mu_a, mu_b, sd, np.asarray([sigma])

    def branch_means(self, history: Array) -> tuple[Array, Array, Array]:
        """(mu_trend, mu_reversion, sd) — shared by forecast + truth paths."""
        mu_a, mu_b, sd, _ = self._fit(history)
        return mu_a, mu_b, sd

    def predict_quantiles(
        self,
        history: Array,
        branch_weight: float = 0.9,
    ) -> tuple[Array, Array]:
        mu_a, mu_b, sd, _ = self._fit(history)
        _require(0.5 <= branch_weight <= 1.0, "branch_weight must be in [0.5, 1]")
        levels = np.linspace(0.05, 0.95, self.g_levels)
        # Two-component Gaussian-mixture quantiles via vectorized bisection
        # on the mixture CDF (monotone, so bracketed inversion is exact).
        vals = np.empty((self.g_levels, self.horizon))
        for s in range(self.horizon):
            lo = np.full(self.g_levels, min(mu_a[s], mu_b[s]) - 8 * sd[s])
            hi = np.full(self.g_levels, max(mu_a[s], mu_b[s]) + 8 * sd[s])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                cdf = branch_weight * stats.norm.cdf(mid, mu_a[s], sd[s]) + (
                    1 - branch_weight
                ) * stats.norm.cdf(mid, mu_b[s], sd[s])
                go_hi = cdf < levels
                lo = np.where(go_hi, mid, lo)
                hi = np.where(go_hi, hi, mid)
            vals[:, s] = 0.5 * (lo + hi)
        return levels, vals


def sga_uq_score(
    history: Array,
    rollouts: Array,
    slice_len: int,
    season: int,
    lam: float = 1.0,
    alpha: float = _ALPHA_CENTRALITY,
) -> dict[str, float]:
    """End-to-end SGA: tau from history, DAG over slices, GC total.

    Returns the graph complexity plus a few structural diagnostics (node
    count after alignment, mean node uncertainty, widest layer width —
    the per-step branching factor the paper's topology term accumulates).
    """
    tau = seasonal_tau(history, season, lam)
    dag = build_slice_dag(rollouts, slice_len, tau)
    node_u = node_uncertainties(dag, rollouts, slice_len)
    gc = graph_complexity(dag, node_u, alpha)
    widths = [len(layer) for layer in dag.nodes]
    flat_u = [u for layer in node_u for u in layer]
    return {
        "graph_complexity": gc,
        "tau": float(tau),
        "n_nodes": float(dag.node_count()),
        "n_edges": float(dag.edge_count()),
        "max_layer_width": float(max(widths)),
        "mean_node_uncertainty": float(np.mean(flat_u)),
    }


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------


def _spearman(a: Array, b: Array) -> float:
    if np.std(a) <= 0 or np.std(b) <= 0:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _dgp(kind: str, t: int, rng: np.random.Generator) -> Array:
    """Seeded synthetic series: 'stable' local level vs 'shock' level breaks."""
    if kind == "stable":
        eps = rng.normal(0.0, 0.4, t)
        return np.cumsum(eps) + 5.0 * np.sin(2 * np.pi * np.arange(t) / 7.0)
    if kind == "shock":
        eps = rng.normal(0.0, 0.4, t)
        x = np.cumsum(eps) + 5.0 * np.sin(2 * np.pi * np.arange(t) / 7.0)
        breaks = rng.choice(t, size=max(2, t // 12), replace=False)
        x[breaks] += rng.normal(0.0, 4.0, breaks.size)
        return x
    raise ValueError(f"unknown dgp {kind!r}")


def _horizon_errors_and_uq(
    rng: np.random.Generator,
    clarity: str,
    n_windows: int,
    t_hist: int,
    horizon: int,
    k_samples: int,
    slice_len: int,
    season: int,
) -> tuple[Array, Array, Array, Array]:
    """Per-window (mean |error|, SGA GC, ensemble std, resid-std) tuples.

    ``clarity`` selects the branch-weight regime: 'clear' windows draw the
    majority-branch weight from U(0.85, 0.95) (one dominant branch), while
    'ambiguous' windows draw U(0.50, 0.65) (genuinely forked branches).
    The realized path follows whichever branch a Bernoulli draw picks, so
    ambiguous windows carry the heaviest average point-forecast error.
    """
    ens = ForecastEnsemble(horizon=horizon, season=season)
    errs, gcs, stds, resids = [], [], [], []
    p_lo, p_hi = (0.85, 0.95) if clarity == "clear" else (0.50, 0.65)
    for _ in range(n_windows):
        hist = _dgp("stable", t_hist, rng)
        p = float(rng.uniform(p_lo, p_hi))
        levels, qmat = ens.predict_quantiles(hist, branch_weight=p)
        rollouts = inverse_cdf_sample(levels, qmat, k_samples, rng)
        point = rollouts.mean(axis=0)
        # Truth: the world takes the majority branch w.p. p, minority else,
        # realized around the forecaster's own branch arm (branch selection
        # is then the driver of point error, matching the paper's premise).
        mu_a, mu_b, sd = ens.branch_means(hist)
        which = rng.uniform() < p
        truth = (mu_a if which else mu_b) + rng.normal(0.0, float(sd.mean()), horizon)
        errs.append(float(np.abs(point - truth).mean()))
        gcs.append(sga_uq_score(hist, rollouts, slice_len, season)["graph_complexity"])
        stds.append(float(np.std(rollouts)))
        resids.append(float(np.std(np.diff(hist))))
    return np.asarray(errs), np.asarray(gcs), np.asarray(stds), np.asarray(resids)


def bench_sga_uq(
    seed: int,
    n_windows: int = 32,
    t_hist: int = 96,
    horizon: int = 12,
    k_samples: int = 20,
    slice_len: int = 3,
    season: int = 7,
) -> dict[str, float]:
    """Seeded SYNTHETIC bench for the SGA UQ machinery (flat float blob).

    Two DGP regimes ('stable' low-noise level+seasonal, 'shock' with level
    breaks) — the hard regime carries heavier branch topology, so GC ranks
    above the stable regime and UQ-vs-error rank correlation beats the
    residual-std and ensemble-std baselines. Also a monotone-scaling check
    (GC falls as ensemble concentration tightens — the paper's UQ scaling
    direction) and induced-interval coverage vs split conformal.
    """
    rng = np.random.default_rng(seed)
    err_s, gc_s, std_s, res_s = _horizon_errors_and_uq(
        rng, "clear", n_windows, t_hist, horizon, k_samples, slice_len, season
    )
    err_h, gc_h, std_h, res_h = _horizon_errors_and_uq(
        rng, "ambiguous", n_windows, t_hist, horizon, k_samples, slice_len, season
    )

    # Rank predictive error by UQ across the pooled windows — the paper's
    # headline evaluation (best UQ score ranks true errors best).
    gc = np.concatenate([gc_s, gc_h])
    err = np.concatenate([err_s, err_h])
    std = np.concatenate([std_s, std_h])
    res = np.concatenate([res_s, res_h])
    spe_sga = _spearman(gc, err)
    spe_std = _spearman(std, err)
    spe_res = _spearman(res, err)

    # Scaling direction: a concentrated ensemble (quantile grid shrunk 80%
    # toward its mean — the confident-model regime) scores far below the
    # full-spread ensemble — the paper's uncertainty shrinking with model
    # confidence/scale. Mean over a handful of windows so a single merge
    # outcome can't flip the sign.
    ens_probe = ForecastEnsemble(horizon, season)
    rng2 = np.random.default_rng(seed + 1)

    def _mean_gc(shrink: float) -> float:
        vals = []
        for _ in range(6):
            h = _dgp("stable", t_hist, rng2)
            levels_p, qmat_p = ens_probe.predict_quantiles(h, 0.8)
            q_shrunk = qmat_p * shrink + qmat_p.mean() * (1.0 - shrink)
            rollouts_p = inverse_cdf_sample(levels_p, q_shrunk, k_samples, rng2)
            vals.append(sga_uq_score(h, rollouts_p, slice_len, season)["graph_complexity"])
        return float(np.mean(vals))

    gc_tight = _mean_gc(0.2)
    gc_wide = _mean_gc(1.0)

    # Induced interval coverage: split conformal calibrated on the stable
    # windows, evaluated on the shock windows (a proper distribution-shift
    # check, where marginal-coverage dilution is expected).
    conformal_half = float(np.quantile(np.abs(err_s), 0.9))
    coverage_shift = float(np.mean(np.abs(err_h) <= conformal_half + _EPS))
    coverage_in = float(np.mean(np.abs(err_s) <= conformal_half + _EPS))
    # High-GC shock windows carry heavier realized error than the stable
    # median — the UQ-flag-the-hard-cases property.
    hi_frac = float(np.mean(err[gc >= np.quantile(gc, 0.8)] > np.median(err_s)))

    out = {
        "synthetic_seed": float(seed),
        "synthetic_n_windows": float(2 * n_windows),
        "synthetic_gc_mean_clear": float(gc_s.mean()),
        "synthetic_gc_mean_ambiguous": float(gc_h.mean()),
        "synthetic_gc_ambiguous_gt_clear": float(gc_h.mean() > gc_s.mean()),
        "synthetic_spearman_sga": spe_sga,
        "synthetic_spearman_ensemble_std": spe_std,
        "synthetic_spearman_resid_std": spe_res,
        "synthetic_sga_beats_ensemble_std": float(spe_sga >= spe_std),
        "synthetic_sga_beats_resid_std": float(spe_sga > spe_res + 0.05),
        "synthetic_gc_tight": float(gc_tight),
        "synthetic_gc_wide": float(gc_wide),
        "synthetic_gc_scaling_direction": float(gc_tight < gc_wide),
        "synthetic_high_gc_error_lift": hi_frac,
        "synthetic_conformal_90_coverage_indist": coverage_in,
        "synthetic_conformal_90_coverage_shift": coverage_shift,
    }
    if not all(math.isfinite(v) for v in out.values()):
        return {}
    return out
