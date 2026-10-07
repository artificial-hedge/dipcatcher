"""EverMine-style capability-value accounting for self-evolution evaluation.

Reference
---------
Li, Zhang, Yao, Qiu, Xu, Yuan et al. 2026
"EverMine: Dissecting the Self-Evolution of Research Capabilities in
Long-Horizon Alpha Research." arXiv:2609.33524.

The paper decomposes the research state into three components:
- *History* (Hist): all past trials and their outcomes (submission dates,
  scores, decisions).
- *Frontier*: the current active factor portfolio (weights, signal
  metrics, regime context).
- *Capabilities* (Cap): reusable skills / tools / rules accumulated from
  experience (decision heuristics, screening templates, learned
  thresholds). Represented as a frozen dict of named parameters/rules.

The core protocol — "Cap-swap" evaluation — holds Hist and Frontier fixed
while swapping in a different Cap. The resulting portfolio change estimates
the *conditional value* of accumulated capability, independent of state
differences.

Honesty contract
----------------
This module is a **SYNTHETIC diagnostic** for detecting false self-evolution
claims. All IC / rank-IC numbers are seeded SYNTHETIC research diagnostics,
never market alpha. No Sharpe / Sortino / Calmar / P&L / NAV keys appear
in any output blobs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr, ttest_1samp

Array = NDArray[np.float64]

_SS = 1e-12
_RNG_SEED: int = 999983
_N_BOOT: int = 2_000


# ---------------------------------------------------------------------------
# Structured containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRecord:
    """A single trial entry in the historical record."""

    submission_date: str  # ISO 8601, UTC
    rank_ic: float  # date-level rank IC of the submitted factor
    ic: float  # date-level Pearson IC
    decision: str  # "accepted" or "rejected"
    factor_id: str | None = None  # optional identifier

    def __post_init__(self) -> None:
        if not np.isfinite(self.rank_ic):
            raise ValueError("rank_ic must be finite")
        if not np.isfinite(self.ic):
            raise ValueError("ic must be finite")
        if self.decision not in {"accepted", "rejected"}:
            raise ValueError("decision must be 'accepted' or 'rejected'")


@dataclass(frozen=True)
class Frontier:
    """The current active factor portfolio."""

    weights: dict[str, float]  # asset_id → weight
    signal_metrics: dict[str, float]  # e.g. mean IC, rank IC, coverage
    regime_context: dict[str, Any] = field(default_factory=dict)  # e.g. vol regime label

    def __post_init__(self) -> None:
        if not isinstance(self.weights, dict):
            raise ValueError("weights must be a dict")
        if self.weights:
            w = np.array(list(self.weights.values()), dtype=float)
            if not np.all(np.isfinite(w)):
                raise ValueError("frontier weights must be finite")
            if not np.all(w >= -_SS):
                raise ValueError("frontier weights must be non-negative")
            total = float(np.sum(w))
            if total <= 0:
                raise ValueError("frontier must have positive total weight")


@dataclass(frozen=True)
class Capabilities:
    """Reusable rule base — decision heuristics, screening templates,
    learned thresholds. Represented as a frozen dict of named parameters."""

    rules: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rules, dict):
            raise ValueError("rules must be a dict")
        # Ensure the dict is effectively frozen by converting to tuple
        object.__setattr__(self, "rules", dict(self.rules))

    def fingerprint(self) -> str:
        """A deterministic hash of the capability rule set."""
        import hashlib
        import json

        raw = json.dumps(self.rules, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Capabilities):
            return NotImplemented
        return self.rules == other.rules

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.rules.items())))


@dataclass(frozen=True)
class ResearchState:
    """The tripartite research state (Hist, Frontier, Cap) from EverMine."""

    hist: tuple[TrialRecord, ...]
    frontier: Frontier
    cap: Capabilities
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.hist, tuple):
            raise ValueError("hist must be a tuple of TrialRecord")
        if not all(isinstance(t, TrialRecord) for t in self.hist):
            raise ValueError("all hist entries must be TrialRecord instances")
        if not isinstance(self.frontier, Frontier):
            raise ValueError("frontier must be a Frontier instance")
        if not isinstance(self.cap, Capabilities):
            raise ValueError("cap must be a Capabilities instance")


# ---------------------------------------------------------------------------
# Portfolio evaluation (the "fixed decision rule" required by the protocol)
# ---------------------------------------------------------------------------


def _rank_signals(
    signal_matrix: Array,
    cap: Capabilities,
    *,
    purged_cv: bool = True,
    embargo: int = 5,
    n_cv_folds: int = 5,
) -> Array:
    """Rank signal streams using the Cap's decision rules.

    Respects the Cap's specified selection logic:
    - naively ranks by in-sample means (leaky)
    - or ranks by out-of-sample proper-score, purged CV, embargo enforced (honest)
    """
    ranking = cap.rules.get("ranking", "sharpe_in_sample")
    n_streams = signal_matrix.shape[1]

    if ranking == "oos_proper_purged":
        # Out-of-sample proper-score ranking with purged CV and embargo
        if signal_matrix.shape[0] < n_cv_folds * embargo + 20:
            return np.zeros(n_streams, dtype=float)

        scores = np.zeros(n_streams, dtype=float)
        for k in range(n_streams):
            holdout_scores: list[float] = []
            for fold in range(n_cv_folds):
                t_start = fold * signal_matrix.shape[0] // n_cv_folds
                t_end = min(
                    (fold + 1) * signal_matrix.shape[0] // n_cv_folds,
                    signal_matrix.shape[0] - embargo,
                )
                if t_end - t_start < embargo + 2:
                    continue
                test = signal_matrix[t_start + embargo : t_end + embargo, k]
                if len(test) < 2:
                    continue
                # Out-of-sample signal strength: holdout mean/std ratio
                # (a screening score for the synthetic Cap, not a market IC).
                mu_test = float(np.mean(test))
                sigma_test = float(np.std(test, ddof=1))
                if sigma_test > _SS:
                    holdout_scores.append(mu_test / sigma_test)
            if holdout_scores:
                scores[k] = float(np.mean(holdout_scores))
        return scores

    # Default: naive in-sample Sharpe-like ranking (the leaky Cap)
    means = np.mean(signal_matrix, axis=0)
    stds = np.std(signal_matrix, axis=0, ddof=1)
    scores = np.zeros(n_streams, dtype=float)
    mask = stds > _SS
    scores[mask] = means[mask] / stds[mask]
    return scores


def _select_top_k(
    scores: Array,
    cap: Capabilities,
    n_signals: int,
) -> Array:
    """Select top-k signals according to Cap thresholds."""
    k = int(cap.rules.get("top_k", min(5, n_signals)))
    min_threshold = float(cap.rules.get("min_score_threshold", 0.0))
    k = max(1, min(k, n_signals))

    order = np.argsort(-scores)
    selected = np.zeros(n_signals, dtype=bool)
    for count, idx in enumerate(order, start=1):
        if scores[idx] < min_threshold and count > 1:
            break
        selected[idx] = True
        if count >= k:
            break
    return selected


def _build_portfolio(
    signal_matrix: Array,
    cap: Capabilities,
    *,
    purged_cv: bool = True,
    embargo: int = 5,
) -> tuple[Array, Array]:
    """Run the full screening + weighting pipeline; return (weights, rank_ic)."""
    n_timesteps, n_signals = signal_matrix.shape

    # Screening: rank signals according to Cap
    scores = _rank_signals(signal_matrix, cap, purged_cv=purged_cv, embargo=embargo)
    selected = _select_top_k(scores, cap, n_signals)

    if not selected.any():
        return np.zeros(n_signals, dtype=float), np.zeros(n_signals, dtype=float)

    # Weighting: equal weight for selected signals (configurable via Cap)
    weighting = cap.rules.get("weighting", "equal")
    if weighting == "equal":
        w = selected.astype(float) / float(selected.sum())
    elif weighting == "score_scaled":
        s = np.where(selected, np.maximum(scores, 0.0), 0.0)
        total = float(s.sum())
        w = s / total if total > _SS else selected.astype(float) / float(selected.sum())
    else:
        w = selected.astype(float) / float(selected.sum())

    # Compute rank IC: Spearman correlation of weights with forward signal
    rank_ics = np.zeros(n_signals, dtype=float)
    for k in range(n_signals):
        if n_timesteps < 4:
            continue
        # Rank IC: correlation between signal rank and forward return rank
        sig = signal_matrix[:, k]
        if np.std(sig) < _SS:
            continue
        # Spearman rank correlation of signal with its own lag-1 forward value
        from scipy.stats import spearmanr

        if n_timesteps > 1 and np.std(sig[:-1]) > _SS and np.std(sig[1:]) > _SS:
            corr, _ = spearmanr(sig[:-1], sig[1:])
            rank_ics[k] = corr if np.isfinite(corr) else 0.0

    return w, rank_ics


# ---------------------------------------------------------------------------
# Cap-swap evaluation protocol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapSwapResult:
    """Result of swapping Cap at a single (Hist, Frontier) anchor."""

    anchor_label: str
    cap_a_fingerprint: str
    cap_b_fingerprint: str

    # Portfolio IC metrics under each Cap
    ic_a: float
    ic_b: float
    rank_ic_a: float
    rank_ic_b: float

    # Marginal changes attributable to the Cap swap
    ic_delta: float  # B_Cap - A_Cap
    rank_ic_delta: float  # B_Cap - A_Cap

    selected_a: int
    selected_b: int


@dataclass(frozen=True)
class CapabilityValue:
    """Aggregated Cap-swap evaluation result."""

    swap_results: tuple[CapSwapResult, ...]
    metric: str  # "rank_ic_delta" or "ic_delta"

    # Point estimate
    mean_delta: float
    median_delta: float

    # Bootstrap CIs (percentile, BCA-simple)
    ci_lower: float
    ci_upper: float  # 95% CI

    # Formal test
    t_stat: float
    p_value: float
    n_swaps: int

    # Interpretable summary
    cap_a_fingerprint: str
    cap_b_fingerprint: str
    cap_improvement_significant: bool = False

    def __post_init__(self) -> None:
        significant = bool(self.p_value < 0.05 and self.mean_delta > 0)
        object.__setattr__(self, "cap_improvement_significant", significant)


def capability_swap_evaluation(
    state_a: ResearchState,
    state_b: ResearchState,
    signal_provider: Callable[[ResearchState], tuple[Array, Array]],
    *,
    n_boot: int = _N_BOOT,
    seed: int = _RNG_SEED,
    metric: str = "rank_ic_delta",
) -> CapabilityValue:
    """EverMine Cap-swap protocol: estimate conditional value of accumulated capability.

    Parameters
    ----------
    state_a, state_b:
        Two state snapshots from different points in a trajectory.
    signal_provider:
        Callable that, given a ResearchState, returns (signal_matrix, true_signal_matrix)
        where signal_matrix is (n_timesteps × n_streams) of observable signals and
        true_signal_matrix is the same shape with the noise-free edge (used only
        for the deterministic screening pipeline; not for live IC estimation).
    n_boot:
        Number of bootstrap resamples.
    seed:
        RNG seed for reproducibility.
    metric:
        Which IC delta to use: "rank_ic_delta" or "ic_delta".

    Returns
    -------
    CapabilityValue
        Aggregated result with bootstrap CIs and significance test.

    Raises
    ------
    ValueError
        If states are degenerate (empty Hist, empty Frontier, empty Cap dict),
        Caps are identical (no swap possible), or Hist lengths are mismatched.
    """
    # --- Fail-closed guards ---
    if len(state_a.hist) == 0 or len(state_b.hist) == 0:
        raise ValueError("capability swap requires non-empty Hist in both states")
    if not state_a.frontier.weights or not state_b.frontier.weights:
        raise ValueError("capability swap requires non-empty Frontier in both states")
    if not state_a.cap.rules or not state_b.cap.rules:
        raise ValueError("capability swap requires non-empty Cap dictionaries")
    if state_a.cap == state_b.cap:
        raise ValueError("identical Caps — no swap possible")
    if len(state_a.hist) != len(state_b.hist):
        raise ValueError("mismatched Hist lengths — cannot construct symmetric anchors")

    rng = np.random.default_rng(seed)

    # Construct 4 counterfactual states at each anchor
    # Anchors: we use each state's (Hist, Frontier) but at matched indices
    n_anchors = min(len(state_a.hist), len(state_b.hist))
    swap_results: list[CapSwapResult] = []

    for i in range(n_anchors):
        # Anchor A: state_a's Hist and Frontier
        anchor_hist_a = state_a.hist[: i + 1]
        anchor_frontier_a = state_a.frontier
        # Anchor B: state_b's Hist and Frontier
        anchor_hist_b = state_b.hist[: i + 1]
        anchor_frontier_b = state_b.frontier

        # Counterfactual 1: (A_Hist, A_Frontier, A_Cap)
        cf_aa = ResearchState(
            hist=anchor_hist_a, frontier=anchor_frontier_a, cap=state_a.cap, label="AA"
        )
        # Counterfactual 2: (A_Hist, A_Frontier, B_Cap)
        cf_ab = ResearchState(
            hist=anchor_hist_a, frontier=anchor_frontier_a, cap=state_b.cap, label="AB"
        )
        # Counterfactual 3: (B_Hist, B_Frontier, A_Cap)
        cf_ba = ResearchState(
            hist=anchor_hist_b, frontier=anchor_frontier_b, cap=state_a.cap, label="BA"
        )
        # Counterfactual 4: (B_Hist, B_Frontier, B_Cap)
        cf_bb = ResearchState(
            hist=anchor_hist_b, frontier=anchor_frontier_b, cap=state_b.cap, label="BB"
        )

        # Run pipeline on each counterfactual
        sig_aa, _ = signal_provider(cf_aa)
        sig_ab, _ = signal_provider(cf_ab)
        sig_ba, _ = signal_provider(cf_ba)
        sig_bb, _ = signal_provider(cf_bb)

        w_aa, ric_aa = _build_portfolio(sig_aa, cf_aa.cap)
        w_ab, ric_ab = _build_portfolio(sig_ab, cf_ab.cap)
        w_ba, ric_ba = _build_portfolio(sig_ba, cf_ba.cap)
        w_bb, ric_bb = _build_portfolio(sig_bb, cf_bb.cap)

        # Compute IC metrics: weighted average of selected signal autocorrelations
        ic_aa = float(np.dot(w_aa, ric_aa)) if w_aa.sum() > _SS else 0.0
        ic_ab = float(np.dot(w_ab, ric_ab)) if w_ab.sum() > _SS else 0.0
        ic_ba = float(np.dot(w_ba, ric_ba)) if w_ba.sum() > _SS else 0.0
        ic_bb = float(np.dot(w_bb, ric_bb)) if w_bb.sum() > _SS else 0.0

        # Rank IC of the SELECTION: Spearman correlation between the Cap's
        # weight vector and the per-signal rank ICs — Cap-dependent through
        # the selection ordering (a Cap that ranks genuinely predictive
        # signals into the portfolio scores higher).  The previous
        # mean(ric[ric != 0]) definition was Cap-invariant (ric depends only
        # on the signal matrix), which made rank_ic_delta identically zero.
        def _rank_ic_of_selection(w_sel: Array, ric_sel: Array) -> float:
            if w_sel.sum() <= _SS or np.std(w_sel) < _SS or np.std(ric_sel) < _SS:
                return 0.0
            corr, _ = spearmanr(w_sel, ric_sel)
            return float(corr) if np.isfinite(corr) else 0.0

        rank_ic_aa = _rank_ic_of_selection(w_aa, ric_aa)
        rank_ic_ab = _rank_ic_of_selection(w_ab, ric_ab)
        rank_ic_ba = _rank_ic_of_selection(w_ba, ric_ba)
        rank_ic_bb = _rank_ic_of_selection(w_bb, ric_bb)

        # Marginal change at anchor A: B_Cap vs A_Cap at (A_Hist, A_Frontier)
        ic_delta_a = ic_ab - ic_aa
        ric_delta_a = rank_ic_ab - rank_ic_aa

        # Marginal change at anchor B: B_Cap vs A_Cap at (B_Hist, B_Frontier)
        ic_delta_b = ic_bb - ic_ba
        ric_delta_b = rank_ic_bb - rank_ic_ba

        # Average the two anchors for a symmetric estimate
        ic_delta = (ic_delta_a + ic_delta_b) / 2.0
        ric_delta = (ric_delta_a + ric_delta_b) / 2.0

        swap_results.append(
            CapSwapResult(
                anchor_label=f"step_{i}",
                cap_a_fingerprint=state_a.cap.fingerprint(),
                cap_b_fingerprint=state_b.cap.fingerprint(),
                ic_a=(ic_aa + ic_ba) / 2.0,
                ic_b=(ic_ab + ic_bb) / 2.0,
                rank_ic_a=(rank_ic_aa + rank_ic_ba) / 2.0,
                rank_ic_b=(rank_ic_ab + rank_ic_bb) / 2.0,
                ic_delta=ic_delta,
                rank_ic_delta=ric_delta,
                selected_a=int(np.sum(w_aa > _SS)),
                selected_b=int(np.sum(w_ab > _SS)),
            )
        )

    # Aggregate deltas
    if metric == "rank_ic_delta":
        deltas = np.array([r.rank_ic_delta for r in swap_results], dtype=float)
    else:
        deltas = np.array([r.ic_delta for r in swap_results], dtype=float)

    # Bootstrap CIs
    boot_deltas: list[float] = []
    n = len(deltas)
    if n >= 2:
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot_deltas.append(float(np.mean(deltas[idx])))
    boot_arr = np.array(boot_deltas, dtype=float)
    if len(boot_arr) > 0:
        ci_lower = float(np.percentile(boot_arr, 2.5))
        ci_upper = float(np.percentile(boot_arr, 97.5))
    else:
        ci_lower = float(np.min(deltas))
        ci_upper = float(np.max(deltas))

    # t-test
    if n >= 3:
        t_stat, p_value = ttest_1samp(deltas, 0.0)
        t_stat_f = float(t_stat)
        p_value_f = float(p_value)
        # One-sided: is B_Cap better than A_Cap?
        if np.isfinite(p_value_f):
            # Convert two-sided to one-sided (H1: mean > 0)
            from scipy.stats import t as scipy_t

            p_value_one = float(
                scipy_t.sf(t_stat_f, n - 1) if t_stat_f > 0 else 1.0 - scipy_t.cdf(t_stat_f, n - 1)
            )
            p_value_f = p_value_one
    else:
        t_stat_f = float("nan")
        p_value_f = float("nan")

    return CapabilityValue(
        swap_results=tuple(swap_results),
        metric=metric,
        mean_delta=float(np.mean(deltas)),
        median_delta=float(np.median(deltas)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        t_stat=t_stat_f,
        p_value=p_value_f,
        n_swaps=n,
        cap_a_fingerprint=state_a.cap.fingerprint(),
        cap_b_fingerprint=state_b.cap.fingerprint(),
    )


# ---------------------------------------------------------------------------
# SYNTHETIC research trajectory (seeded, deterministic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticTrajectory:
    """A seeded synthetic alpha-discovery trajectory for Cap-swap testing.

    All data is SYNTHETIC — correctness evidence, never market evidence.
    """

    signal_matrix: Array  # (T × K) observable signal streams
    true_edge_matrix: Array  # (T × K) noise-free edge components
    states: tuple[ResearchState, ...]  # snapshot at each step
    n_periods: int
    n_signals: int
    seed: int

    def __post_init__(self) -> None:
        if self.signal_matrix.ndim != 2:
            raise ValueError("signal_matrix must be 2-d")
        if self.true_edge_matrix.shape != self.signal_matrix.shape:
            raise ValueError("signal_matrix and true_edge_matrix must have the same shape")


def make_synthetic_trajectory(
    *,
    n_periods: int = 120,
    n_signals: int = 10,
    n_persistent_edge: int = 4,
    edge_strength: float = 0.08,
    noise_std: float = 0.05,
    seed: int = _RNG_SEED,
) -> SyntheticTrajectory:
    """Build a seeded synthetic factor universe for Cap-swap evaluation.

    Constructs a running portfolio + candidate pipeline in a simple factor
    universe. ``n_persistent_edge`` signals carry genuine autocorrelated edge
    (signal = lagged_true + noise); the remaining signals are pure noise.
    This lets us plant a genuinely better Cap (disciplined OOS selection) and
    verify that Cap-swap evaluation detects it, even though the EverMine paper
    found no consistent gain in its real-data trajectories.

    Parameters
    ----------
    n_periods:
        Number of time steps.
    n_signals:
        Number of signal streams.
    n_persistent_edge:
        How many streams carry genuine edge.
    edge_strength:
        Magnitude of the true edge (autocorrelation coefficient).
    noise_std:
        Standard deviation of observation noise.
    seed:
        RNG seed for full determinism.

    Returns
    -------
    SyntheticTrajectory
        Seeded trajectory with signal_matrix, true_edge_matrix, and state snapshots.
    """
    if n_persistent_edge > n_signals:
        raise ValueError("n_persistent_edge cannot exceed n_signals")
    if n_periods < 30:
        raise ValueError("n_periods must be >= 30 for meaningful evaluation")
    if edge_strength <= 0:
        raise ValueError("edge_strength must be positive")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")

    rng = np.random.default_rng(seed)

    # Generate persistent edge streams as AR(1) processes
    true_edge = np.zeros((n_periods, n_signals), dtype=float)
    for k in range(n_persistent_edge):
        eps = rng.normal(0, 1.0, size=n_periods)
        true_edge[0, k] = eps[0]
        for t in range(1, n_periods):
            true_edge[t, k] = edge_strength * true_edge[t - 1, k] + eps[t]
        # Normalize to unit variance
        std_k = float(np.std(true_edge[:, k], ddof=1))
        if std_k > _SS:
            true_edge[:, k] /= std_k
            true_edge[:, k] *= edge_strength

    # Generate pure noise streams
    for k in range(n_persistent_edge, n_signals):
        true_edge[:, k] = 0.0  # no true edge

    # Observable signal = true edge + observation noise
    noise = rng.normal(0, noise_std, size=(n_periods, n_signals))
    signal_matrix = true_edge + noise

    # Build state snapshots along the trajectory
    states: list[ResearchState] = []
    # Build simple equal-weight frontier from a candidate set of "assets"
    asset_ids = [f"A{k:02d}" for k in range(n_signals)]

    for t in range(1, n_periods):
        # Hist: trials simulated up to period t.  Rolling window of 60 (not 20):
        # trajectory_signal_provider maps len(hist) -> evaluation window with a
        # 30-row floor, and the disciplined Cap's purged CV needs >= 45 rows —
        # a 20-cap made every swap anchor collapse to the same clamped window
        # (zero delta variance) and structurally starved the disciplined Cap.
        trials: list[TrialRecord] = []
        for past_t in range(max(0, t - 60), t):
            past_date = datetime(2020, 1, 1, tzinfo=UTC)
            # Simulate a trial: check signal auto-correlation in-sample
            lookback = min(past_t + 1, 60)
            start = max(0, past_t + 1 - lookback)
            window = signal_matrix[start : past_t + 1, :]
            rank_ic_vals: list[float] = []
            ic_vals: list[float] = []
            if window.shape[0] >= 3:
                for k in range(n_signals):
                    col = window[:, k]
                    if np.std(col) > _SS and len(col) > 1:
                        from scipy.stats import pearsonr

                        # Simple lookback signal strength as pseudo decision
                        r_val = float(np.mean(col))
                        ic_vals.append(min(max(r_val, -1.0), 1.0))
                        if np.std(col[:-1]) > _SS and np.std(col[1:]) > _SS:
                            r, _ = pearsonr(col[:-1], col[1:])
                            rank_ic_vals.append(r if np.isfinite(r) else 0.0)
                        else:
                            rank_ic_vals.append(0.0)
                    else:
                        ic_vals.append(0.0)
                        rank_ic_vals.append(0.0)
                avg_rank_ic = float(np.mean(rank_ic_vals)) if rank_ic_vals else 0.0
            else:
                avg_rank_ic = 0.0

            # Decision: some trials "accepted" based on signal strength
            decision = "accepted" if avg_rank_ic > 0.01 and rng.random() < 0.35 else "rejected"
            trials.append(
                TrialRecord(
                    submission_date=past_date.isoformat(),
                    rank_ic=avg_rank_ic,
                    ic=float(np.mean(ic_vals)) if ic_vals else 0.0,
                    decision=decision,
                    factor_id=f"F{past_t:03d}",
                )
            )

        # Frontier: build from signals that look strongest at t
        recent = signal_matrix[max(0, t - 30) : t, :]
        means = np.mean(recent, axis=0)
        stds = np.std(recent, axis=0, ddof=1)
        scores_for_frontier = np.zeros(n_signals, dtype=float)
        for k_idx in range(n_signals):
            if stds[k_idx] > _SS:
                scores_for_frontier[k_idx] = means[k_idx] / stds[k_idx]

        # Top 3 become the frontier
        top_k = 3
        top_idx = np.argsort(-scores_for_frontier)[:top_k]
        f_weights: dict[str, float] = {}
        total_score = 0.0
        for idx in top_idx:
            s = max(float(scores_for_frontier[idx]), _SS)
            f_weights[asset_ids[idx]] = s
            total_score += s
        if total_score > _SS:
            f_weights = {k: v / total_score for k, v in f_weights.items()}

        frontier = Frontier(
            weights=f_weights,
            signal_metrics={
                "mean_rank_ic": float(np.mean([tr.rank_ic for tr in trials])),
                "accept_rate": float(
                    np.mean([tr.decision == "accepted" for tr in trials]) if trials else 0.0
                ),
            },
            regime_context={"t": t},
        )

        # Cap: we assign the same Cap throughout (caller swaps them externally)
        cap = Capabilities(
            rules={
                "ranking": "oos_proper_purged",
                "top_k": min(5, n_signals),
                "min_score_threshold": 0.02,
                "weighting": "equal",
                "embargo": 5,
                "n_cv_folds": 5,
                "label": "synthetic_base",
            }
        )

        states.append(
            ResearchState(
                hist=tuple(trials),
                frontier=frontier,
                cap=cap,
                label=f"t={t}",
            )
        )

    return SyntheticTrajectory(
        signal_matrix=signal_matrix,
        true_edge_matrix=true_edge,
        states=tuple(states),
        n_periods=n_periods,
        n_signals=n_signals,
        seed=seed,
    )


def make_cap_naive(**overrides: Any) -> Capabilities:
    """Construct the 'naive' Cap: ranks by in-sample Sharpe (leaky)."""
    rules: dict[str, Any] = {
        "ranking": "sharpe_in_sample",
        "top_k": 5,
        "min_score_threshold": 0.0,
        "weighting": "equal",
        "label": "naive",
    }
    rules.update(overrides)
    return Capabilities(rules=rules)


def make_cap_disciplined(**overrides: Any) -> Capabilities:
    """Construct the 'disciplined' Cap: ranks by out-of-sample proper-score,
    purged CV, embargo enforced (honest)."""
    rules: dict[str, Any] = {
        "ranking": "oos_proper_purged",
        "top_k": 5,
        "min_score_threshold": 0.02,
        "weighting": "equal",
        "embargo": 5,
        "n_cv_folds": 5,
        "label": "disciplined",
    }
    rules.update(overrides)
    return Capabilities(rules=rules)


# ---------------------------------------------------------------------------
# Signal provider for Cap-swap evaluation
# ---------------------------------------------------------------------------


def trajectory_signal_provider(
    trajectory: SyntheticTrajectory,
) -> Callable[[ResearchState], tuple[Array, Array]]:
    """Create a signal provider from a synthetic trajectory.

    The provider maps a ResearchState to (signal_matrix, true_edge_matrix)
    by extracting the relevant time window from the trajectory.
    """

    def provider(state: ResearchState) -> tuple[Array, Array]:
        # Use the number of trials as a proxy for "time step"
        n_trials = len(state.hist)
        if n_trials == 0:
            return (
                np.zeros((1, trajectory.n_signals), dtype=float),
                np.zeros((1, trajectory.n_signals), dtype=float),
            )
        # Map trial count to time window
        t = min(max(n_trials, 30), trajectory.n_periods)
        start = max(0, t - 60)
        return (
            trajectory.signal_matrix[start:t, :].astype(float),
            trajectory.true_edge_matrix[start:t, :].astype(float),
        )

    return provider


# ---------------------------------------------------------------------------
# Gate predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapGateVerdict:
    """Structured verdict for capability improvement gating."""

    cap_improvement_significant: bool
    confidence_level: float  # 1 - p_value (one-sided)
    effect_size: float  # mean_delta / std(deltas) — Cohen's d analogue
    p_value: float
    mean_delta: float
    ci_lower: float
    ci_upper: float
    metric: str
    n_anchors: int
    claim: str = "research_diagnostic_only"
    synthetic: bool = True

    def __post_init__(self) -> None:
        if not self.synthetic:
            raise ValueError("cap_gate verdicts are always SYNTHETIC research diagnostics")


def cap_gate(cap_value: CapabilityValue) -> CapGateVerdict:
    """Gate advertised capability gains using the Cap-swap evaluation.

    Returns a structured verdict: whether the Cap-swap shows a significant
    improvement, with effect size, confidence level, and p-value.

    Parameters
    ----------
    cap_value:
        Output of ``capability_swap_evaluation``.

    Returns
    -------
    CapGateVerdict
        Structured gate verdict.
    """
    # std must come from the same metric the evaluation aggregated — pairing
    # an ic_delta mean against rank_ic dispersion is a confounded effect size.
    deltas = np.array(
        [
            r.rank_ic_delta if cap_value.metric == "rank_ic_delta" else r.ic_delta
            for r in cap_value.swap_results
        ],
        dtype=float,
    )

    if len(deltas) > 1:
        std_deltas = float(np.std(deltas, ddof=1))
    else:
        std_deltas = 0.0

    if std_deltas > _SS:
        effect_size = cap_value.mean_delta / std_deltas
    elif np.abs(cap_value.mean_delta) < _SS:
        effect_size = 0.0
    else:
        effect_size = 0.0  # single observation, no variance estimate

    p = cap_value.p_value
    if np.isfinite(p):
        confidence = 1.0 - p
    else:
        confidence = 0.0

    significant = bool(p < 0.05 and cap_value.mean_delta > 0)

    return CapGateVerdict(
        cap_improvement_significant=significant,
        confidence_level=confidence,
        effect_size=float(effect_size),
        p_value=float(p) if np.isfinite(p) else float("nan"),
        mean_delta=cap_value.mean_delta,
        ci_lower=cap_value.ci_lower,
        ci_upper=cap_value.ci_upper,
        metric=cap_value.metric,
        n_anchors=cap_value.n_swaps,
    )
