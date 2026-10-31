"""FASE: feedback-aware self-evolving forecast evaluation machinery.

Implements the deterministic core of FASE, the Feedback-Aware Self-Evolving
forecasting agent of

    Wang, J., Wang, Y., Wu, W., Zhang, C. (2026). "Self-Evolving Time-Series
    Forecasting Agents with Episodic Memory and Online Policy Learning."
    arXiv:2609.32689 [cs.LG]. Citation verified against
    https://arxiv.org/abs/2609.32689 and the full HTML text (fetched
    2026-09-30).

Spec-vs-paper correction: the lane spec named FASE "forecast-aware /
factor-augmented score evaluation". The cited paper defines FASE as a
*feedback-aware self-evolving* forecasting agent; this module implements what
the paper actually specifies — the two feedback stores (episodic memory and
an online pairwise-ranking policy) and the GIFT-Eval evaluation protocol used
to measure it — not a factor-augmented scoring rule.

Paper machinery implemented here (section references are to the fetched text)
---------------------------------------------------------------------------
1. ``statistical_features`` — §4.2 / Appendix D, Table 7: the 18 statistical
   features summarising each observed history, in the paper's eight feature
   groups (data quality, temporal dynamics, structural change, spectral
   structure, missingness, periodicity, covariates, intermittency). Table 7
   gives one-line descriptions only; each estimator below states its exact
   operational definition. Unobservable features (e.g. covariate features
   with no covariates) are NaN, never silently zeroed.
2. ``normalised_statistical_distance`` / ``statistical_distance_row`` —
   §3.2 eq. (1): group-then-feature averaged bounded discrepancy
   ``d(s_i, s_j) = (1/|G_ij|) Σ_g (1/|Q_ij,g|) Σ_q ρ(|s_i,q − s_j,q| / σ_q)``
   with ``ρ(z) = z/(1+z)`` and ``σ_q`` the feature std across active entries.
3. ``EpisodicMemory`` — §3.2: recent + long-term pools (paper capacities 100 /
   900, retrieval k = 10), the eq. (2) retention contribution (partition
   nearest-neighbour margin over ``(invoked tool, tied-best-rank tool set)``
   cells), running-average retention values, and retention-gated promotion to
   the long-term pool.
4. ``OnlineToolRanker`` — §3.3 / Appendix C: the lightweight MLP policy
   ``g_θ : R^{2N} → R^{|F|}`` over ``z_i = [Normalise(x_i), Mask(x_i)]``
   (median/std normalisation of the last N = 8H history values), trained
   online on the eq. (3) pairwise loss
   ``L(z, ℓ) = (1/C(F,2)) Σ_{p<q} (|ℓ_p − ℓ_q| / ℓ̄) softplus(sgn(ℓ_q − ℓ_p)
   (u_p − u_q))`` with AdamW (lr 1e-3, wd 1e-5, paper seed 7), a 3,000-record
   FIFO feedback pool, initialisation after 100 records (10 epochs) and a
   periodic update every 25 records (25 uniform minibatches of 256).
5. ``seasonal_naive_forecast`` / ``seasonal_naive_scale`` / ``mase_score`` /
   ``normalised_geometric_mean`` / ``mean_config_ranks`` — §4.1 GIFT-Eval
   protocol: MASE/MAE/RMSE point metrics and quantile-pinball CRPS, each
   metric normalised by the Seasonal Naive score per configuration, geometric
   mean across configurations, and within-configuration average-tie ranks
   averaged arithmetically.
6. ``self_evolution_curve`` — §5.2: cumulative mean-MASE reduction of the
   agent relative to a feedback-free base controller, in percent.
7. ``fase_benchmark`` — Appendix B online protocol: replay forecasting
   instances in fixed order, run every tool, score all tools, feed point and
   probabilistic errors back into memory and the feedback pool. Returns the
   repo's flat ``dict[str, float]`` bench style.

Composition (imported, not reimplemented)
------------------------------------------
- ``metrics.scoring.crps_from_quantiles`` — the pinball-Riemann CRPS the paper
  estimates from nine quantile levels (0.1–0.9); not reimplemented here.
- ``metrics.forecast_eval`` / ``metrics.hac`` own DM/HAC comparison tests —
  the paper reports no significance test, so none is added here.
- ``utils.numeric.midrank`` — average-tie ranks (per-instance tool ranks and
  per-configuration method ranks), shared with the rest of the metrics canon.
- ``utils.series`` — shared series guards.

Deliberate deviations (documented, not hidden)
---------------------------------------------
- The paper's action selector is an LLM (Qwen3.8-27B) consuming textual
  memory/policy guidance. This port is deterministic and LLM-free: the
  ``fase`` variant follows the policy ranking (argmin score) once trained and
  a retrieved-evidence vote before that; ``memory_only`` votes from retrieved
  entries; ``base`` selects a fixed tool (``baseline_index``). This is the
  faithful deterministic reduction of "the controller follows the guidance";
  it cannot reproduce LLM idiosyncrasies and is labelled as such.
- MLP hidden widths default to the paper's 2048 → 128 but are configurable;
  the paper's N = 8H input window is likewise configurable.
- Promotion to a full long-term pool replaces the lowest-retention entry iff
  the candidate's retention strictly exceeds it (the paper says only that
  promotion is "determined by its retention value").

Honesty: every score here is a proper score or an honest error measure
(MAE/RMSE/MASE/CRPS and rank aggregates). No Sharpe/Sortino/Calmar/P&L/NAV is
produced; nothing in this module is a live-trading claim. All randomness
(MLP init, minibatch sampling) flows through a single seeded numpy
Generator, so repeated calls are bit-identical. Tests are SYNTHETIC
correctness checks only — never market evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit as _sigmoid

from quant_fund.metrics.scoring import crps_from_quantiles
from quant_fund.utils.numeric import midrank
from quant_fund.utils.series import as_named_1d

__all__ = [
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "N_FEATURES",
    "QUANTILE_LEVELS",
    "BenchmarkInstance",
    "EpisodicMemory",
    "MemoryEntry",
    "OnlineToolRanker",
    "PolicyConfig",
    "fase_benchmark",
    "mean_config_ranks",
    "mase_score",
    "normalised_geometric_mean",
    "normalised_statistical_distance",
    "seasonal_naive_forecast",
    "seasonal_naive_scale",
    "self_evolution_curve",
    "statistical_distance_row",
    "statistical_features",
]

Array = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Appendix D, Table 7 — the 18 statistical features in eight groups.
# Order within a group is the table order; group order is the table order.
# ---------------------------------------------------------------------------

FEATURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("data_quality", ("missing_ratio", "zero_ratio")),
    (
        "temporal_dynamics",
        (
            "global_trend_strength",
            "recent_trend_strength",
            "recent_level_shift_strength",
            "recent_scale_change_strength",
        ),
    ),
    (
        "structural_change",
        (
            "change_strength",
            "linear_fit_error_ratio",
            "turning_behaviour_ratio",
            "direction_entropy",
        ),
    ),
    (
        "spectral_structure",
        ("spectral_concentration", "spectral_entropy", "stability"),
    ),
    ("missingness", ("longest_missing_block",)),
    ("periodicity", ("recent_period_strength",)),
    (
        "covariates",
        ("strongest_covariate_relation", "covariate_relation_stability"),
    ),
    ("intermittency", ("recent_event_rate_change",)),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for _, names in FEATURE_GROUPS for name in names)
N_FEATURES: int = len(FEATURE_NAMES)

#: Probabilistic-feedback quantile levels (paper §4.1): 0.1 … 0.9.
QUANTILE_LEVELS: Array = np.round(np.arange(0.1, 1.0, 0.1), 10)

_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_GROUP_SPANS: tuple[tuple[int, ...], ...] = tuple(
    tuple(_FEATURE_INDEX[name] for name in names) for _, names in FEATURE_GROUPS
)

# Paper Table 6 defaults.
RECENT_CAPACITY = 100
LONGTERM_CAPACITY = 900
RETRIEVE_K = 10
FEEDBACK_CAPACITY = 3000
POLICY_WARMUP = 100
POLICY_INIT_EPOCHS = 10
POLICY_UPDATE_EVERY = 25
POLICY_UPDATE_BATCHES = 25
POLICY_BATCH_SIZE = 256
POLICY_LR = 1e-3
POLICY_WEIGHT_DECAY = 1e-5
POLICY_SEED = 7
POLICY_HIDDEN_DIMS: tuple[int, ...] = (2048, 128)

# Fraction of the observed tail treated as "recent" by the trend/level/scale
# and event-rate features; the paper does not pin the split point.
_RECENT_FRAC = 0.25


# ---------------------------------------------------------------------------
# Validation helpers (fail-closed; house style)
# ---------------------------------------------------------------------------


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _history(values: Array, *, min_observed: int = 4) -> tuple[Array, Array]:
    """Return (raw series with NaNs allowed, finite observed values)."""
    x = as_named_1d("history", np.asarray(values, dtype=float))
    if x.size < min_observed:
        raise ValueError(f"history must have length >= {min_observed}")
    obs = x[np.isfinite(x)]
    if obs.size < min_observed:
        raise ValueError(f"history must contain >= {min_observed} observed values")
    return x, obs


def _finite_vector(name: str, values: Array, size: int | None = None) -> Array:
    v = as_named_1d(name, np.asarray(values, dtype=float))
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if size is not None and v.size != size:
        raise ValueError(f"{name} must have length {size}")
    return v


def _ols_slope(t: Array, x: Array) -> float:
    """Least-squares slope of x on t; fails closed on degenerate spread."""
    tt = t - t.mean()
    denom = float(tt @ tt)
    if denom <= 0.0:
        raise ValueError("degenerate time index")
    return float(tt @ (x - x.mean()) / denom)


def _recent_split(obs: Array, frac: float = _RECENT_FRAC) -> tuple[Array, Array]:
    """Split observed values into (prior, recent) at the ``frac`` tail boundary."""
    n = obs.size
    k = max(2, int(round(frac * n)))
    if k >= n:
        k = n // 2
    return obs[: n - k], obs[n - k :]


# ---------------------------------------------------------------------------
# 1. Statistical features (Appendix D, Table 7)
# ---------------------------------------------------------------------------


def _spectral_shares(obs: Array) -> Array:
    """Normalised power shares of the non-DC rFFT bins of the detrended series."""
    n = obs.size
    t = np.arange(n, dtype=float)
    detr = obs - (obs.mean() + _ols_slope(t, obs) * t)
    power = np.abs(np.fft.rfft(detr)) ** 2
    power = power[1:]  # drop DC
    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("degenerate spectrum")
    return power / total


def statistical_features(
    history: Array,
    *,
    covariates: Array | None = None,
    recent_frac: float = _RECENT_FRAC,
    n_blocks: int = 8,
    top_spectral: int = 2,
) -> Array:
    """Table-7 statistical representation ``s_i`` of one observed history.

    Returns the 18-vector in ``FEATURE_NAMES`` order. Missing target values
    are handled honestly: missingness features are computed on the raw series
    and all other features on the observed values only. Features with no
    observable basis (too few observations, no covariates) return NaN so the
    eq. (1) distance drops them from shared-feature averages — a feature is
    never fabricated.

    Operational definitions (Table 7 gives one-line descriptions only):
    trend strengths are ``|OLS slope| · span / σ``; level shift is
    ``(mean_recent − mean_prior) / σ``; scale change is
    ``(σ_recent − σ_prior) / σ_prior``; change strength is the max
    standardised mean gap over decile splits; linear fit error ratio is
    one-step linear-extrapolation MSE over persistence MSE; turning and
    direction features use the signs of successive differences; spectral
    features act on the OLS-detrended observed series; stability is the
    variance of σ-standardised contiguous block means; periodicity is the
    largest spectral share inside the recent segment; covariate features use
    |Pearson corr| between target and covariates; intermittency is the
    nonzero-rate change between recent and prior parts.

    Fail-closed: history shorter than 4 positions or with fewer than 4
    observed values raises ValueError.
    """
    x, obs = _history(history, min_observed=4)
    n = x.size
    prior, recent = _recent_split(obs, recent_frac)
    sigma = _safe_std(obs)
    diffs = np.diff(obs)

    feats = np.full(N_FEATURES, np.nan, dtype=float)

    # --- data quality ---------------------------------------------------------
    feats[_FEATURE_INDEX["missing_ratio"]] = float(np.mean(~np.isfinite(x)))
    feats[_FEATURE_INDEX["zero_ratio"]] = float(np.mean(obs == 0.0))

    # --- temporal dynamics ----------------------------------------------------
    if sigma > 0.0:
        t_obs = np.arange(obs.size, dtype=float)
        feats[_FEATURE_INDEX["global_trend_strength"]] = (
            abs(_ols_slope(t_obs, obs)) * (obs.size - 1) / sigma
        )
        rt = np.arange(recent.size, dtype=float)
        feats[_FEATURE_INDEX["recent_trend_strength"]] = (
            abs(_ols_slope(rt, recent)) * (recent.size - 1) / sigma
        )
        feats[_FEATURE_INDEX["recent_level_shift_strength"]] = (
            recent.mean() - prior.mean()
        ) / sigma
        sp, sr = _safe_std(prior), _safe_std(recent)
        if np.isfinite(sp) and np.isfinite(sr):
            if sp > 0.0:
                feats[_FEATURE_INDEX["recent_scale_change_strength"]] = (sr - sp) / sp
            elif sr == 0.0:
                feats[_FEATURE_INDEX["recent_scale_change_strength"]] = 0.0

    # --- structural change and complexity -------------------------------------
    if sigma > 0.0 and obs.size >= 8:
        splits = np.unique(
            np.clip(np.round(np.linspace(0.1, 0.9, 9) * obs.size).astype(int), 1, obs.size - 1)
        )
        feats[_FEATURE_INDEX["change_strength"]] = float(
            np.max(np.abs([obs[:s].mean() - obs[s:].mean() for s in splits]) / sigma)
        )
    if obs.size >= 3:
        pers = float(np.sum(diffs[1:] ** 2))
        lin = float(np.sum((obs[2:] - (2.0 * obs[1:-1] - obs[:-2])) ** 2))
        if pers > 0.0:
            feats[_FEATURE_INDEX["linear_fit_error_ratio"]] = lin / pers
        elif lin == 0.0:
            feats[_FEATURE_INDEX["linear_fit_error_ratio"]] = 1.0
    nz = np.sign(diffs)
    nz = nz[nz != 0.0]
    if nz.size >= 2:
        feats[_FEATURE_INDEX["turning_behaviour_ratio"]] = float(np.mean(nz[1:] != nz[:-1]))
        p_down = float(np.mean(nz < 0.0))
        if 0.0 < p_down < 1.0:
            feats[_FEATURE_INDEX["direction_entropy"]] = float(
                -(p_down * math.log(p_down) + (1.0 - p_down) * math.log(1.0 - p_down))
                / math.log(2.0)
            )
        else:
            feats[_FEATURE_INDEX["direction_entropy"]] = 0.0

    # --- spectral structure ----------------------------------------------------
    if obs.size >= 4:
        try:
            shares = _spectral_shares(obs)
        except ValueError:
            shares = None
        if shares is not None:
            k = min(top_spectral, shares.size)
            feats[_FEATURE_INDEX["spectral_concentration"]] = float(np.sort(shares)[::-1][:k].sum())
            nzp = shares[shares > 0.0]
            feats[_FEATURE_INDEX["spectral_entropy"]] = float(
                -np.sum(nzp * np.log(nzp)) / math.log(shares.size)
            )
    if sigma > 0.0 and obs.size >= 2 * n_blocks:
        blocks = np.array_split(obs, n_blocks)
        feats[_FEATURE_INDEX["stability"]] = float(
            np.var(np.asarray([b.mean() for b in blocks]) / sigma)
        )

    # --- missingness ------------------------------------------------------------
    isnan = ~np.isfinite(x)
    if isnan.any():
        runs = np.diff(np.concatenate(([0], isnan.astype(np.int8), [0])))
        starts = np.flatnonzero(runs == 1)
        ends = np.flatnonzero(runs == -1)
        feats[_FEATURE_INDEX["longest_missing_block"]] = float(np.max(ends - starts) / n)
    else:
        feats[_FEATURE_INDEX["longest_missing_block"]] = 0.0

    # --- periodicity --------------------------------------------------------------
    if recent.size >= 4:
        try:
            rshares = _spectral_shares(recent)
        except ValueError:
            rshares = None
        if rshares is not None:
            feats[_FEATURE_INDEX["recent_period_strength"]] = float(np.max(rshares))

    # --- covariates -----------------------------------------------------------------
    if covariates is not None:
        c = np.asarray(covariates, dtype=float)
        if c.ndim != 2 or c.shape[0] != n or c.shape[1] < 1:
            raise ValueError("covariates must be a (T, K >= 1) panel aligned with history")
        c = c[np.isfinite(x)]
        ok_cols = np.isfinite(c).all(axis=0)
        c = c[:, ok_cols]
        if c.shape[1] == 0:
            raise ValueError("covariates contain no fully observed column")
        sd_x = float(np.std(obs))
        corrs = np.empty(c.shape[1])
        early = np.empty(c.shape[1])
        late = np.empty(c.shape[1])
        half = obs.size // 2
        for j in range(c.shape[1]):
            sd_c = float(np.std(c[:, j]))
            corrs[j] = float(np.corrcoef(obs, c[:, j])[0, 1]) if sd_x > 0.0 and sd_c > 0.0 else 0.0
            xe, xl = obs[:half], obs[half:]
            ce, cl = c[:half, j], c[half:, j]
            early[j] = (
                float(np.corrcoef(xe, ce)[0, 1])
                if xe.size >= 2 and np.std(xe) > 0.0 and np.std(ce) > 0.0
                else 0.0
            )
            late[j] = (
                float(np.corrcoef(xl, cl)[0, 1])
                if xl.size >= 2 and np.std(xl) > 0.0 and np.std(cl) > 0.0
                else 0.0
            )
        feats[_FEATURE_INDEX["strongest_covariate_relation"]] = float(np.max(np.abs(corrs)))
        # Agreement between early- and late-segment correlations, mapped to
        # [0, 1]: |Δcorr| ∈ [0, 2] → 1 − mean|Δ| / 2.
        feats[_FEATURE_INDEX["covariate_relation_stability"]] = float(
            1.0 - 0.5 * np.mean(np.abs(early - late))
        )

    # --- intermittency --------------------------------------------------------------
    feats[_FEATURE_INDEX["recent_event_rate_change"]] = float(
        np.mean(recent != 0.0) - np.mean(prior != 0.0)
    )
    return feats


def _safe_std(x: Array) -> float:
    """Population std; NaN when undefined (n < 2)."""
    return float(np.std(x)) if x.size >= 2 else float("nan")


# ---------------------------------------------------------------------------
# 2. Normalised statistical distance (§3.2, eq. 1)
# ---------------------------------------------------------------------------


def _rho(z: Array) -> Array:
    """Bounded discrepancy transform ρ(z) = z / (1 + z), z >= 0."""
    return z / (1.0 + z)


def _distance_row(query: Array, entries: Array, sigma: Array) -> Array:
    """Vector of eq. (1) distances from ``query`` to each row of ``entries``."""
    m = entries.shape[0]
    dists = np.zeros(m, dtype=float)
    counts = np.zeros(m, dtype=int)
    shared_feature = np.isfinite(query)[None, :] & np.isfinite(entries)
    for span in _GROUP_SPANS:
        shared = shared_feature[:, list(span)]
        has = shared.any(axis=1)
        if not has.any():
            continue
        diff = np.abs(query[None, list(span)] - entries[:, list(span)])
        sig = sigma[list(span)][None, :]
        # σ_q == 0 means the feature is constant over active entries: a zero
        # discrepancy contributes 0, any discrepancy saturates ρ at 1.
        term = np.where(
            diff == 0.0,
            0.0,
            np.where(sig > 0.0, _rho(diff / np.where(sig > 0.0, sig, 1.0)), 1.0),
        )
        term = np.where(shared, term, 0.0)
        n_shared = np.maximum(shared.sum(axis=1), 1)
        dists += np.where(has, term.sum(axis=1) / n_shared, 0.0)
        counts += has.astype(int)
    if not counts.any():
        raise ValueError("no shared observed feature group — distance undefined")
    # Entries sharing no group get the maximal ρ-saturated distance so
    # retrieval can never prefer an incomparable row.
    return np.asarray(np.where(counts > 0, dists / np.maximum(counts, 1), 1.0), dtype=float)


def normalised_statistical_distance(s_i: Array, s_j: Array, sigma: Array) -> float:
    """Eq. (1) distance between two 18-feature representations.

    ``sigma`` is the per-feature std across active memory entries; features
    missing from either representation are dropped group-wise. Fails closed
    (ValueError) when no group shares an observed feature.
    """
    q = as_named_1d("s_i", np.asarray(s_i, dtype=float))
    j = as_named_1d("s_j", np.asarray(s_j, dtype=float))
    sg = as_named_1d("sigma", np.asarray(sigma, dtype=float))
    if not (q.size == j.size == sg.size == N_FEATURES):
        raise ValueError(f"representations and sigma must all have length {N_FEATURES}")
    if np.any(sg < 0.0) or not np.all(np.isfinite(sg)):
        raise ValueError("sigma must be finite and non-negative")
    both = np.isfinite(q) & np.isfinite(j)
    if not any(any(both[list(span)]) for span in _GROUP_SPANS):
        raise ValueError("no shared observed feature group — distance undefined")
    return float(_distance_row(q, j.reshape(1, -1), sg)[0])


def statistical_distance_row(query: Array, entries: Array, sigma: Array) -> Array:
    """Eq. (1) distances from ``query`` to every active entry.

    ``entries`` is the (M, 18) matrix of stored representations. Entries
    sharing no observed feature group with the query receive distance 1.0
    (the ρ-saturated maximum) so retrieval can never prefer an incomparable
    entry. Empty ``entries`` returns an empty vector.
    """
    q = as_named_1d("query", np.asarray(query, dtype=float))
    e = np.asarray(entries, dtype=float)
    sg = as_named_1d("sigma", np.asarray(sigma, dtype=float))
    if q.size != N_FEATURES or sg.size != N_FEATURES:
        raise ValueError(f"query and sigma must have length {N_FEATURES}")
    if e.ndim != 2 or e.shape[1] != N_FEATURES:
        raise ValueError(f"entries must be (M, {N_FEATURES})")
    if np.any(sg < 0.0) or not np.all(np.isfinite(sg)):
        raise ValueError("sigma must be finite and non-negative")
    if e.shape[0] == 0:
        return np.asarray([], dtype=float)
    return _distance_row(q, e, sg)


# ---------------------------------------------------------------------------
# 3. Episodic memory (§3.2)
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    """One completed forecasting instance held in episodic memory.

    ``point_errors`` is the per-tool point forecast error (MAE over the
    horizon); ``point_ranks`` its average-tie rank (1 = best);
    ``prob_errors`` the per-tool probabilistic error (CRPS) auxiliary
    feedback, NaN-filled when no quantile feedback was supplied;
    ``best_tools`` is the tied-best-rank tool set used by the eq. (2)
    partition key; ``retention`` is the running mean of the entry's eq. (2)
    contributions over retention updates.
    """

    features: Array
    point_errors: Array
    point_ranks: Array
    prob_errors: Array
    invoked_tool: int
    best_tools: frozenset[int]
    entry_id: int
    retention: float = 0.0
    retention_count: int = 0


class EpisodicMemory:
    """Two-pool episodic memory with eq. (2) retention gating.

    Newly completed instances enter the recent pool; when it exceeds
    ``recent_capacity`` the oldest entry is promoted to the long-term pool if
    capacity remains, otherwise it replaces the lowest-retention long-term
    entry only when strictly more distinctive (higher retention). Retrieval
    returns the ``retrieve_k`` active entries nearest to the query under the
    eq. (1) distance, with σ computed across all active entries.
    """

    def __init__(
        self,
        *,
        recent_capacity: int = RECENT_CAPACITY,
        longterm_capacity: int = LONGTERM_CAPACITY,
        retrieve_k: int = RETRIEVE_K,
    ) -> None:
        self.recent_capacity = _positive_int("recent_capacity", recent_capacity)
        self.longterm_capacity = _positive_int("longterm_capacity", longterm_capacity)
        self.retrieve_k = _positive_int("retrieve_k", retrieve_k)
        self._recent: list[MemoryEntry] = []
        self._longterm: list[MemoryEntry] = []
        self._next_id = 0

    @property
    def n_active(self) -> int:
        return len(self._recent) + len(self._longterm)

    @property
    def recent_size(self) -> int:
        return len(self._recent)

    @property
    def longterm_size(self) -> int:
        return len(self._longterm)

    def _active(self) -> list[MemoryEntry]:
        return self._recent + self._longterm

    def feature_sigma(self) -> Array:
        """Per-feature std across active entries (all-zero on an empty memory)."""
        active = self._active()
        if not active:
            return np.zeros(N_FEATURES, dtype=float)
        mat = np.stack([e.features for e in active])
        sig = np.zeros(N_FEATURES, dtype=float)
        for q in range(N_FEATURES):
            col = mat[:, q]
            col = col[np.isfinite(col)]
            if col.size >= 2:
                sig[q] = float(np.std(col))
        return sig

    def retrieve(self, query: Array, k: int | None = None) -> list[MemoryEntry]:
        """The ``k`` (default ``retrieve_k``) nearest active entries."""
        active = self._active()
        if not active:
            return []
        kk = self.retrieve_k if k is None else _positive_int("k", k)
        mat = np.stack([e.features for e in active])
        d = statistical_distance_row(
            as_named_1d("query", np.asarray(query, dtype=float)), mat, self.feature_sigma()
        )
        order = np.argsort(d, kind="mergesort")[: min(kk, len(active))]
        return [active[int(i)] for i in order]

    def _update_retention(self, features: Array, context: Sequence[MemoryEntry]) -> None:
        """Apply the eq. (2) contributions of one completion to ``context``.

        Active entries are partitioned by ``(invoked_tool, best_tools)``; per
        partition the entry nearest to the completed instance's
        representation earns the margin ``d(s_i, j_C^(2)) − d(s_i, j_C^(1))``
        (``1 − d(s_i, j_C^(1))`` when the partition is a singleton) — but only
        when that nearest entry was itself in the retrieved context. Every
        context entry's retention is the running mean of its contributions
        (0 when it was not its partition's nearest entry).
        """
        active = self._active()
        context_ids = {e.entry_id for e in context}
        if not active or not context_ids:
            return
        sigma = self.feature_sigma()
        mat = np.stack([e.features for e in active])
        dists = _distance_row(features, mat, sigma)
        partitions: dict[tuple[int, frozenset[int]], list[int]] = {}
        for idx, e in enumerate(active):
            partitions.setdefault((e.invoked_tool, e.best_tools), []).append(idx)
        delta: dict[int, float] = {}
        for members in partitions.values():
            order = np.argsort(dists[np.asarray(members)], kind="mergesort")
            j1 = members[int(order[0])]
            if active[j1].entry_id in context_ids:
                if len(members) > 1:
                    j2 = members[int(order[1])]
                    delta[active[j1].entry_id] = float(dists[j2] - dists[j1])
                else:
                    delta[active[j1].entry_id] = float(1.0 - dists[j1])
        for e in active:
            if e.entry_id in context_ids:
                e.retention = (e.retention * e.retention_count + delta.get(e.entry_id, 0.0)) / (
                    e.retention_count + 1
                )
                e.retention_count += 1

    def complete_instance(
        self,
        features: Array,
        invoked_tool: int,
        point_errors: Array,
        prob_errors: Array | None = None,
        context: Sequence[MemoryEntry] = (),
    ) -> MemoryEntry:
        """Fold one completed instance into memory (§3.2 order).

        First updates the retention values of the entries that formed the
        decision context, then inserts the new entry into the recent pool and
        applies retention-gated promotion when the pool overflows.
        """
        s = as_named_1d("features", np.asarray(features, dtype=float))
        if s.size != N_FEATURES:
            raise ValueError(f"features must have length {N_FEATURES}")
        errs = _finite_vector("point_errors", point_errors)
        if np.any(errs < 0.0):
            raise ValueError("point_errors must be non-negative")
        n_tools = errs.size
        if isinstance(invoked_tool, bool) or not isinstance(invoked_tool, int):
            raise ValueError("invoked_tool must be an int tool index")
        if not 0 <= invoked_tool < n_tools:
            raise ValueError("invoked_tool out of range")
        if prob_errors is None:
            perrs = np.full(n_tools, np.nan, dtype=float)
        else:
            perrs = _finite_vector("prob_errors", prob_errors, n_tools)
            if np.any(perrs < 0.0):
                raise ValueError("prob_errors must be non-negative")
        ranks = midrank(errs)
        best = frozenset(int(i) for i in np.flatnonzero(ranks == ranks.min()))

        self._update_retention(s, context)

        entry = MemoryEntry(
            features=s.copy(),
            point_errors=errs.copy(),
            point_ranks=ranks,
            prob_errors=perrs,
            invoked_tool=invoked_tool,
            best_tools=best,
            entry_id=self._next_id,
        )
        self._next_id += 1
        self._recent.append(entry)
        if len(self._recent) > self.recent_capacity:
            candidate = self._recent.pop(0)
            if len(self._longterm) < self.longterm_capacity:
                self._longterm.append(candidate)
            else:
                weakest = min(
                    range(len(self._longterm)),
                    key=lambda i: (self._longterm[i].retention, self._longterm[i].entry_id),
                )
                if candidate.retention > self._longterm[weakest].retention:
                    self._longterm[weakest] = candidate
        return entry


# ---------------------------------------------------------------------------
# 4. Online policy learning (§3.3, Appendix C) — numpy MLP + AdamW
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Hyperparameters of the online tool-ranking policy (paper Table 6)."""

    n_tools: int
    history_window: int  # N: the paper uses the 8H most recent values
    hidden_dims: tuple[int, ...] = POLICY_HIDDEN_DIMS
    feedback_capacity: int = FEEDBACK_CAPACITY
    warmup: int = POLICY_WARMUP
    init_epochs: int = POLICY_INIT_EPOCHS
    update_every: int = POLICY_UPDATE_EVERY
    update_batches: int = POLICY_UPDATE_BATCHES
    batch_size: int = POLICY_BATCH_SIZE
    lr: float = POLICY_LR
    weight_decay: float = POLICY_WEIGHT_DECAY
    seed: int = POLICY_SEED

    def __post_init__(self) -> None:
        _positive_int("n_tools", self.n_tools)
        if self.n_tools < 2:
            raise ValueError("n_tools must be >= 2 for a pairwise ranking loss")
        _positive_int("history_window", self.history_window)
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one layer")
        for h in self.hidden_dims:
            _positive_int("hidden_dim", h)
        _positive_int("feedback_capacity", self.feedback_capacity)
        _positive_int("warmup", self.warmup)
        _positive_int("init_epochs", self.init_epochs)
        _positive_int("update_every", self.update_every)
        _positive_int("update_batches", self.update_batches)
        _positive_int("batch_size", self.batch_size)
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an int")


def _softplus(x: Array) -> Array:
    return np.logaddexp(0.0, x)


def _pairwise_loss_and_grad(u: Array, ell: Array) -> tuple[Array, Array]:
    """Eq. (3) loss per record and its gradient w.r.t. the score vector.

    ``u``: (B, F) scores; ``ell``: (B, F) observed point errors. Returns the
    per-record loss (B,) and ``dL/du`` (B, F) with the 1/C(F,2) pair mean
    applied; the minibatch mean over records is applied by the caller.
    Records with ℓ̄ = 0 carry all-zero pair weights and contribute 0.
    """
    n = u.shape[1]
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    n_pairs = float(len(pairs))
    losses = np.zeros(u.shape[0], dtype=float)
    grad = np.zeros_like(u)
    ell_bar = ell.mean(axis=1)
    safe_bar = np.where(ell_bar > 0.0, ell_bar, 1.0)
    for p, q in pairs:
        w = np.where(ell_bar > 0.0, np.abs(ell[:, p] - ell[:, q]) / safe_bar, 0.0)
        s = np.sign(ell[:, q] - ell[:, p])
        margin = s * (u[:, p] - u[:, q])
        losses += w * _softplus(margin)
        coef = w * s * _sigmoid(margin)
        grad[:, p] += coef
        grad[:, q] -= coef
    return losses / n_pairs, grad / n_pairs


class OnlineToolRanker:
    """MLP tool-ranking policy trained online on the eq. (3) pairwise loss.

    ``observe`` maps a raw history to the paper's ``z = [normalised last-N
    values, binary observation mask]`` input (median/std normalisation within
    the instance, NaNs zero-filled with mask 0). ``feedback`` appends the
    (z, ℓ) record to the FIFO pool and follows the paper's schedule: initial
    training after ``warmup`` records (``init_epochs`` passes over the pool),
    thereafter ``update_batches`` uniform minibatches of ``batch_size`` every
    ``update_every`` records. ``predict_scores`` returns lower-is-better
    scores, or None before the policy has trained (the paper supplies no
    ranking during warm-up). A single seeded Generator drives weight init and
    all sampling, so the ranker is bit-deterministic.
    """

    def __init__(self, config: PolicyConfig) -> None:
        if not isinstance(config, PolicyConfig):
            raise ValueError("config must be a PolicyConfig")
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._pool_z: list[Array] = []
        self._pool_ell: list[Array] = []
        self._n_seen = 0
        self._n_updates = 0
        self._weights: list[Array] = []
        self._biases: list[Array] = []
        self._mw: list[Array] = []
        self._vw: list[Array] = []
        self._mb: list[Array] = []
        self._vb: list[Array] = []
        self._t = 0

    @property
    def ready(self) -> bool:
        """True once the predictor has been initialised and trained."""
        return len(self._weights) > 0

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def n_feedback(self) -> int:
        return self._n_seen

    @property
    def pool_size(self) -> int:
        return len(self._pool_z)

    def observe(self, history: Array) -> Array:
        """``z_i = [Normalise(x_i), Mask(x_i)]`` over the last N values."""
        x = as_named_1d("history", np.asarray(history, dtype=float))
        if x.size == 0 or not np.isfinite(x).any():
            raise ValueError("history must contain at least one observed value")
        n = self.config.history_window
        window = x[-n:] if x.size >= n else np.concatenate([np.full(n - x.size, np.nan), x])
        mask = np.isfinite(window).astype(float)
        obs = window[np.isfinite(window)]
        med = float(np.median(obs))
        sd = float(np.std(obs))
        normed = np.where(mask > 0.0, (window - med) / sd, 0.0) if sd > 0.0 else mask * 0.0
        return np.asarray(np.concatenate([normed, mask]), dtype=float)

    def _init_params(self) -> None:
        dims = [2 * self.config.history_window, *self.config.hidden_dims, self.config.n_tools]
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            scale = math.sqrt(2.0 / a)  # He init for the ReLU hidden layers
            self._weights.append(self._rng.normal(0.0, scale, size=(a, b)).astype(float))
            self._biases.append(np.zeros(b, dtype=float))
            self._mw.append(np.zeros((a, b), dtype=float))
            self._vw.append(np.zeros((a, b), dtype=float))
            self._mb.append(np.zeros(b, dtype=float))
            self._vb.append(np.zeros(b, dtype=float))

    def _forward(self, z: Array) -> tuple[Array, list[Array], list[Array]]:
        """Return (scores, pre-activations, activations) for a (B, 2N) batch."""
        a = z
        pres: list[Array] = []
        acts: list[Array] = [a]
        last = len(self._weights) - 1
        for i, (w, b) in enumerate(zip(self._weights, self._biases, strict=True)):
            zl = a @ w + b
            pres.append(zl)
            a = zl if i == last else np.maximum(zl, 0.0)
            acts.append(a)
        return a, pres, acts

    def _train_minibatch(self, zb: Array, eb: Array) -> float:
        """One AdamW step on a minibatch; returns the mean eq. (3) loss."""
        u, pres, acts = self._forward(zb)
        losses, du = _pairwise_loss_and_grad(u, eb)
        du = du / zb.shape[0]
        grads_w: list[Array] = []
        grads_b: list[Array] = []
        da = du
        for i in reversed(range(len(self._weights))):
            grads_w.insert(0, acts[i].T @ da)
            grads_b.insert(0, da.sum(axis=0))
            if i > 0:
                da = (da @ self._weights[i].T) * (pres[i - 1] > 0.0)
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr, wd = self.config.lr, self.config.weight_decay
        for i in range(len(self._weights)):
            for param, grad, m, v, decay in (
                (self._weights[i], grads_w[i], self._mw[i], self._vw[i], wd),
                (self._biases[i], grads_b[i], self._mb[i], self._vb[i], 0.0),
            ):
                m *= b1
                m += (1.0 - b1) * grad
                v *= b2
                v += (1.0 - b2) * grad * grad
                m_hat = m / (1.0 - b1**self._t)
                v_hat = v / (1.0 - b2**self._t)
                param -= lr * (m_hat / (np.sqrt(v_hat) + eps) + decay * param)
        self._n_updates += 1
        return float(losses.mean())

    def _epoch(self) -> float:
        """One pass over the feedback pool in seeded-shuffled minibatches."""
        order = self._rng.permutation(len(self._pool_z))
        zb_all = np.stack(self._pool_z)
        eb_all = np.stack(self._pool_ell)
        total, count = 0.0, 0
        for start in range(0, order.size, self.config.batch_size):
            idx = order[start : start + self.config.batch_size]
            total += self._train_minibatch(zb_all[idx], eb_all[idx])
            count += 1
        return total / max(count, 1)

    def _periodic(self) -> float:
        """``update_batches`` uniform minibatches of ``batch_size`` records."""
        total = 0.0
        zb_all = np.stack(self._pool_z)
        eb_all = np.stack(self._pool_ell)
        for _ in range(self.config.update_batches):
            idx = self._rng.choice(len(self._pool_z), size=self.config.batch_size, replace=True)
            total += self._train_minibatch(zb_all[idx], eb_all[idx])
        return total / self.config.update_batches

    def feedback(self, z: Array, point_errors: Array) -> None:
        """Append one completed instance's feedback and train on schedule."""
        zz = _finite_vector("z", z, 2 * self.config.history_window)
        ell = _finite_vector("point_errors", point_errors, self.config.n_tools)
        if np.any(ell < 0.0):
            raise ValueError("point_errors must be non-negative")
        self._pool_z.append(zz.copy())
        self._pool_ell.append(ell.copy())
        if len(self._pool_z) > self.config.feedback_capacity:
            self._pool_z.pop(0)
            self._pool_ell.pop(0)
        self._n_seen += 1
        if not self.ready:
            if self._n_seen >= self.config.warmup:
                self._init_params()
                for _ in range(self.config.init_epochs):
                    self._epoch()
        elif self._n_seen % self.config.update_every == 0:
            self._periodic()

    def predict_scores(self, history_or_z: Array) -> Array | None:
        """Lower-is-better per-tool scores; None until the policy is trained.

        A vector of length ``2 * history_window`` is taken as an already
        encoded ``z``; anything else is passed through ``observe`` first.
        """
        v = as_named_1d("history_or_z", np.asarray(history_or_z, dtype=float))
        z = v if v.size == 2 * self.config.history_window else self.observe(v)
        if not np.all(np.isfinite(z)):
            raise ValueError("policy input must be finite")
        if not self.ready:
            return None
        u, _, _ = self._forward(z.reshape(1, -1))
        return np.asarray(u[0], dtype=float)


# ---------------------------------------------------------------------------
# 5. GIFT-Eval protocol helpers (§4.1)
# ---------------------------------------------------------------------------


def seasonal_naive_forecast(history: Array, horizon: int, season: int = 1) -> Array:
    """Seasonal-naive point forecast: repeat the last observed seasonal cycle.

    Only observed values are used (the paper linearly interpolates missing
    values for Seasonal Naive; callers needing that must supply an
    already-interpolated history — no interpolation is hidden here).
    """
    h = _positive_int("horizon", horizon)
    m = _positive_int("season", season)
    x = as_named_1d("history", np.asarray(history, dtype=float))
    obs = x[np.isfinite(x)]
    if obs.size < m:
        raise ValueError("fewer observed values than the season length")
    tail = obs[-m:]
    reps = int(math.ceil(h / m))
    return np.asarray(np.tile(tail, reps)[:h], dtype=float)


def seasonal_naive_scale(history: Array, season: int = 1) -> float:
    """In-sample seasonal-naive MAE scale: mean |x_t − x_{t−m}| over the history.

    Only position-exact pairs with both values observed contribute (missing
    values are never interpolated silently). Fails closed when no valid pair
    exists or the scale is zero — MASE is then undefined, not 0.
    """
    m = _positive_int("season", season)
    x = as_named_1d("history", np.asarray(history, dtype=float))
    if x.size <= m:
        raise ValueError("history must be longer than the season")
    diffs = np.abs(x[m:] - x[:-m])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        raise ValueError("no observed seasonal pair in history")
    scale = float(np.mean(diffs))
    if scale <= 0.0:
        raise ValueError("seasonal-naive scale is zero — MASE undefined")
    return scale


def mase_score(target: Array, forecast: Array, scale: float) -> float:
    """MASE = mean |target − forecast| / in-sample seasonal-naive scale."""
    y = _finite_vector("target", target)
    f = _finite_vector("forecast", forecast, y.size)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    return float(np.mean(np.abs(y - f)) / scale)


def normalised_geometric_mean(
    scores_by_method: Mapping[str, Array], *, baseline: str
) -> dict[str, float]:
    """Per-config normalisation by the baseline, geometric mean across configs.

    Each method's per-configuration score is divided by the baseline's score
    for the same configuration (Seasonal Naive ≡ 1.0, matching the paper's
    norm. columns); the geometric mean over configurations weights all
    configurations equally, as in §4.1. A zero or negative baseline fails
    closed — the protocol is then undefined, not silently 1.0.
    """
    if baseline not in scores_by_method:
        raise ValueError(f"baseline {baseline!r} not in scores")
    base = np.asarray(scores_by_method[baseline], dtype=float).reshape(-1)
    if base.size == 0 or not np.all(np.isfinite(base)):
        raise ValueError("baseline scores must be non-empty and finite")
    if np.any(base <= 0.0):
        raise ValueError("baseline scores must be strictly positive for normalisation")
    out: dict[str, float] = {}
    for name, scores in scores_by_method.items():
        s = np.asarray(scores, dtype=float).reshape(-1)
        if s.size != base.size or not np.all(np.isfinite(s)):
            raise ValueError(f"{name!r} scores must be finite and aligned with baseline")
        if np.any(s < 0.0):
            raise ValueError(f"{name!r} scores must be non-negative")
        if np.any(s == 0.0):
            out[name] = 0.0
        else:
            out[name] = float(np.exp(np.mean(np.log(s / base))))
    return out


def mean_config_ranks(scores_by_method: Mapping[str, Array]) -> dict[str, float]:
    """Within-config ranks (ties share the mean) averaged over configs.

    §4.1 rank columns: systems are ranked per configuration and metric,
    tied systems receive the mean of their ranks, and the resulting ranks are
    averaged arithmetically across configurations.
    """
    names = list(scores_by_method)
    if not names:
        raise ValueError("scores_by_method must be non-empty")
    cols = [np.asarray(scores_by_method[nm], dtype=float).reshape(-1) for nm in names]
    n = cols[0].size
    if n == 0 or any(c.size != n for c in cols):
        raise ValueError("all methods must share the same non-empty config count")
    if any(not np.all(np.isfinite(c)) for c in cols):
        raise ValueError("scores must be finite")
    mat = np.stack(cols, axis=1)  # (n_configs, n_methods)
    ranks = np.apply_along_axis(midrank, 1, mat)
    return {nm: float(ranks[:, j].mean()) for j, nm in enumerate(names)}


def self_evolution_curve(agent_errors: Array, base_errors: Array) -> Array:
    """§5.2 cumulative relative gain: running-mean error reduction vs base.

    ``curve[t] = (mean(e_base[:t+1]) − mean(e_agent[:t+1])) /
    mean(e_base[:t+1]) × 100`` for each processed instance t. Positive means
    the agent's cumulative mean error is below the feedback-free base
    controller's — the paper's percentage-of-base reduction.
    """
    a = _finite_vector("agent_errors", agent_errors)
    b = _finite_vector("base_errors", base_errors, a.size)
    cum_a = np.cumsum(a) / np.arange(1, a.size + 1)
    cum_b = np.cumsum(b) / np.arange(1, b.size + 1)
    if np.any(cum_b <= 0.0):
        raise ValueError("base cumulative mean must stay positive")
    return np.asarray((cum_b - cum_a) / cum_b * 100.0, dtype=float)


# ---------------------------------------------------------------------------
# 6. Online benchmark (Appendix B protocol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkInstance:
    """One forecasting instance: history, target, and per-tool forecasts.

    ``point_forecasts`` is the (n_tools, horizon) point-forecast panel;
    ``quantile_forecasts`` the optional (n_tools, n_quantiles, horizon)
    probabilistic panel — the auxiliary probabilistic feedback of §3.2 and
    the CRPS metric of §4.1. All tools' forecasts must be produced before the
    target is revealed, exactly as in the paper's protocol.
    """

    history: Array
    target: Array
    point_forecasts: Array
    quantile_forecasts: Array | None = None
    covariates: Array | None = None


def _validate_instance(
    inst: BenchmarkInstance, n_tools: int
) -> tuple[Array, Array, Array, Array | None]:
    hist = as_named_1d("history", np.asarray(inst.history, dtype=float))
    if hist.size == 0 or not np.isfinite(hist).any():
        raise ValueError("instance history must contain an observed value")
    target = _finite_vector("target", inst.target)
    pf = np.asarray(inst.point_forecasts, dtype=float)
    if pf.ndim != 2 or pf.shape != (n_tools, target.size) or not np.all(np.isfinite(pf)):
        raise ValueError(f"point_forecasts must be finite ({n_tools}, horizon)")
    qf: Array | None = None
    if inst.quantile_forecasts is not None:
        q = np.asarray(inst.quantile_forecasts, dtype=float)
        if q.ndim != 3 or q.shape[0] != n_tools or q.shape[2] != target.size:
            raise ValueError("quantile_forecasts must be (n_tools, n_quantiles, horizon)")
        if not np.all(np.isfinite(q)):
            raise ValueError("quantile_forecasts must be finite")
        qf = q
    return hist, target, pf, qf


def _memory_vote(context: Sequence[MemoryEntry], n_tools: int) -> int | None:
    """Deterministic reduction of retrieved-evidence guidance.

    The paper's LLM weighs the retrieved entries' per-tool observed errors;
    the deterministic analogue picks the tool with the lowest mean point
    error over the retrieved set (ties → lowest index). None when the
    context is empty.
    """
    if not context:
        return None
    mean_err = np.zeros(n_tools, dtype=float)
    for e in context:
        mean_err += e.point_errors
    return int(np.argmin(mean_err / len(context)))


def fase_benchmark(
    configs: Sequence[Sequence[BenchmarkInstance]] | Sequence[BenchmarkInstance],
    *,
    tool_names: Sequence[str] | None = None,
    baseline_index: int = 0,
    ensemble_indices: Sequence[int] | None = None,
    season: int = 1,
    variant: str = "fase",
    recent_capacity: int = RECENT_CAPACITY,
    longterm_capacity: int = LONGTERM_CAPACITY,
    retrieve_k: int = RETRIEVE_K,
    policy: PolicyConfig | None = None,
    seed: int = POLICY_SEED,
    quantile_levels: Array | None = None,
) -> dict[str, float]:
    """Replay the paper's online protocol and return flat summary metrics.

    ``configs`` is one list of :class:`BenchmarkInstance` (a single task
    configuration) or a list of such lists (multiple configurations; memory
    and policy are reinitialised per configuration exactly as in §4.1). For
    each instance the agent selects a tool using the ``variant``'s feedback
    state, all tools are scored on the revealed target (MAE per instance as
    the point error ``ℓ``, CRPS as auxiliary probabilistic feedback), and
    memory plus the feedback pool are updated before the next instance.

    ``variant`` selects the ablation arm of Table 3: ``"fase"`` (memory +
    policy), ``"memory_only"``, ``"policy_only"``, or ``"base"`` (no feedback
    → the fixed ``baseline_index`` tool; the deterministic reduction of the
    paper's LLM base agent, which sees no feedback).

    Returns a ``dict[str, float]`` with per-method normalised scores and mean
    ranks (GIFT-Eval §4.1: per-config normalisation by the ``baseline_index``
    tool's score, geometric mean across configs), self-evolution curve
    endpoints vs the ``base`` arm (§5.2), and bookkeeping (pool sizes, policy
    update count). All keys are proper-score or honest error diagnostics —
    no P&L-family content.
    """
    if variant not in ("fase", "memory_only", "policy_only", "base"):
        raise ValueError("variant must be fase|memory_only|policy_only|base")
    m = _positive_int("season", season)
    taus = (
        QUANTILE_LEVELS
        if quantile_levels is None
        else as_named_1d("quantile_levels", np.asarray(quantile_levels, dtype=float))
    )
    if taus.size < 1 or not np.all(np.isfinite(taus)) or np.any((taus <= 0.0) | (taus >= 1.0)):
        raise ValueError("quantile_levels must lie in (0, 1)")

    if len(configs) == 0:
        raise ValueError("configs must be non-empty")
    config_list: list[Sequence[BenchmarkInstance]] = (
        [configs]  # type: ignore[list-item]
        if isinstance(configs[0], BenchmarkInstance)
        else list(configs)  # type: ignore[arg-type]
    )
    if any(len(c) == 0 for c in config_list):
        raise ValueError("each configuration must contain at least one instance")

    n_tools = int(np.asarray(config_list[0][0].point_forecasts).shape[0])
    if n_tools < 2:
        raise ValueError("at least two tools are required")
    h = int(np.asarray(config_list[0][0].target).size)
    names = list(tool_names) if tool_names is not None else [f"tool_{i}" for i in range(n_tools)]
    if len(names) != n_tools or len(set(names)) != n_tools:
        raise ValueError("tool_names must match n_tools and be unique")
    if (
        isinstance(baseline_index, bool)
        or not isinstance(baseline_index, int)
        or not 0 <= baseline_index < n_tools
    ):
        raise ValueError("baseline_index out of range")
    ens = (
        [i for i in range(n_tools) if i != baseline_index]
        if ensemble_indices is None
        else [int(i) for i in ensemble_indices]
    )
    if len(ens) < 1 or any(not 0 <= i < n_tools for i in ens):
        raise ValueError("ensemble_indices must be a non-empty subset of tool indices")

    use_memory = variant in ("fase", "memory_only")
    use_policy = variant in ("fase", "policy_only")

    metric_names = list(names) + ["uniform_ensemble", "agent"]
    mae_rows: dict[str, list[float]] = {nm: [] for nm in metric_names}
    rmse_rows: dict[str, list[float]] = {nm: [] for nm in metric_names}
    mase_rows: dict[str, list[float]] = {nm: [] for nm in metric_names}
    crps_rows: dict[str, list[float]] = {nm: [] for nm in metric_names}
    have_crps = True

    agent_err_all: list[float] = []
    base_err_all: list[float] = []
    policy_updates = 0
    pool_longterm = 0
    pool_recent = 0
    mean_ctx_dist = 0.0
    n_ctx = 0

    for cfg in config_list:
        mem = EpisodicMemory(
            recent_capacity=recent_capacity,
            longterm_capacity=longterm_capacity,
            retrieve_k=retrieve_k,
        )
        pconf = policy or PolicyConfig(n_tools=n_tools, history_window=8 * h, seed=seed)
        if pconf.n_tools != n_tools or pconf.history_window != 8 * h:
            raise ValueError("policy config must match n_tools and the 8·horizon window")
        ranker = OnlineToolRanker(pconf)

        for inst in cfg:
            hist, target, pf, qf = _validate_instance(inst, n_tools)
            if target.size != h or pf.shape != (n_tools, h):
                raise ValueError("instances must share one (n_tools, horizon) geometry")
            s = statistical_features(hist, covariates=inst.covariates)
            context = mem.retrieve(s) if use_memory else []
            z = ranker.observe(hist)
            scores = ranker.predict_scores(z) if use_policy else None

            if variant == "base":
                action = baseline_index
            elif variant == "memory_only":
                vote = _memory_vote(context, n_tools)
                action = baseline_index if vote is None else vote
            elif variant == "policy_only":
                action = int(np.argmin(scores)) if scores is not None else baseline_index
            else:  # fase: policy guidance, memory vote before warm-up
                vote = _memory_vote(context, n_tools)
                action = (
                    int(np.argmin(scores))
                    if scores is not None
                    else (baseline_index if vote is None else vote)
                )

            point_err = np.asarray([float(np.mean(np.abs(target - pf[j]))) for j in range(n_tools)])
            if not np.all(np.isfinite(point_err)):
                raise ValueError("point forecast errors must be finite")
            prob_err = (
                np.asarray(
                    [
                        float(crps_from_quantiles(target, np.asarray(qf[j].T), taus))
                        for j in range(n_tools)
                    ]
                )
                if qf is not None
                else None
            )
            if qf is None:
                have_crps = False

            if context:
                mat = np.stack([e.features for e in context])
                mean_ctx_dist += float(
                    np.sum(statistical_distance_row(s, mat, mem.feature_sigma()))
                )
                n_ctx += len(context)
            if use_memory:
                mem.complete_instance(s, action, point_err, prob_err, context)
            if use_policy:
                ranker.feedback(z, point_err)

            ens_point = pf[ens].mean(axis=0)
            ens_err = float(np.mean(np.abs(target - ens_point)))
            scale = seasonal_naive_scale(hist, m)
            agent_mae = float(np.mean(np.abs(target - pf[action])))
            agent_err_all.append(agent_mae)
            base_err_all.append(float(np.mean(np.abs(target - pf[baseline_index]))))
            for j, nm in enumerate(names):
                mae_rows[nm].append(float(np.mean(np.abs(target - pf[j]))))
                rmse_rows[nm].append(float(np.sqrt(np.mean((target - pf[j]) ** 2))))
                mase_rows[nm].append(float(np.mean(np.abs(target - pf[j])) / scale))
                if prob_err is not None:
                    crps_rows[nm].append(float(prob_err[j]))
            mae_rows["agent"].append(agent_mae)
            rmse_rows["agent"].append(float(np.sqrt(np.mean((target - pf[action]) ** 2))))
            mase_rows["agent"].append(agent_mae / scale)
            mae_rows["uniform_ensemble"].append(ens_err)
            rmse_rows["uniform_ensemble"].append(float(np.sqrt(np.mean((target - ens_point) ** 2))))
            mase_rows["uniform_ensemble"].append(ens_err / scale)
            if qf is not None:
                ens_q = qf[ens].mean(axis=0)
                crps_rows["uniform_ensemble"].append(
                    float(crps_from_quantiles(target, np.asarray(ens_q.T), taus))
                )
                crps_rows["agent"].append(
                    float(crps_from_quantiles(target, np.asarray(qf[action].T), taus))
                )
        pool_longterm += mem.longterm_size
        pool_recent += mem.recent_size
        policy_updates += ranker.n_updates

    # Per-config aggregate = mean over the config's instances, then §4.1
    # normalisation + geometric mean across configs (identity at n_configs=1).
    split_sizes = [len(c) for c in config_list]
    base_name = names[baseline_index]

    def _per_config(rows: dict[str, list[float]]) -> dict[str, Array]:
        out: dict[str, Array] = {}
        for nm, vals in rows.items():
            arr = np.asarray(vals, dtype=float)
            means = np.empty(len(split_sizes), dtype=float)
            off = 0
            for c_i, n in enumerate(split_sizes):
                seg = arr[off : off + n]
                if seg.size == 0 or not np.all(np.isfinite(seg)):
                    raise ValueError("per-config scores must be finite")
                means[c_i] = float(np.mean(seg))
                off += n
            out[nm] = means
        return out

    def _aggregate(rows: dict[str, list[float]], metric: str) -> dict[str, float]:
        per_cfg = _per_config(rows)
        norm = normalised_geometric_mean(per_cfg, baseline=base_name)
        ranks = mean_config_ranks(per_cfg)
        out: dict[str, float] = {}
        for nm in rows:
            out[f"{nm}_norm_{metric}"] = float(norm[nm])
            out[f"{nm}_rank_{metric}"] = float(ranks[nm])
        return out

    result = _aggregate(mae_rows, "mae")
    result.update(_aggregate(rmse_rows, "rmse"))
    result.update(_aggregate(mase_rows, "mase"))
    if have_crps and crps_rows["agent"]:
        result.update(_aggregate(crps_rows, "crps"))

    curve = self_evolution_curve(np.asarray(agent_err_all), np.asarray(base_err_all))
    n_inst = int(curve.size)
    result["self_evolution_gain_early"] = float(curve[max(0, n_inst // 4 - 1)])
    result["self_evolution_gain_mid"] = float(curve[max(0, n_inst // 2 - 1)])
    result["self_evolution_gain_final"] = float(curve[-1])
    result["self_evolution_gain_mean"] = float(np.mean(curve))

    result["n_configs"] = float(len(config_list))
    result["n_instances"] = float(n_inst)
    result["n_policy_updates"] = float(policy_updates)
    result["policy_ready"] = float(policy_updates > 0)
    result["memory_recent_size"] = float(pool_recent)
    result["memory_longterm_size"] = float(pool_longterm)
    result["mean_context_distance"] = float(mean_ctx_dist / max(n_ctx, 1))
    return result
