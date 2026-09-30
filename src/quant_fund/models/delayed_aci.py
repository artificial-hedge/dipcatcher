"""τ-Delayed adaptive conformal inference (τ-DACI) and delay-to-memory diagnostics.

Adaptive conformal inference when the coverage outcome of a forecast issued at
time t is only observed τ steps later, so the ACI level update runs on delayed
feedback (El Halabi & Brandt, 2026, "Adaptive Conformal Inference Under
Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic",
arXiv:2609.07251 [stat.ME]). This is the adaptive-level layer for exactly the
wave-12 multi-horizon setting of ``quant_fund.models.enbpi_multihorizon``,
where a horizon-h band is evaluated when ``y_{t+h}`` is revealed h steps later;
it composes with (does not modify) that module and with the wave-11 scalar
controller ``quant_fund.models.conformal_pid``.

Recursion (paper Eq. (6), Algorithm 2). With target miscoverage α, step size
γ > 0, and err_t = 1{Y_t ∉ Ĉ_t^τ(α_t)} the miscoverage of the set issued τ
steps earlier:

    α_{t+τ} = α_t + γ (α − err_t).

The first τ sets are issued at the initial level α (empty feedback queue).
Interleaved-sequences view (paper Eq. (7)-(8), Sec. 4.1): writing
t_θ(m) = t_0 + θ + (m−1)τ, every time index belongs to exactly one phase
θ ∈ {1..τ}, and within a phase the delayed recursion is an ordinary one-step
ACI update in the phase-local index m:

    α_{m+1}^{(θ)} = α_m^{(θ)} + γ (α − e_m^{(θ)}).

``DelayedACI`` implements the queue form of Algorithm 2 exactly and exposes
the phase decomposition (``phases_``) — the construction the paper uses to
carry the standard ACI analysis over phase-by-phase.

Guarantees and diagnostics (all worst-case / descriptive — see Honesty):

- Boundedness (paper Eq. (10), App. A.1, following Gibbs & Candès, 2021,
  NeurIPS 34, pp. 1660-1672, arXiv:2106.00170, Lemma 4.1): for the unclipped
  recursion α_m^{(θ)} ∈ [−γ, 1+γ]; boundary levels force degenerate bands
  (α < 0 → all of ℝ, e = 0; α > 1 → empty set, e = 1) that push the level
  back inside. This implementation instead clips to a box inside (0, 1)
  (default (1e-3, 1 − 1e-3), the convention of
  ``quant_fund.models.conformal.AdaptiveConformal``); the phase-wise
  telescoping behind the coverage bound is exact whenever clipping is
  inactive (``clip_updates_`` counts violations).
- Finite-sample long-run coverage bound (paper Eq. (11)/(33), App. A.2):

      |(1/T) Σ_{t=1..T} err_t − α|
          ≤ (1/(γT)) Σ_{θ=1..τ} max{α_1^{(θ)}, 1 − α_1^{(θ)}} + τ/T,

  distribution-free given the recursion; with the default initialization
  α_1^{(θ)} = α this is τ·max{α, 1−α}/(γT) + τ/T — explicit linear τ
  dependence. At τ = 1 it reduces exactly to Proposition 4.1 of Gibbs &
  Candès (2021), as the paper states. ``long_run_coverage_bound`` computes
  it; ``DelayedACI.coverage_bound`` uses the recorded phase initializations.
- Approximate marginal coverage bound (paper Eq. (12)/(50), App. A.3), under
  a stationary hidden-Markov environment A_t, L_Lip-Lipschitz conditional
  miscoverage M(·|a), and a unique optimal level α*_a per environment:

      E[(M(α_t|A_t) − α)²] ≤ (L_Lip (1+γ)/γ) E|α*_{A_{t+τ}} − α*_{A_t}|
                             + (L_Lip/2) γ,

  relating coverage deviation to environment change *across the horizon*
  (the delay enters through E|α*_{A_{t+τ}} − α*_{A_t}|) and to the
  adaptation rate γ. The bound-minimizing step size is
  γ_opt = sqrt(2 E|α*_{A_{t+τ}} − α*_{A_t}|) (paper Sec. 4.4). At τ = 1 this
  is the approximate marginal bound of Gibbs & Candès (2021, Thm. 4.2).
- Delay-to-memory ratio (paper Eq. (20), (23), (26)-(27)): r = τ/L, where L
  is the decay time scale of the signal driving non-exchangeability —
  L = −1/log φ for AR(1) residual autocorrelation ρ(k) = φ^k;
  L_vol = −1/log(α_g + β) for GARCH(1,1) squared-residual dependence (the
  ARMA(1,1) representation of ε_t² decays at rate α_g + β, paper Eq. (22));
  L_regime = −1/log|λ₂| with λ₂ = p₀₀ + p₁₁ − 1 for two-state Markov
  switching (state autocorrelation is exactly λ₂^k, paper App. B.1). The
  autocorrelation surviving the delay is e^{−r}. ``estimate_delay_to_memory``
  estimates L from a stream by regressing log-autocorrelation on lag over the
  usable lags (a constant attenuation — e.g. signal-plus-noise — shifts the
  intercept, not the slope, so the decay *rate* survives); for GARCH-type
  scale dependence use feature='square' (squared residuals).
- Curve-collapse diagnostic (paper Eq. (17), Sec. 6): pool a performance
  value (the paper uses the interval score at the selected γ) across
  persistence configurations, bin a candidate horizontal axis into
  logarithmically spaced bins, and compare the average within-bin standard
  deviation S_r under r = τ/L against S_τ under the raw delay; the reported
  reduction 100(1 − S_r/S_τ) was ≈79% for AR(1), ≈46% for GARCH(1,1), and
  ≈9-27% for Markov switching (paper Table 6) — r organizes performance
  strongly under AR(1) and progressively less under the other families.

Evaluation metrics follow the paper: empirical coverage (Eq. (13)),
worst-case local coverage error LCE_k (Eq. (14)), and the interval score
(Eq. (15)-(16)) — a proper score for interval forecasts. ``run_delayed_conformal``
implements Algorithms 1+2 on a bare seeded residual stream (the paper's Sec. 6
simulation setting: signed residual scores S_t, a sliding window of the most
recent R scores, and the two-sided equal-tail band
[Q̂_t(α_t/2), Q̂_t(1 − α_t/2)] around the implicit point forecast), and
``adaptation_duration`` implements the Sec. 6.5 first-dip recovery measure
used for the abrupt-shift families.

Honesty (AGENTS.md contract): the long-run bound is a worst-case
time-average statement with explicit τ slack — not a distributional or
per-step guarantee; the marginal bound is approximate, resting on HMM
stationarity, Lipschitz continuity, and optimal-level existence; the
delay-to-memory ratio and its stream estimate are DESCRIPTIVE diagnostics
(they organize simulated performance, with strength depending on the
dependence family — they guarantee nothing); the log-ACF decay estimate is
attenuation-robust in rate but biased on short or near-white streams, where
it fails closed rather than fabricate a memory scale. All results here come
from seeded SYNTHETIC streams and are correctness tests, never market
evidence; only coverage / proper-score diagnostics are exposed; no
live-trading claims.

Conventions: numpy core, fail-closed edges (ValueError), deterministic given
the residual stream (the controller holds no RNG). τ = 1 reproduces the
repo's ``AdaptiveConformal`` level recursion bit-for-bit (pinned in
tests/unit/models/test_delayed_aci.py).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import conformal_quantile

Array = NDArray[np.float64]

__all__ = [
    "DelayedACI",
    "DelayedConformalResult",
    "DelayToMemoryEstimate",
    "adaptation_duration",
    "curve_collapse_reduction",
    "delay_to_memory_ratio",
    "estimate_delay_to_memory",
    "long_run_coverage_bound",
    "marginal_coverage_bound",
    "memory_length_ar1",
    "memory_length_garch",
    "memory_length_markov",
    "optimal_gamma",
    "run_delayed_conformal",
    "worst_case_local_coverage_error",
]


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_gamma(gamma: float) -> float:
    g = float(gamma)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("gamma must be positive and finite")
    return g


def _check_tau(tau: int) -> int:
    if not isinstance(tau, (int, np.integer)) or int(tau) < 1:
        raise ValueError("tau must be an integer >= 1")
    return int(tau)


def _check_clip(clip: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(clip[0]), float(clip[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or not 0.0 < lo < hi < 1.0:
        raise ValueError("clip must satisfy 0 < lo < hi < 1")
    return lo, hi


def _check_err(err: float) -> float:
    e = float(err)
    if not np.isfinite(e) or not 0.0 <= e <= 1.0:
        raise ValueError("err must be a finite miscoverage indicator/rate in [0, 1]")
    return e


# --------------------------------------------------------------------- bounds
def long_run_coverage_bound(
    alpha: float,
    gamma: float,
    tau: int,
    T: int,
    alpha_init: float | Array | None = None,
) -> float:
    """Finite-sample long-run coverage bound of τ-DACI (paper Eq. (11)/(33)).

    Returns the worst-case bound on ``|(1/T) Σ_t err_t − α|`` (equivalently on
    the long-run empirical-coverage deviation ``|cov − (1 − α)|``):

        (1/(γT)) Σ_{θ=1..τ} max{α_1^{(θ)}, 1 − α_1^{(θ)}} + τ/T.

    Distribution-free given the delayed recursion: it only uses the
    phase-wise telescoping of Eq. (8) and the boundedness of the iterates
    (paper App. A.2). With ``alpha_init=None`` (the Algorithm-2
    initialization α_1^{(θ)} = α for every phase) it equals
    τ·max{α, 1−α}/(γT) + τ/T; at τ = 1 it reduces exactly to Proposition 4.1
    of Gibbs & Candès (2021). Worst case, not typical case — realized
    deviations are usually far smaller.
    """
    a = _check_alpha(alpha)
    g = _check_gamma(gamma)
    tl = _check_tau(tau)
    if not isinstance(T, (int, np.integer)) or int(T) < 1:
        raise ValueError("T must be an integer >= 1")
    t_steps = int(T)
    if alpha_init is None:
        inits = np.full(tl, a, dtype=float)
    else:
        inits = np.asarray(alpha_init, dtype=float).reshape(-1)
        if inits.size == 1:
            inits = np.full(tl, float(inits[0]), dtype=float)
        if inits.size != tl:
            raise ValueError("alpha_init must be scalar or have one entry per phase (tau)")
        if not np.all(np.isfinite(inits)) or np.any(inits < 0.0) or np.any(inits > 1.0):
            raise ValueError("alpha_init entries must lie in [0, 1] (paper App. A.2)")
    init_term = float(np.sum(np.maximum(inits, 1.0 - inits))) / (g * t_steps)
    return init_term + tl / t_steps


def marginal_coverage_bound(lip: float, gamma: float, level_shift: float) -> float:
    """Approximate marginal coverage bound of τ-DACI (paper Eq. (12)/(50)).

    Bounds ``E[(M(α_t | A_t) − α)²]``, the mean-squared deviation of the
    instantaneous marginal miscoverage from the target, by

        (L_Lip (1 + γ)/γ) · E|α*_{A_{t+τ}} − α*_{A_t}| + (L_Lip / 2) · γ,

    where ``level_shift`` = E|α*_{A_{t+τ}} − α*_{A_t}| is the expected change
    of the optimal level across the forecast horizon (the delay enters
    through this term) and ``lip`` = L_Lip. VALID ONLY under the paper's
    assumptions: stationary hidden-Markov environment, L_Lip-Lipschitz
    conditional miscoverage in the level, and a unique optimal level per
    environment. Approximate and worst-case — not a distribution-free
    guarantee. At τ = 1 this is Gibbs & Candès (2021, Thm. 4.2).
    """
    lp = float(lip)
    if not np.isfinite(lp) or lp <= 0.0:
        raise ValueError("lip must be positive and finite")
    g = _check_gamma(gamma)
    sh = float(level_shift)
    if not np.isfinite(sh) or sh < 0.0:
        raise ValueError("level_shift must be non-negative and finite")
    return lp * (1.0 + g) / g * sh + 0.5 * lp * g


def optimal_gamma(level_shift: float) -> float:
    """Bound-minimizing step size γ_opt = sqrt(2 · level_shift) (paper Sec. 4.4).

    Minimizes the right-hand side of ``marginal_coverage_bound`` over γ:
    the shift term decays like 1/γ while the adaptation penalty grows like γ.
    Descriptive tuning guidance, not a guarantee.
    """
    sh = float(level_shift)
    if not np.isfinite(sh) or sh <= 0.0:
        raise ValueError("level_shift must be positive and finite")
    return float(np.sqrt(2.0 * sh))


# ---------------------------------------------------- delay-to-memory ratio
def memory_length_ar1(phi: float) -> float:
    """AR(1) memory length L = −1/log φ (paper Eq. (20)).

    For ρ(k) = φ^k, φ^k = e^{−k/L} defines the lag at which the residual
    autocorrelation has decayed to e^{−1} of its lag-zero value.
    """
    p = float(phi)
    if not np.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("phi must lie in (0, 1) for a stationary AR(1) memory length")
    return float(-1.0 / np.log(p))


def memory_length_garch(arch_alpha: float, arch_beta: float) -> float:
    """GARCH(1,1) volatility memory length L_vol = −1/log(α_g + β) (paper Eq. (23)).

    ``arch_alpha`` / ``arch_beta`` are the GARCH(1,1) coefficients (NOT the
    miscoverage level α): the squared residuals follow the ARMA(1,1)
    representation ε_t² = ω + (α_g + β) ε_{t−1}² + ν_t − β ν_{t−1}
    (paper Eq. (22)), so squared-residual dependence decays geometrically at
    rate α_g + β.
    """
    a = float(arch_alpha)
    b = float(arch_beta)
    if not (np.isfinite(a) and np.isfinite(b)) or a < 0.0 or b < 0.0:
        raise ValueError("arch_alpha and arch_beta must be non-negative and finite")
    s = a + b
    if not 0.0 < s < 1.0:
        raise ValueError("arch_alpha + arch_beta must lie in (0, 1) (covariance stationarity)")
    return float(-1.0 / np.log(s))


def memory_length_markov(p00: float, p11: float) -> float:
    """Two-state Markov-switching regime memory L_regime = −1/log|λ₂| (paper Eq. (27)).

    λ₂ = p₀₀ + p₁₁ − 1 is the non-unit eigenvalue of the transition matrix;
    the latent-state autocorrelation is exactly Corr(A_t, A_{t+k}) = λ₂^k
    (paper Eq. (26), App. B.1). Requires 0 < |λ₂| < 1.
    """
    q00 = float(p00)
    q11 = float(p11)
    if not (np.isfinite(q00) and np.isfinite(q11)) or not (0.0 < q00 < 1.0 and 0.0 < q11 < 1.0):
        raise ValueError("p00 and p11 must lie in (0, 1)")
    lam2 = q00 + q11 - 1.0
    if not 0.0 < abs(lam2) < 1.0:
        raise ValueError("lambda_2 = p00 + p11 - 1 must satisfy 0 < |lambda_2| < 1")
    return float(-1.0 / np.log(abs(lam2)))


def delay_to_memory_ratio(tau: int, memory_length: float) -> float:
    """Delay-to-memory ratio r = τ/L (paper Eq. (20)/(23)/(27)).

    The dependence surviving the delay is e^{−r}: r ≪ 1 means the feedback is
    still informative about the target, r ≫ 1 means it has largely decayed.
    DESCRIPTIVE axis — it organizes simulated performance (strongly under
    AR(1), less under GARCH(1,1)/Markov switching; paper Table 6) but is not
    a coverage guarantee.
    """
    tl = _check_tau(tau)
    ml = float(memory_length)
    if not np.isfinite(ml) or ml <= 0.0:
        raise ValueError("memory_length must be positive and finite")
    return tl / ml


@dataclass(frozen=True)
class DelayToMemoryEstimate:
    """Stream estimate of the delay-to-memory ratio (descriptive diagnostic).

    Fields
    ------
    ratio:
        r̂ = τ / L̂.
    memory_length:
        L̂ = −1/slope of log-autocorrelation against lag, over the usable
        lags of the selected feature.
    decay_per_step:
        e^{slope} — the estimated per-step decay factor of the dependence
        (φ̂ for AR(1) levels, α̂_g + β̂ for squared residuals under GARCH).
    lags_used:
        Number of lags entering the log-linear fit.
    tau:
        Delay used for the ratio.
    """

    ratio: float
    memory_length: float
    decay_per_step: float
    lags_used: int
    tau: int


def estimate_delay_to_memory(
    residuals: Array,
    tau: int,
    feature: str = "level",
    max_lags: int = 8,
    acf_floor: float = 0.02,
) -> DelayToMemoryEstimate:
    """Estimate r = τ/L from a residual stream (descriptive, fail-closed).

    L is estimated from the decay RATE of the sample autocorrelation of the
    selected feature: log ρ̂(k) is regressed on lag k over the usable lags
    (ρ̂(k) > ``acf_floor``), and L̂ = −1/slope. For a signal-plus-noise stream
    the attenuation is a constant factor on ρ̂ (intercept shift), so the
    fitted rate still targets the driving signal's decay; for AR(1) levels it
    is consistent for L = −1/log φ (paper Eq. (20)). Use ``feature='square'``
    when the dependence lives in the scale (GARCH(1,1): squared residuals
    decay at rate α_g + β, paper Eq. (22)-(23)); for Markov-switching scale
    dependence the latent-state rate is not directly observable from
    residuals — prefer the closed-form ``memory_length_markov`` when the
    transition matrix is known.

    Fail-closed: raises when fewer than two usable lags survive the floor or
    the fit is non-decaying — no memory scale is fabricated for (near-)white
    streams, where the diagnostic is undefined. Biased on short streams; the
    ratio is descriptive, never a guarantee.
    """
    x = np.asarray(residuals, dtype=float)
    if x.ndim != 1:
        raise ValueError("residuals must be a 1-d stream")
    tl = _check_tau(tau)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("residuals must be non-empty and finite")
    if not isinstance(max_lags, (int, np.integer)) or int(max_lags) < 2:
        raise ValueError("max_lags must be an integer >= 2")
    ml = int(max_lags)
    if x.size < 10 * ml:
        raise ValueError(f"need at least {10 * ml} residuals for max_lags={ml}")
    if feature not in ("level", "square"):
        raise ValueError("feature must be 'level' or 'square'")
    fl = float(acf_floor)
    if not np.isfinite(fl) or not 0.0 < fl < 1.0:
        raise ValueError("acf_floor must lie in (0, 1)")
    v = x if feature == "level" else x * x
    v = v - float(np.mean(v))
    denom = float(np.dot(v, v))
    if denom <= 0.0:
        raise ValueError("residuals are constant; autocorrelation is undefined")
    n = v.size
    rho = np.array(
        [float(np.dot(v[: n - k], v[k:]) / denom) for k in range(1, ml + 1)], dtype=float
    )
    lags = np.arange(1, ml + 1, dtype=float)
    usable = rho > fl
    n_used = int(usable.sum())
    if n_used < 2:
        raise ValueError(
            "no positive autocorrelation decay identified (fewer than two usable lags); "
            "the delay-to-memory diagnostic is undefined for this (near-)white stream"
        )
    slope = float(np.polyfit(lags[usable], np.log(rho[usable]), 1)[0])
    if not np.isfinite(slope) or slope >= 0.0:
        raise ValueError("autocorrelation is not decaying; memory length is undefined")
    length = -1.0 / slope
    return DelayToMemoryEstimate(
        ratio=tl / length,
        memory_length=float(length),
        decay_per_step=float(np.exp(slope)),
        lags_used=n_used,
        tau=tl,
    )


# ------------------------------------------------------- evaluation metrics
def worst_case_local_coverage_error(err: Array, alpha: float, k: int) -> float:
    """Worst-case local coverage error LCE_k (paper Eq. (14)).

    max over contiguous windows of length k of |α − (1/k) Σ err_t|. Global
    empirical coverage can mask clustered miscoverage; LCE_k quantifies the
    worst local calibration failure. Smaller is better.
    """
    e = np.asarray(err, dtype=float)
    if e.ndim != 1:
        raise ValueError("err must be a 1-d stream")
    a = _check_alpha(alpha)
    if not isinstance(k, (int, np.integer)) or int(k) < 1:
        raise ValueError("k must be an integer >= 1")
    kk = int(k)
    if e.size < kk:
        raise ValueError(f"err must have at least k = {kk} entries")
    if not np.all(np.isfinite(e)) or np.any(e < 0.0) or np.any(e > 1.0):
        raise ValueError("err must contain finite values in [0, 1]")
    cs = np.concatenate([[0.0], np.cumsum(e)])
    window_means = (cs[kk:] - cs[:-kk]) / kk
    return float(np.max(np.abs(a - window_means)))


def adaptation_duration(
    covered: Array,
    threshold: float = 0.89,
    roll: int = 100,
    search_start: int = 0,
) -> tuple[int, bool]:
    """First coverage-dip adaptation duration (paper Sec. 6.5).

    Number of steps between the first time the ``roll``-observation rolling
    empirical coverage falls below ``threshold`` (at or after
    ``search_start``) and the first subsequent time it returns to at least
    ``threshold``. Returns ``(duration, recovered)``; ``recovered=False``
    means the dip was right-censored at the end of the series (the paper's
    convention records the duration at the segment boundary). No dip inside
    the search range returns ``(0, True)``.
    """
    c = np.asarray(covered, dtype=float)
    if c.ndim != 1:
        raise ValueError("covered must be a 1-d stream")
    if c.size == 0 or not np.all(np.isfinite(c)) or not np.all((c == 0.0) | (c == 1.0)):
        raise ValueError("covered must be a non-empty 0/1 indicator stream")
    th = float(threshold)
    if not np.isfinite(th) or not 0.0 < th < 1.0:
        raise ValueError("threshold must lie in (0, 1)")
    if not isinstance(roll, (int, np.integer)) or int(roll) < 1:
        raise ValueError("roll must be an integer >= 1")
    rl = int(roll)
    if not isinstance(search_start, (int, np.integer)) or int(search_start) < 0:
        raise ValueError("search_start must be a non-negative integer")
    ss = int(search_start)
    if c.size < rl:
        raise ValueError(f"covered must have at least roll = {rl} entries")
    cs = np.concatenate([[0.0], np.cumsum(c)])
    rolling = (cs[rl:] - cs[:-rl]) / rl
    if ss >= rolling.size:
        raise ValueError("search_start is beyond the rolling-coverage series")
    below = np.nonzero(rolling[ss:] < th)[0]
    if below.size == 0:
        return 0, True
    dip = ss + int(below[0])
    above = np.nonzero(rolling[dip:] >= th)[0]
    if above.size == 0:
        return int(rolling.size) - dip, False
    return int(above[0]), True


def curve_collapse_reduction(
    values: Array,
    ratio_axis: Array,
    delay_axis: Array,
    n_bins: int = 12,
) -> float:
    """Curve-collapse diagnostic 100·(1 − S_r/S_τ) (paper Eq. (17), Sec. 6).

    ``values`` is a performance measure (the paper uses the interval score at
    the selected γ) pooled across persistence configurations; ``ratio_axis``
    is r = τ/L and ``delay_axis`` is the raw delay τ for the same points.
    Each axis is divided into ``n_bins`` logarithmically spaced bins; S is the
    average within-bin standard deviation over bins holding at least two
    points. A positive reduction means the delay-to-memory ratio aligns the
    pooled curves more tightly than the raw delay. DESCRIPTIVE: the paper
    reports ≈79% (AR(1)), ≈46% (GARCH(1,1)), ≈9-27% (Markov switching,
    Table 6) — it organizes performance, it does not bound anything.
    """
    v = np.asarray(values, dtype=float).reshape(-1)
    r_ax = np.asarray(ratio_axis, dtype=float).reshape(-1)
    t_ax = np.asarray(delay_axis, dtype=float).reshape(-1)
    if v.size == 0 or v.size != r_ax.size or v.size != t_ax.size:
        raise ValueError("values, ratio_axis, and delay_axis must be non-empty and equal length")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(r_ax)) and np.all(np.isfinite(t_ax))):
        raise ValueError("values and axes must be finite")
    if np.any(r_ax <= 0.0) or np.any(t_ax <= 0.0):
        raise ValueError("axes must be positive (logarithmic binning)")
    if not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 2:
        raise ValueError("n_bins must be an integer >= 2")

    def _avg_within_bin_std(axis: Array) -> float:
        edges = np.exp(np.linspace(np.log(axis.min()), np.log(axis.max()) + 1e-9, int(n_bins) + 1))
        idx = np.clip(np.searchsorted(edges, axis, side="right") - 1, 0, int(n_bins) - 1)
        sds = [float(np.std(v[idx == b])) for b in range(int(n_bins)) if np.sum(idx == b) >= 2]
        if not sds:
            raise ValueError("no bin holds at least two points on this axis; cannot collapse")
        return float(np.mean(sds))

    s_r = _avg_within_bin_std(r_ax)
    s_t = _avg_within_bin_std(t_ax)
    if s_r < 0.0 or s_t <= 0.0:
        raise ValueError("degenerate within-bin scatter; collapse diagnostic undefined")
    return 100.0 * (1.0 - s_r / s_t)


# -------------------------------------------------------------- controller
class DelayedACI:
    """Scalar ACI level controller with τ-delayed miscoverage feedback.

    Implements the paper's Eq. (6) / Algorithm 2 queue form: the level issued
    at calendar step s targets step s + τ; when its outcome err arrives τ
    steps later, the NEXT issued level is

        clip(level_issued_at_s + γ (α − err), lo, hi),

    so levels evolve in τ interleaved one-step ACI phases (Eq. (8)): phase θ
    collects the steps s ≡ θ (mod τ) and updates once per τ calendar steps.
    At τ = 1 this is exactly ``quant_fund.models.conformal.AdaptiveConformal``'s
    recursion ``alpha_{t+1} = clip(alpha_t + γ(α − err_t), 1e-3, 1−1e-3)`` —
    the clip box defaults to the same (1e-3, 1 − 1e-3), and the τ = 1 level
    path is bit-for-bit identical (pinned in the tests).

    Parameters
    ----------
    alpha:
        Target miscoverage α ∈ (0, 1); the band built from the issued level
        has target coverage 1 − α.
    gamma:
        Step size γ > 0 (paper Sec. 4.4: γ_opt = sqrt(2 E|α*_{t+τ} − α*_t|)).
    tau:
        Feedback delay / forecast horizon τ ≥ 1 (integer). τ = 1 recovers
        standard ACI (Gibbs & Candès, 2021).
    clip:
        Saturation box (lo, hi), 0 < lo < hi < 1, applied after every update
        (repo convention; the paper analyzes the unclipped recursion, whose
        iterates stay in [−γ, 1+γ] by the degenerate-band self-correction of
        App. A.1). ``clip_updates_`` counts updates the box altered — the
        phase-wise telescoping behind ``coverage_bound`` is exact while this
        counter stays 0.

    Notes
    -----
    ``step(err)`` advances one calendar step and returns the level issued at
    that step. Feedback is delayed-but-guaranteed (paper Sec. 1): ``err``
    must be None for the first τ steps (nothing is in flight yet) and must be
    provided once τ sets are in flight — fail-closed either way. ``err`` may
    be a 0/1 indicator or a batch miscoverage rate in [0, 1].
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,
        tau: int = 1,
        clip: tuple[float, float] = (1e-3, 1.0 - 1e-3),
    ) -> None:
        self.alpha = _check_alpha(alpha)
        self.gamma = _check_gamma(gamma)
        self.tau = _check_tau(tau)
        self.clip = _check_clip(clip)
        self._initial_alpha = self.alpha
        self._current = self.alpha
        self._inflight: deque[float] = deque()
        self._levels: list[float] = []
        self._feedback: list[float] = []
        self._phase_levels: list[list[float]] = [[] for _ in range(self.tau)]
        self._phase_errs: list[list[float]] = [[] for _ in range(self.tau)]
        self._clip_updates = 0
        self._n = 0

    # ---------------------------------------------------------------- step
    def step(self, err: float | None) -> float:
        """Advance one calendar step; return the level issued at this step."""
        if len(self._inflight) == self.tau:
            if err is None:
                raise ValueError(
                    "err is required: delayed feedback for the set issued tau steps ago is due"
                )
            e = _check_err(err)
            issued = self._inflight.popleft()
            raw = issued + self.gamma * (self.alpha - e)
            self._current = float(np.clip(raw, self.clip[0], self.clip[1]))
            if self._current != raw:
                self._clip_updates += 1
            self._feedback.append(e)
            consumed: float | None = e
        else:
            if err is not None:
                raise ValueError(
                    "err must be None while fewer than tau sets are in flight "
                    "(no feedback is due yet)"
                )
            self._feedback.append(float("nan"))
            consumed = None
        self._inflight.append(self._current)
        self._levels.append(self._current)
        phase = self._n % self.tau  # 0-based phase of the level issued at this step
        self._phase_levels[phase].append(self._current)
        if consumed is not None:
            # The popped level was issued tau steps ago — the same phase.
            self._phase_errs[phase].append(consumed)
        self._n += 1
        return self._current

    # ----------------------------------------------------------- properties
    @property
    def current_level_(self) -> float:
        """Most recently issued level (α before the first step)."""
        return self._current

    @property
    def alpha_history(self) -> Array:
        """Level issued at each calendar step, shape (n_steps,)."""
        return np.asarray(self._levels, dtype=float)

    @property
    def feedback_history(self) -> Array:
        """Feedback consumed at each step (NaN when none was due), (n_steps,)."""
        return np.asarray(self._feedback, dtype=float)

    @property
    def n_steps_(self) -> int:
        return self._n

    @property
    def clip_updates_(self) -> int:
        """Number of updates whose raw value was altered by the clip box."""
        return self._clip_updates

    @property
    def phases_(self) -> list[tuple[Array, Array]]:
        """Interleaved-sequences view (paper Eq. (8)), one entry per phase θ.

        Entry θ − 1 is ``(levels, errs)``: the levels α_m^{(θ)} issued in that
        phase (calendar steps ≡ θ mod τ) and the feedback e_m^{(θ)} consumed
        by each update, with ``levels[m+1] = clip(levels[m] + γ(α − errs[m]))``
        and ``len(errs) = len(levels) − 1`` once the phase has updated.
        """
        return [
            (
                np.asarray(lv, dtype=float),
                np.asarray(er, dtype=float),
            )
            for lv, er in zip(self._phase_levels, self._phase_errs, strict=True)
        ]

    def coverage_bound(self, T: int) -> float:
        """Eq. (11) long-run bound at horizon T from the recorded phase inits."""
        inits = np.array(
            [
                lv[0] if lv else self._initial_alpha
                for lv in self._phase_levels  # α_1^{(θ)}: first level issued in phase θ
            ],
            dtype=float,
        )
        return long_run_coverage_bound(self.alpha, self.gamma, self.tau, T, alpha_init=inits)


# ------------------------------------------------------- simulation harness
@dataclass(frozen=True)
class DelayedConformalResult:
    """One online τ-DACI run on a residual stream (paper Algorithms 1+2).

    Fields
    ------
    levels, lower, upper:
        Per construction step c: the issued level α_c and the band
        [Q̂(α_c/2), Q̂(1 − α_c/2)] of the sliding score window, targeting
        residual index (window − 1 + c + τ).
    covered, err:
        Coverage indicators of the bands against their targets and the
        miscoverage indicators err = 1 − covered fed back τ steps later.
    interval_score, targets:
        Per-step interval score (paper Eq. (15)) and the target residuals.
    empirical_coverage, mean_interval_score:
        Time averages over all construction steps (paper Eq. (13)/(16)).
    long_run_deviation:
        |mean(err) − α| — compare against ``coverage_bound``.
    coverage_bound:
        Eq. (11) bound at T = n_constructions with α_1^{(θ)} = α.
    clip_updates:
        Updates altered by the clip box (0 ⇒ the bound's telescoping exact).
    controller:
        The ``DelayedACI`` instance (phase view via ``controller.phases_``).
    """

    levels: Array
    lower: Array
    upper: Array
    covered: Array
    err: Array
    interval_score: Array
    targets: Array
    empirical_coverage: float
    mean_interval_score: float
    long_run_deviation: float
    coverage_bound: float
    clip_updates: int
    controller: DelayedACI


def run_delayed_conformal(
    residuals: Array | Sequence[float],
    tau: int,
    alpha: float = 0.1,
    gamma: float = 0.01,
    window: int = 500,
    clip: tuple[float, float] = (1e-3, 1.0 - 1e-3),
) -> DelayedConformalResult:
    """τ-DACI online split conformal run on a seeded residual stream.

    The paper's Sec. 6 simulation setting: the residual stream ε_t (signed
    scores of the implicit point forecast, Ŷ = 0) is walked once; at
    construction step c (stream index i = window − 1 + c, window full) the
    band for target ε_{i+τ} is [Q̂_i(α_c/2), Q̂_i(1 − α_c/2)] over the most
    recent ``window`` scores (paper Algorithm 1, Eq. (3)-(4)); when ε_{i+τ}
    is revealed the miscoverage indicator re-enters the level recursion τ
    steps after issuance (Algorithm 2). No prediction ever sees its own
    target. Every construction step is evaluable, so all arrays have length
    n − window − τ + 1.

    Fail-closed: non-1-d/non-finite residuals, ``window < 2``,
    ``len(residuals) < window + tau``, or invalid (α, γ, τ, clip) raise.
    Determinism: no RNG — output is a pure function of the inputs.
    """
    x = np.asarray(residuals, dtype=float)
    if x.ndim != 1:
        raise ValueError("residuals must be a 1-d stream")
    tl = _check_tau(tau)
    a = _check_alpha(alpha)
    g = _check_gamma(gamma)
    box = _check_clip(clip)
    if not isinstance(window, (int, np.integer)) or int(window) < 2:
        raise ValueError("window must be an integer >= 2")
    R = int(window)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("residuals must be non-empty and finite")
    if x.size < R + tl:
        raise ValueError(f"need at least window + tau = {R + tl} residuals")
    n_constr = x.size - R - tl + 1
    if n_constr < 1:
        raise ValueError("stream too short for even one construction step")

    controller = DelayedACI(alpha=a, gamma=g, tau=tl, clip=box)
    levels = np.empty(n_constr, dtype=float)
    lower = np.empty(n_constr, dtype=float)
    upper = np.empty(n_constr, dtype=float)
    for c in range(n_constr):
        i = R - 1 + c
        fb: float | None = None
        if c >= tl:
            fb = 0.0 if lower[c - tl] <= x[i] <= upper[c - tl] else 1.0
        lvl = controller.step(fb)
        levels[c] = lvl
        w = x[i - R + 1 : i + 1]
        lower[c] = -conformal_quantile(-w, lvl / 2.0)
        upper[c] = conformal_quantile(w, lvl / 2.0)
    targets = x[R - 1 + tl : R - 1 + tl + n_constr]
    covered = ((targets >= lower) & (targets <= upper)).astype(float)
    err = 1.0 - covered
    iscore = (
        (upper - lower)
        + (2.0 / a) * np.maximum(lower - targets, 0.0)
        + (2.0 / a) * np.maximum(targets - upper, 0.0)
    )
    mean_err = float(np.mean(err))
    return DelayedConformalResult(
        levels=levels,
        lower=lower,
        upper=upper,
        covered=covered,
        err=err,
        interval_score=np.asarray(iscore, dtype=float),
        targets=targets,
        empirical_coverage=float(np.mean(covered)),
        mean_interval_score=float(np.mean(iscore)),
        long_run_deviation=abs(mean_err - a),
        coverage_bound=controller.coverage_bound(n_constr),
        clip_updates=controller.clip_updates_,
        controller=controller,
    )
