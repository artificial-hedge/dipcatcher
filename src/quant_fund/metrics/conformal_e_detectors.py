"""Minimax-optimal conformal change detection: restarted conformal e-processes.

Bhattacharyya & Ramdas (2026), "Change detection with conformal martingales:
new optimal constructions, and suboptimality of existing methods",
arXiv:2609.27179 [math.ST], 76 pp. Distribution-free sequential changepoint
detection for independent observations X_1, X_2, ... with UNKNOWN and
UNRESTRICTED pre-/post-change laws P_0, P_1 and unknown changepoint T (their
Eq. (1); T is the last pre-change index). All statistics live in the reduced
filtration of the conformal p-values, G_n = sigma(p_1, ..., p_n): their
Proposition 3.1 and Theorem B.1 show that NO universally valid detector (or
full-filtration test supermartingale) can have power one against a fixed
changepoint over a composite class, so rates must be — and here are — stated
in the growing-changepoint double-array regime (their Sec. 3.1, Eq. (21)).

P-values: the sequential randomized conformal p-values of Vovk (2021,
"Testing randomness online", Statistical Science 36(4)) as implemented by
:func:`quant_fund.metrics.conformal_martingale.conformal_p_values` — i.i.d.
U(0,1) under exchangeability (their Prop. 2.1), ties handled by the
randomizer. CONVENTION NOTE (paper vs. repo): their Eq. (7) writes the
numerator with 1(X_i < X_n) (lower-tail), but their Section 6 simulations bet
with the LEFT-tail density f(p) = 0.15 p^{-0.85} against a RIGHT mean shift
N(0,1) -> N(1.2,1), which only profits under the upper-tail (Vovk/repo)
convention p_t ~ #{i: s_i > s_t}/t; the two are the relabeling p <-> 1 - p and
have identical null validity. This module reuses the repo's upper-tail
p-values and defaults to a TWO-SIDED betting grid, so the profitable direction
never has to be known (their setup assumes neither direction of P_1).

Constructions (their equations; all fail-closed on invalid input):

* Conformal test martingale CTM (Vovk 2021, their Eq. (9)): S_n = prod f(p_i)
  for one betting density; alarm tau^CTM(1/alpha). SUBOPTIMAL: delay can be
  Omega(T) under PFA control (their Example 2.2) and Omega(sqrt(gamma)) under
  ARL control via their CUSUM/SR wrappers (Eq. (10), Example 2.3).
* Conformal mixture martingale CMM (their Eq. (11)): M_n = int prod f_theta
  (p_i) dPi(theta) over a betting class with prior Pi; plus their CMM-CUSUM /
  CMM-SR (Eq. (13)). Mixing alone does NOT remove the dilution of the
  accumulated pre-change history: still Omega(T) / Omega(sqrt(gamma))
  (their Examples 2.4-2.5); at a fixed threshold the CMM delay scale is
  sqrt(T log b) (their Thm. 5.2-5.3).
* OPTIMAL restarted constructions (their Eq. (14)-(16), Def. 2.6, Alg. 1):
  locally restarted mixture martingales M_{k,t} = int prod_{j=k}^t f_theta(p_j)
  dPi(theta) aggregated over candidate changepoints k with deterministic
  restart weights w = (w_k),

      S_t^(w) = sum_{k<=t} w_k M_{k,t}   (Shiryaev-Roberts type, "sum"),
      Z_t^(w) = max_{k<=t} w_k M_{k,t}   (CUSUM type, "max"),

  unified by Theorem 2.7: if ||w||_1 <= 1 then S~_t = S_t + sum_{k>t} w_k
  (their Eq. (6)) is a nonnegative martingale — a conformal E-PROCESS — so
  thresholding S or Z at b = 1/alpha gives PFA <= ||w||_1 * alpha <= alpha at
  ANY stopping time (Eq. (17), Ville); if ||w||_inf <= 1 then S/||w||_inf is
  an e-detector in the sense of Shin, Ramdas & Rinaldo (2023, Ann. Statist.
  51(4):1233-1268, arXiv:2203.03532; cf. quant_fund.metrics.e_detectors) and
  thresholding at b = gamma gives ARL >= gamma/||w||_inf (Eq. (20)) plus the
  optional-horizon / linear early-alarm bound P(tau <= m) <= m * ||w||_inf / b
  (Eq. (19)). The null-bet modification M-bar = rho + (1 - rho) M (their
  Prop. 4.10, Thm. 3.10) keeps every calibration and forces sure stopping by
  ceil(b/rho) when the weights are unit, with gamma <= E tau <= ceil(gamma/rho).
* Delay rates (growing-changepoint regime, separation D = D_KL(H || U) with H
  the law of F_{P_0}(Y), Y ~ P_1): polynomial weights w_k prop. k^{-(1+eta)}
  give delay ~ ((1+eta)/D) log T under PFA control (their Cor. 3.6);
  near-harmonic w_k prop. 1/(k log^2(e+k)) improve the constant to
  (log T + 2 log log T)/D (their Rem. 3.7/4.7); unit weights give
  ~ log(gamma)/D under ARL control (their Cor. 3.8). Without oracle knowledge
  of H, a prior atom at a profitable density f with mass v_f gives delay
  ~ (log(b/w_{T+1}) + log(1/v_f)) / I_f(H), I_f(H) = int log f dH > 0 (their
  Thm. 4.11 weighted oracle inequality); betting classes satisfying their
  Assumptions 2-4 exist without any knowledge of P_0/P_1 (Prop. A.1-A.3).
  These rates are FIRST-ORDER RESTRICTED MINIMAX OPTIMAL over late
  changepoints T in I_N(kappa) (their Thm. 3.9 for PFA, Thm. 3.10 for ARL,
  matching lower bound 1/D_KL(P_1 || P_0) attained with the likelihood-ratio
  score, which is the only information-lossless score transformation — their
  Prop. 3.11 data-processing inequality).

Betting class implemented: the power densities f_theta(p) = theta p^(theta-1)
on the left tail (bets on small p-values) and their mirrors theta (1-p)^(theta
-1) on the right tail, theta in (0, 1], discrete prior — the exact class of
their Sec. 6 / Fig. 2 simulations (theta in {0.15, 0.30, 0.50, 0.75, 1},
uniform prior; Fig. 1/5 uses the single density theta = 0.15). Each atom
integrates to one against U(0,1), so every mixture martingale here is a test
martingale under exchangeability.

Comparison harness: :func:`pfa_delay_study` / :func:`arl_delay_study` run the
existing repo constructions (conformal_martingale power/mixture/jumper test
martingales — Vovk 2021-style — and e_detectors parametric mixture-SR)
against the new optimal restarted e-processes/e-detectors on IDENTICAL seeded
SYNTHETIC Gaussian streams with a planted changepoint, at matched PFA level
(threshold 1/alpha for every Ville-valid method) or matched ARL level
(threshold gamma for every ARL-valid method); :func:`delay_growth_study`
sweeps the changepoint location to exhibit the polynomial-vs-logarithmic
delay gap. Their Sec. 6 protocol is mirrored: P_0 = N(0,1), P_1 = N(1.2, 1),
alpha = 0.05, monitoring until T + 5T, capped delay (tau - T) ^ 5T, and
false-alarm replicates excluded from the delay summary.

Honesty: every stream here is seeded SYNTHETIC data (np.random.default_rng);
the Monte-Carlo studies are correctness demonstrations of the error control
and ILLUSTRATE — never prove — the asymptotic rates at finite sample sizes.
Delays and false-alarm rates are proper sequential-testing diagnostics
(error-probability semantics, time-uniform under exchangeability); nothing
here is market evidence, a P&L/Sharpe-style performance claim, or a
live-trading capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from quant_fund.metrics.conformal_martingale import (
    conformal_p_values,
    mixture_martingale,
    simple_jumper,
)
from quant_fund.metrics.e_detectors import EDetectorGaussian, run_detector

Array = NDArray[np.float64]

__all__ = [
    "ARL_METHODS",
    "CTM_THETA",
    "DEFAULT_ETA",
    "DEFAULT_RHO",
    "DEFAULT_THETAS",
    "PFA_METHODS",
    "ConformalDetectionResult",
    "ConformalRestartDetector",
    "DelayStudyRow",
    "PowerBetting",
    "arl_calibration_study",
    "arl_delay_study",
    "cmm_log_wealth",
    "conformal_e_detector",
    "conformal_e_process",
    "ctm_log_wealth",
    "delay_growth_study",
    "false_alarm_study",
    "first_crossing",
    "first_crossing_log",
    "pfa_delay_study",
    "power_betting_class",
    "power_log_bets",
    "restart_weights",
    "summarize_delays",
    "synthetic_changepoint_scores",
    "vovk_cusum_stat",
    "vovk_sr_stat",
]

# Mirrors conformal_martingale._P_FLOOR: flooring p away from 0 understates
# p^(theta-1) (theta - 1 < 0), so capital — and alarms — stay conservative.
_P_FLOOR = 1e-300
# Mirrors e_detectors._E_MAX: log-space cap so reported paths stay finite.
_LOG_CAP = float(np.log(1e300))
# O(n_max * K) engine memory and O(T^2) p-value ranking cost: fail closed
# beyond this horizon instead of exhausting memory/time.
_N_MAX_LIMIT = 20_000
# Keeps study data seeds and p-value randomizer seeds from colliding.
_RANDOMIZER_SEED_OFFSET = 1_000_003

# Paper Sec. 6 grids/protocols (arXiv:2609.27179).
DEFAULT_THETAS: tuple[float, ...] = (0.15, 0.30, 0.50, 0.75, 1.0)
DEFAULT_ETA: float = 0.10  # their w_k prop. k^{-(1+eta)}, eta = 0.10
DEFAULT_RHO: float = 0.05  # null-bet fraction (their Prop. 4.10: "small rho")
CTM_THETA: float = 0.15  # their Fig. 1/5 single density f(p) = 0.15 p^{-0.85}

PFA_METHODS: tuple[str, ...] = (
    "vovk_ctm",
    "cmm_grid",
    "repo_power_mixture",
    "repo_jumper",
    "optimal_sum",
    "optimal_max",
)
ARL_METHODS: tuple[str, ...] = (
    "vovk_sr_ctm",
    "vovk_cusum_ctm",
    "vovk_sr_cmm",
    "vovk_cusum_cmm",
    "repo_mixture_sr",
    "optimal_sum",
    "optimal_max",
)
_PFA_NULL_METHODS: tuple[str, ...] = (
    "vovk_ctm",
    "cmm_grid",
    "repo_power_mixture",
    "optimal_sum",
    "optimal_max",
)
_ARL_NULL_METHODS: tuple[str, ...] = (
    "vovk_sr_ctm",
    "vovk_cusum_ctm",
    "vovk_sr_cmm",
    "vovk_cusum_cmm",
    "optimal_sum",
    "optimal_max",
)


# ---------------------------------------------------------------------------
# validation helpers (fail-closed)
# ---------------------------------------------------------------------------


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _check_gamma(gamma: float) -> float:
    g = float(gamma)
    if not np.isfinite(g) or g < 1.0:
        raise ValueError("gamma must be finite and >= 1 (ARL level)")
    return g


def _check_threshold(b: float) -> float:
    v = float(b)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("threshold must be finite and > 0")
    if v >= 1e300:
        raise ValueError("threshold must stay below the 1e300 log-space cap")
    return v


def _check_n_max(n_max: int) -> int:
    n = int(n_max)
    if n < 1:
        raise ValueError("n_max must be >= 1")
    if n > _N_MAX_LIMIT:
        raise ValueError(f"n_max exceeds the {_N_MAX_LIMIT} horizon limit (O(T^2) ranking cost)")
    return n


def _check_p_array(p: Array | Sequence[float]) -> Array:
    pa = np.asarray(p, dtype=float).ravel()
    if pa.size == 0:
        raise ValueError("p-values must be non-empty")
    if not bool(np.all(np.isfinite(pa))) or bool(np.any(pa < 0.0)) or bool(np.any(pa > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    return pa


def _check_scores(scores: Array | Sequence[float]) -> Array:
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not bool(np.all(np.isfinite(s))):
        raise ValueError("scores must be finite (NaN/inf rejected)")
    return s


def _first_crossing_generic(
    path: Array | Sequence[float], threshold: float, log_space: bool
) -> int | None:
    v = _check_threshold(threshold)
    arr = np.asarray(path, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("path must be non-empty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("path must be finite")
    if log_space:
        hits = np.flatnonzero(arr >= float(np.log(v)))
    else:
        if bool(np.any(arr < 0.0)):
            raise ValueError("path must be nonnegative")
        hits = np.flatnonzero(arr >= v)
    return int(hits[0]) if hits.size else None


def first_crossing(path: Array | Sequence[float], threshold: float) -> int | None:
    """0-based index of the first entry >= threshold, or None (Ville alarm)."""
    return _first_crossing_generic(path, threshold, log_space=False)


def first_crossing_log(log_path: Array | Sequence[float], threshold: float) -> int | None:
    """First crossing for a LOG-wealth path: first entry >= log(threshold)."""
    return _first_crossing_generic(log_path, threshold, log_space=True)


# ---------------------------------------------------------------------------
# betting class (their Sec. 6 power densities, two-sided)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerBetting:
    """Discrete betting class of power densities with a fixed prior.

    Atom k bets ``f_theta(q) = theta * q^(theta - 1)`` — a density on [0, 1]
    integrating to one, so each atom is a fair one-step bet against U(0,1) —
    on ``q = p`` for left-side atoms (profit when p-values concentrate near 0)
    and ``q = 1 - p`` for right-side atoms (profit near 1). ``theta = 1`` is
    the null bet f == 1. This is the class of the paper's Sec. 6 simulations
    (theta in {0.15, 0.30, 0.50, 0.75, 1}, uniform prior); a two-sided grid
    covers both drift directions, matching their unknown/unrestricted P_1.
    """

    thetas: Array  # (K,) in (0, 1]
    sides: Array  # (K,) in {+1 (left), -1 (right)}
    log_prior: Array  # (K,), logsumexp == 0, all finite (strictly positive prior)

    @property
    def n_atoms(self) -> int:
        """Number of betting atoms K."""
        return int(self.thetas.shape[0])

    def log_bets(self, p: float | Array) -> Array:
        """log f_theta(p) per atom: (K,) for scalar p, (T, K) for a (T,) path."""
        pa = np.asarray(p, dtype=float)
        if pa.ndim > 1:
            raise ValueError("p must be a scalar or 1-D array")
        scalar = pa.ndim == 0
        pa1 = np.atleast_1d(pa)
        if pa1.size == 0 or not bool(np.all(np.isfinite(pa1))):
            raise ValueError("p-values must be non-empty and finite")
        if bool(np.any(pa1 < 0.0)) or bool(np.any(pa1 > 1.0)):
            raise ValueError("p-values must lie in [0, 1]")
        q = np.where(self.sides[None, :] > 0.0, pa1[:, None], 1.0 - pa1[:, None])
        out = self.log_theta_broadcast() + (self.thetas[None, :] - 1.0) * np.log(
            np.clip(q, _P_FLOOR, 1.0)
        )
        out = np.asarray(out, dtype=float)
        return out[0] if scalar else out

    def log_theta_broadcast(self) -> Array:
        """log(theta) row vector (1, K) for broadcasting in :meth:`log_bets`."""
        return np.asarray(np.log(self.thetas)[None, :], dtype=float)


def power_betting_class(
    thetas: Sequence[float] | Array = DEFAULT_THETAS,
    sides: Literal["both", "left", "right"] = "both",
) -> PowerBetting:
    """Uniform-prior power betting class (their Fig. 2 grid by default).

    ``sides="both"`` mirrors every theta onto the right tail as well, so the
    detector never needs to know the direction of the change; ``"left"`` /
    ``"right"`` reproduce the single-direction grids of their simulations.
    """
    th = np.asarray(list(thetas), dtype=float).ravel()
    if th.size == 0 or not bool(np.all(np.isfinite(th))) or bool(np.any(th <= 0.0)):
        raise ValueError("thetas must be non-empty, finite, and in (0, 1]")
    if bool(np.any(th > 1.0)):
        raise ValueError("thetas must lie in (0, 1] (f_theta a density on [0, 1])")
    if sides == "both":
        th_all = np.concatenate([th, th])
        side_all = np.concatenate([np.ones_like(th), -np.ones_like(th)])
    elif sides == "left":
        th_all, side_all = th, np.ones_like(th)
    elif sides == "right":
        th_all, side_all = th, -np.ones_like(th)
    else:
        raise ValueError("sides must be one of 'both', 'left', 'right'")
    k = int(th_all.shape[0])
    return PowerBetting(
        thetas=np.asarray(th_all, dtype=float),
        sides=np.asarray(side_all, dtype=float),
        log_prior=np.full(k, -float(np.log(k)), dtype=float),
    )


# ---------------------------------------------------------------------------
# restart weights (their Cor. 3.6 / Rem. 3.7 / Cor. 3.8 choices)
# ---------------------------------------------------------------------------


def restart_weights(kind: str, n_max: int, *, eta: float = DEFAULT_ETA) -> Array:
    """Deterministic restart weights over candidate changepoints k = 1..n_max.

    * ``"polynomial"``: w_k prop. k^{-(1+eta)}, eta > 0, normalized to total
      mass 1 — their Cor. 3.6 / Sec. 6 choice (delay constant (1+eta)/D).
    * ``"near_harmonic"``: w_k prop. 1/(k log^2(e+k)), normalized to mass 1 —
      their Rem. 3.7/4.7 near-harmonic weights; summable, and the delay
      constant improves to (log T + 2 log log T)/D.
    * ``"unit"``: w_k = 1 — their ARL regime (||w||_inf = 1, ARL >= b).

    Normalization is over the DECLARED horizon (finite support is a valid
    deterministic weight sequence; relative to their infinite-horizon zeta /
    c_nh normalization it only enlarges w_{T+1}, shrinking the effective
    boundary log(b / w_{T+1}) and hence the delay bound, preserving order).
    """
    n = _check_n_max(n_max)
    k = np.arange(1, n + 1, dtype=float)
    if kind == "polynomial":
        e = float(eta)
        if not np.isfinite(e) or e <= 0.0:
            raise ValueError("eta must be finite and > 0")
        w = k ** -(1.0 + e)
    elif kind == "near_harmonic":
        w = 1.0 / (k * np.log(np.e + k) ** 2)
    elif kind == "unit":
        w = np.ones(n, dtype=float)
    else:
        raise ValueError("kind must be one of 'polynomial', 'near_harmonic', 'unit'")
    total = float(np.sum(w))
    if kind == "unit":
        return np.asarray(w, dtype=float)
    return np.asarray(w / total, dtype=float)


# ---------------------------------------------------------------------------
# optimal restarted detector (their Def. 2.6 / Alg. 1 / Thm. 2.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConformalDetectionResult:
    """Output of :meth:`ConformalRestartDetector.run` / ``run_p_values``.

    ``alarm_index`` is the 0-based index of the first crossing of ``statistic``
    at ``threshold`` (None if the stream ends unalarmed); ``statistic`` is the
    alarmed process — S-bar_t^(w,rho) for ``stat="sum"`` or Z-bar_t^(w,rho)
    for ``stat="max"`` (their Eq. (15)-(16), Prop. 4.10). ``e_process`` (sum
    statistic only) is S-bar_t + sum_{k>t} w_k, their Eq. (6) conformal
    e-process: with rho = 0 and ||w||_1 <= 1 a nonnegative martingale with
    initial value ||w||_1, so Ville gives PFA <= ||w||_1 * alpha at ANY
    stopping. Paths are truncated at the alarm when ``stop_on_alarm`` is set.
    """

    alarm_index: int | None
    statistic: Array
    e_process: Array | None
    threshold: float
    p_values: Array
    method: str


class ConformalRestartDetector:
    """Restarted conformal mixture detector (RCMM) — the paper's Alg. 1.

    Consumes either raw nonconformity scores (computing the sequential
    randomized conformal p-values of :func:`conformal_p_values` internally) or
    precomputed p-values (``update_p_value`` / ``run_p_values`` — the studies
    feed EVERY method the identical p-value sequence for a matched
    comparison). Per betting atom m the sum statistic uses the O(1) recursion

        A_t(m) = f_m(p_t) * (A_{t-1}(m) + w_t),   S_t = sum_m pi_m A_t(m),

    which telescopes the restart sum sum_{k<=t} w_k M_{k,t} exactly (the
    mixture commutes with the sum over restarts); the max statistic maintains
    the full (t, K) log-product matrix G_{k,t}(m) of their Alg. 1 lines
    10-16 and takes max_k logsumexp_m. The null-bet modification (their
    Prop. 4.10) reports S-bar_t = rho * W_t + (1 - rho) * S_t (and the
    per-restart analogue for Z-bar), preserving PFA/ARL calibration while
    forcing sure stopping by ceil(b/rho) for unit weights.

    Validity (their Thm. 2.7, under exchangeability of the score stream):
    PFA of the 1/alpha alarm <= ||w||_1 * alpha at any stopping time; ARL of
    the b alarm >= b / ||w||_inf, with P(tau <= m) <= m * ||w||_inf / b.
    Prefer the regime factories :func:`conformal_e_process` (||w||_1 <= 1,
    b = 1/alpha) and :func:`conformal_e_detector` (unit weights, b = gamma).
    """

    def __init__(
        self,
        *,
        n_max: int,
        threshold: float,
        weights: Array,
        betting: PowerBetting,
        stat: Literal["sum", "max"] = "sum",
        rho: float = 0.0,
        seed: int = 0,
        weights_kind: str = "custom",
    ) -> None:
        self.n_max = _check_n_max(n_max)
        self.threshold = _check_threshold(threshold)
        if stat not in ("sum", "max"):
            raise ValueError("stat must be 'sum' or 'max'")
        r = float(rho)
        if not np.isfinite(r) or not 0.0 <= r < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != self.n_max or not bool(np.all(np.isfinite(w))) or bool(np.any(w < 0.0)):
            raise ValueError("weights must be an (n_max,) array of nonnegative finite values")
        if float(np.max(w)) <= 0.0:
            raise ValueError("weights must carry positive mass")
        if not isinstance(betting, PowerBetting):
            raise ValueError("betting must be a PowerBetting instance")
        self.stat = stat
        self.rho = r
        self.seed = int(seed)
        self.weights_kind = str(weights_kind)
        self.betting = betting
        self.weights = np.array(w, dtype=float)
        with np.errstate(divide="ignore"):
            self._log_w = np.asarray(np.log(w), dtype=float)
        self._log_threshold = float(np.log(self.threshold))
        self._cum_w = np.asarray(np.cumsum(w), dtype=float)
        self._w_total = float(self._cum_w[-1])
        self._tail = np.asarray(self._w_total - self._cum_w, dtype=float)  # sum_{k>t} w_k
        self._log_rho = float(np.log(r)) if r > 0.0 else -np.inf
        self._log_1m_rho = float(np.log1p(-r)) if r > 0.0 else 0.0
        self.method = f"rcmm-{stat}:{self.weights_kind}:rho={self.rho:g}"
        self._log_g: Array | None = None
        self._history = np.empty(0, dtype=float)
        self._log_a = np.empty(0, dtype=float)
        self._log_stat = -np.inf
        self._t = 0
        self._rng = np.random.default_rng(seed)
        self.reset()

    # -- protocol surface (mirrors quant_fund.metrics.e_detectors) ----------

    def reset(self) -> None:
        """Restore the initial state (no data seen, statistics at 0)."""
        k = self.betting.n_atoms
        self._log_a = np.full(k, -np.inf, dtype=float)
        self._log_g = np.full((self.n_max, k), -np.inf, dtype=float) if self.stat == "max" else None
        self._history = np.empty(self.n_max, dtype=float)
        self._log_stat = -np.inf
        self._t = 0
        self._rng = np.random.default_rng(self.seed)

    @property
    def value(self) -> float:
        """Current primary statistic (S-bar_t or Z-bar_t; 0.0 before data)."""
        return float(np.exp(self._log_stat))

    @property
    def alarmed(self) -> bool:
        """True once the primary statistic reached the threshold."""
        return bool(self._log_stat >= self._log_threshold)

    @property
    def n_seen(self) -> int:
        """Number of observations incorporated since the last reset."""
        return self._t

    @property
    def pfa_bound(self) -> float:
        """Guaranteed PFA of the threshold alarm: min(1, ||w||_1 * alpha_eff).

        alpha_eff = 1/threshold; the bound is ||w||_1 / threshold (their
        Eq. (17)) — <= alpha exactly when ||w||_1 <= 1 and b = 1/alpha.
        """
        return float(min(1.0, self._w_total / self.threshold))

    @property
    def arl_bound(self) -> float:
        """Guaranteed ARL of the threshold alarm: threshold / ||w||_inf (Eq. (20))."""
        return float(self.threshold / float(np.max(self.weights)))

    # -- sequential updates --------------------------------------------------

    def update(self, score: float) -> float:
        """Feed one raw score; returns the primary statistic value."""
        v = float(score)
        if not np.isfinite(v):
            raise ValueError("score must be finite")
        if self._t >= self.n_max:
            raise ValueError(f"stream exceeds the declared horizon n_max={self.n_max}")
        p = self._p_from_score(v)
        self._history[self._t] = v
        return self.update_p_value(p)

    def update_p_value(self, p: float) -> float:
        """Feed one precomputed conformal p-value; returns the statistic value."""
        pv = float(p)
        if not np.isfinite(pv) or pv < 0.0 or pv > 1.0:
            raise ValueError("p-value must be finite and lie in [0, 1]")
        if self._t >= self.n_max:
            raise ValueError(f"stream exceeds the declared horizon n_max={self.n_max}")
        self._step(pv)
        return self.value

    def _p_from_score(self, score: float) -> float:
        """Sequential randomized conformal p-value (repo/Vovk upper-tail form).

        Identical to :func:`conformal_martingale.conformal_p_values` step t:
        p_t = (#{i <= t: s_i > s_t} + u_t #{i <= t: s_i = s_t}) / (t + 1) with
        the count including s_t itself (self-tie), u_t ~ U(0,1) from this
        detector's seeded rng — i.i.d. U(0,1) under exchangeability.
        """
        t = self._t
        hist = self._history[:t]
        greater = int(np.count_nonzero(hist > score))
        ties = int(np.count_nonzero(hist == score)) + 1
        u = float(self._rng.uniform())
        return float((greater + u * ties) / (t + 1))

    def _step(self, p: float) -> None:
        t = self._t
        log_f = self.betting.log_bets(p)  # (K,)
        # Sum statistic: A_t(m) = f_m(p_t) (A_{t-1}(m) + w_t); S_t = sum pi_m A_t.
        self._log_a = np.asarray(log_f + np.logaddexp(self._log_a, self._log_w[t]), dtype=float)
        log_s = float(logsumexp(self._log_a + self.betting.log_prior))
        log_sum_stat = self._apply_null_bet(log_s, t)
        if self.stat == "sum":
            self._log_stat = min(log_sum_stat, _LOG_CAP)
        else:
            if not (self._log_g is not None):
                raise ValueError("self._log_g is not None")  # guaranteed by __init__ for stat="max"
            g = self._log_g
            g[:t] += log_f  # inactive rows are -inf: -inf + finite = -inf
            g[t] = log_f  # new restart k = t+1 (their Alg. 1 line 10)
            log_m = np.asarray(
                logsumexp(g[: t + 1] + self.betting.log_prior[None, :], axis=1), dtype=float
            )
            lw = self._log_w[: t + 1]
            if self.rho > 0.0:
                terms = np.logaddexp(self._log_rho + lw, self._log_1m_rho + log_m + lw)
            else:
                terms = log_m + lw
            self._log_stat = min(float(np.max(terms)), _LOG_CAP)
        self._t = t + 1

    def _apply_null_bet(self, log_s: float, t: int) -> float:
        """S-bar_t = rho W_t + (1 - rho) S_t in log space (their Prop. 4.10)."""
        if self.rho <= 0.0:
            return log_s
        det = self._log_rho + float(np.log(self._cum_w[t]))
        return float(np.logaddexp(det, self._log_1m_rho + log_s))

    def _e_process_value(self, t: int) -> float:
        """S-bar_t + sum_{k>t} w_k (their Eq. (6)); a martingale when rho = 0."""
        stat_log = self._log_stat if self.stat == "sum" else None
        if stat_log is None:
            raise ValueError("e-process path requires stat='sum'")
        return float(min(np.exp(stat_log) + self._tail[t], 1e300))

    # -- batch runs -----------------------------------------------------------

    def run(
        self, scores: Array | Sequence[float], *, stop_on_alarm: bool = True
    ) -> ConformalDetectionResult:
        """Reset, then feed a raw score stream (p-values computed internally).

        The result depends only on (config, scores): :meth:`reset` re-seeds
        the randomizer rng. The returned ``p_values`` equal
        :func:`conformal_p_values` on the same scores and seed.
        """
        s = _check_scores(scores)
        if s.shape[0] > self.n_max:
            raise ValueError(f"stream of length {s.shape[0]} exceeds n_max={self.n_max}")
        self.reset()
        p = conformal_p_values(s, seed=self.seed)
        return self._run_p(p, stop_on_alarm=stop_on_alarm)

    def run_p_values(
        self, p_values: Array | Sequence[float], *, stop_on_alarm: bool = True
    ) -> ConformalDetectionResult:
        """Reset, then feed a precomputed conformal p-value path."""
        p = _check_p_array(p_values)
        if p.shape[0] > self.n_max:
            raise ValueError(f"stream of length {p.shape[0]} exceeds n_max={self.n_max}")
        self.reset()
        return self._run_p(p, stop_on_alarm=stop_on_alarm)

    def _run_p(self, p: Array, *, stop_on_alarm: bool) -> ConformalDetectionResult:
        n = int(p.shape[0])
        stat_path = np.empty(n, dtype=float)
        e_path = np.empty(n, dtype=float) if self.stat == "sum" else None
        alarm_index: int | None = None
        for i in range(n):
            self._step(float(p[i]))
            stat_path[i] = float(np.exp(self._log_stat))
            if e_path is not None:
                e_path[i] = self._e_process_value(i)
            if stop_on_alarm and self._log_stat >= self._log_threshold:
                alarm_index = i
                stat_path = stat_path[: i + 1]
                if e_path is not None:
                    e_path = e_path[: i + 1]
                p = p[: i + 1]
                break
        if alarm_index is None and not stop_on_alarm:
            hits = np.flatnonzero(stat_path >= self.threshold)
            alarm_index = int(hits[0]) if hits.size else None
        return ConformalDetectionResult(
            alarm_index=alarm_index,
            statistic=np.asarray(stat_path, dtype=float),
            e_process=e_path,
            threshold=self.threshold,
            p_values=np.asarray(p, dtype=float),
            method=self.method,
        )


def conformal_e_process(
    *,
    alpha: float,
    n_max: int,
    weights: Literal["polynomial", "near_harmonic"] | Array = "polynomial",
    eta: float = DEFAULT_ETA,
    betting: PowerBetting | None = None,
    stat: Literal["sum", "max"] = "max",
    rho: float = 0.0,
    seed: int = 0,
) -> ConformalRestartDetector:
    """PFA-regime factory: conformal e-process alarm at 1/alpha (Thm. 2.7(1)).

    ``weights`` must be summable with ||w||_1 <= 1 (the named kinds are
    normalized to mass 1 over the horizon); the false-alarm probability of
    ``inf{t: stat_t >= 1/alpha}`` is then <= alpha under EVERY exchangeable
    no-change law, at any stopping time. Default ``stat="max"`` mirrors their
    Sec. 6 PFA simulation (Z_t^(w), w_k prop. k^{-1.1}).
    """
    a = _check_alpha(alpha)
    n = _check_n_max(n_max)
    if isinstance(weights, str):
        if weights not in ("polynomial", "near_harmonic"):
            raise ValueError("weights kind must be 'polynomial' or 'near_harmonic' for PFA control")
        w = restart_weights(weights, n, eta=eta)
        kind = weights if weights == "near_harmonic" else f"polynomial(eta={eta:g})"
    else:
        w = np.asarray(weights, dtype=float).ravel()
        kind = "custom"
    if w.shape[0] != n or not bool(np.all(np.isfinite(w))) or bool(np.any(w < 0.0)):
        raise ValueError("weights must be an (n_max,) nonnegative finite array")
    w_sum = float(np.sum(w))
    if w_sum > 1.0 + 1e-12:
        raise ValueError(
            "PFA control at level alpha requires ||w||_1 <= 1 (Thm. 2.7(1)); custom weights "
            "are used exactly as given (no hidden renormalization) — pass "
            "restart_weights(kind, n_max) output or normalize manually"
        )
    return ConformalRestartDetector(
        n_max=n,
        threshold=1.0 / a,
        weights=w,
        betting=betting if betting is not None else power_betting_class(),
        stat=stat,
        rho=rho,
        seed=seed,
        weights_kind=kind,
    )


def conformal_e_detector(
    *,
    gamma: float,
    n_max: int,
    weights: Literal["unit", "polynomial", "near_harmonic"] | Array = "unit",
    eta: float = DEFAULT_ETA,
    betting: PowerBetting | None = None,
    stat: Literal["sum", "max"] = "sum",
    rho: float = DEFAULT_RHO,
    seed: int = 0,
) -> ConformalRestartDetector:
    """ARL-regime factory: conformal e-detector alarm at gamma (Thm. 2.7(2)).

    Default unit weights give ARL >= gamma and the linear early-alarm bound
    P(tau <= m) <= m/gamma (their Eq. (19)-(20)); with the null-bet ``rho``
    (their Prop. 4.10 / Thm. 3.10) the sum statistic stops surely by
    ceil(gamma/rho) with gamma <= E tau <= ceil(gamma/rho). Default
    ``stat="sum"`` mirrors their Thm. 3.10 / Fig. 2 ARL construction.
    """
    g = _check_gamma(gamma)
    n = _check_n_max(n_max)
    if isinstance(weights, str):
        if weights not in ("unit", "polynomial", "near_harmonic"):
            raise ValueError("weights kind must be 'unit', 'polynomial', or 'near_harmonic'")
        w = restart_weights(weights, n, eta=eta)
        kind = weights if weights == "unit" else f"{weights}(eta={eta:g})"
    else:
        w = np.asarray(weights, dtype=float).ravel()
        kind = "custom"
    if w.shape[0] != n or not bool(np.all(np.isfinite(w))) or bool(np.any(w < 0.0)):
        raise ValueError("weights must be an (n_max,) nonnegative finite array")
    if float(np.max(w)) > 1.0 + 1e-12:
        raise ValueError("ARL control at level gamma requires ||w||_inf <= 1 (Thm. 2.7(2))")
    if stat == "max" and float(rho) > 0.0:
        raise ValueError(
            "the null-bet sure-stopping guarantee (Prop. 4.10) is stated for the sum "
            "statistic; use rho=0 with stat='max'"
        )
    return ConformalRestartDetector(
        n_max=n,
        threshold=g,
        weights=w,
        betting=betting if betting is not None else power_betting_class(),
        stat=stat,
        rho=rho,
        seed=seed,
        weights_kind=kind,
    )


# ---------------------------------------------------------------------------
# existing-method comparators (Vovk 2021 CTM/CUSUM/SR; their Eq. (9)-(13))
# ---------------------------------------------------------------------------


def power_log_bets(
    p: Array | Sequence[float], theta: float, side: Literal["left", "right"] = "left"
) -> Array:
    """log f_theta(p_t) path for a single power density (their Eq. (9) bet)."""
    pa = _check_p_array(p)
    th = float(theta)
    if not np.isfinite(th) or not 0.0 < th <= 1.0:
        raise ValueError("theta must lie in (0, 1]")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    q = pa if side == "left" else 1.0 - pa
    qc = np.clip(q, _P_FLOOR, 1.0)
    return np.asarray(float(np.log(th)) + (th - 1.0) * np.log(qc), dtype=float)


def ctm_log_wealth(
    p: Array | Sequence[float], theta: float = CTM_THETA, side: Literal["left", "right"] = "left"
) -> Array:
    """Vovk (2021) conformal test martingale log S_n = sum log f(p_i) (Eq. (9))."""
    return np.asarray(np.cumsum(power_log_bets(p, theta, side)), dtype=float)


def cmm_log_wealth(p: Array | Sequence[float], betting: PowerBetting) -> Array:
    """Conformal mixture martingale log M_n = log int prod f_theta dPi (Eq. (11))."""
    if not isinstance(betting, PowerBetting):
        raise ValueError("betting must be a PowerBetting instance")
    log_f = betting.log_bets(_check_p_array(p))  # (T, K)
    cum = np.cumsum(np.clip(log_f, -_LOG_CAP, _LOG_CAP), axis=0)
    return np.asarray(logsumexp(cum + betting.log_prior[None, :], axis=1), dtype=float)


def vovk_cusum_stat(log_wealth: Array | Sequence[float]) -> Array:
    """Vovk conformal CUSUM statistic max_{i<=n} S_n/S_i (their Eq. (10)).

    Input is the log wealth path of a CTM/CMM (S_0 = 1); output is the
    statistic path (>= 1). ARL-valid at threshold gamma (Vovk 2021).
    """
    lw = np.asarray(log_wealth, dtype=float).ravel()
    if lw.size == 0 or not bool(np.all(np.isfinite(lw))):
        raise ValueError("log wealth path must be non-empty and finite")
    acc = np.minimum.accumulate(np.concatenate([np.zeros(1, dtype=float), lw]))
    # stat[j] = exp(lw[j] - min(0, lw[0..j])) = max_{i<=n} S_n/S_i with S_0 = 1.
    return np.asarray(np.exp(np.minimum(lw - acc[1:], _LOG_CAP)), dtype=float)


def vovk_sr_stat(log_bets: Array | Sequence[float]) -> Array:
    """Vovk conformal Shiryaev-Roberts statistic sum_{i=1}^{n-1} S_n/S_i (Eq. (10)).

    Built from the one-step log-bet path log f(p_n) via the recursion
    R_n = f_n (1 + R_{n-1}), R_1 = 0 (log space). ARL-valid at threshold
    gamma; P(tau <= m) <= m/gamma since the statistic is an e-detector
    (cf. quant_fund.metrics.e_detectors module doc).
    """
    lb = np.asarray(log_bets, dtype=float).ravel()
    if lb.size == 0 or not bool(np.all(np.isfinite(lb))):
        raise ValueError("log bet path must be non-empty and finite")
    out = np.zeros(lb.shape[0], dtype=float)
    log_r = -np.inf
    for n in range(1, lb.shape[0]):
        log_r = float(lb[n]) + float(np.logaddexp(0.0, log_r))
        out[n] = float(np.exp(min(log_r, _LOG_CAP)))
    return out


# ---------------------------------------------------------------------------
# SYNTHETIC streams and the matched comparison harness
# ---------------------------------------------------------------------------


def synthetic_changepoint_scores(
    n_pre: int, n_post: int, *, shift: float = 1.2, scale: float = 1.0, seed: int = 0
) -> Array:
    """SYNTHETIC stream: n_pre draws N(0, scale^2), then n_post draws N(shift, scale^2).

    The paper's Sec. 6 data-generating process (P_0 = N(0,1), P_1 = N(1.2, 1)
    by default); ``n_pre`` is the changepoint T (their convention: the last
    pre-change index). Seeded ⇒ deterministic. SYNTHETIC only — a
    correctness/study fixture, never market data or market evidence.
    """
    n0, n1 = int(n_pre), int(n_post)
    if n0 < 1 or n1 < 1:
        raise ValueError("n_pre and n_post must both be >= 1")
    mu = float(shift)
    sc = float(scale)
    if not np.isfinite(mu):
        raise ValueError("shift must be finite")
    if not np.isfinite(sc) or sc <= 0.0:
        raise ValueError("scale must be finite and > 0")
    rng = np.random.default_rng(seed)
    pre = rng.normal(0.0, sc, n0)
    post = rng.normal(mu, sc, n1)
    return np.asarray(np.concatenate([pre, post]), dtype=float)


@dataclass(frozen=True)
class DelayStudyRow:
    """Per-method detection-delay summary at a planted changepoint.

    Delay convention mirrors their Sec. 6: tau is the 1-based alarm time;
    a replicate with tau <= T is a FALSE ALARM and is excluded from the delay
    summary (reported separately as ``false_alarm_rate``); over the remaining
    replicates the capped delay is (tau - T) when an alarm fired within the
    horizon and ``cap`` otherwise. ``median_delay_over_log_t`` is the
    normalized quantity of their Fig. 5: flat ~ constant for Theta(log T)
    delay, diverging for polynomial delay.
    """

    method: str
    regime: str
    changepoint: int
    cap: int
    n_reps: int
    false_alarm_rate: float
    detection_rate: float
    median_capped_delay: float
    mean_capped_delay: float
    median_delay_over_log_t: float


def summarize_delays(
    method: str,
    regime: str,
    alarm_indices: Sequence[int | None],
    changepoint: int,
    cap: int,
) -> DelayStudyRow:
    """Aggregate per-replicate 0-based alarm indices into a :class:`DelayStudyRow`.

    ``alarm_indices[i]`` is the first-crossing index on replicate i (None if
    the stream ended unalarmed); the horizon is changepoint + cap. Raises if
    every replicate false-alarmed (the delay summary would be empty).
    """
    t0, c = int(changepoint), int(cap)
    if t0 < 1 or c < 1:
        raise ValueError("changepoint and cap must be >= 1")
    if regime not in ("pfa", "arl"):
        raise ValueError("regime must be 'pfa' or 'arl'")
    idx = list(alarm_indices)
    if len(idx) == 0:
        raise ValueError("alarm_indices must be non-empty")
    horizon = t0 + c
    for a in idx:
        if a is not None and not (isinstance(a, int) and 0 <= a < horizon):
            raise ValueError("alarm indices must be None or lie in [0, changepoint + cap)")
    fa = [a is not None and a < t0 for a in idx]
    fa_rate = float(np.mean(np.asarray(fa, dtype=float)))
    clean = [a for a, f in zip(idx, fa, strict=True) if not f]
    if len(clean) == 0:
        raise ValueError("all replicates false-alarmed; delay summary undefined")
    delays = np.asarray(
        [float(a + 1 - t0) if a is not None else float(c) for a in clean], dtype=float
    )
    detected = np.asarray([a is not None for a in clean], dtype=float)
    med = float(np.median(delays))
    return DelayStudyRow(
        method=str(method),
        regime=str(regime),
        changepoint=t0,
        cap=c,
        n_reps=len(idx),
        false_alarm_rate=fa_rate,
        detection_rate=float(np.mean(detected)),
        median_capped_delay=med,
        mean_capped_delay=float(np.mean(delays)),
        median_delay_over_log_t=float(med / np.log(t0)),
    )


def _check_study_params(
    changepoint: int, cap: int, n_reps: int, methods: Sequence[str], valid: tuple[str, ...]
) -> None:
    t0, c, r = int(changepoint), int(cap), int(n_reps)
    if t0 < 1 or c < 1:
        raise ValueError("changepoint and cap must be >= 1")
    if r < 1:
        raise ValueError("n_reps must be >= 1")
    if t0 + c > _N_MAX_LIMIT:
        raise ValueError(f"horizon {t0 + c} exceeds the {_N_MAX_LIMIT} limit")
    if len(methods) == 0:
        raise ValueError("methods must be non-empty")
    for m in methods:
        if m not in valid:
            raise ValueError(f"unknown method {m!r}; expected a subset of {valid}")


def _pfa_method_alarm(
    method: str,
    p: Array,
    *,
    alpha: float,
    betting: PowerBetting,
    eta: float,
    n_max: int,
    seed: int,
) -> int | None:
    thr = 1.0 / alpha
    if method == "vovk_ctm":
        return first_crossing_log(ctm_log_wealth(p, CTM_THETA, "left"), thr)
    if method == "cmm_grid":
        return first_crossing_log(cmm_log_wealth(p, betting), thr)
    if method == "repo_power_mixture":
        return first_crossing(mixture_martingale(p), thr)
    if method == "repo_jumper":
        return first_crossing(simple_jumper(p, jump=0.01), thr)
    if method in ("optimal_sum", "optimal_max"):
        det = conformal_e_process(
            alpha=alpha,
            n_max=n_max,
            weights="polynomial",
            eta=eta,
            betting=betting,
            stat="sum" if method == "optimal_sum" else "max",
            seed=seed,
        )
        return det.run_p_values(p, stop_on_alarm=True).alarm_index
    raise ValueError(f"unknown PFA method {method!r}")


def _arl_method_alarm(
    method: str,
    p: Array,
    scores: Array,
    *,
    gamma: float,
    rho: float,
    betting: PowerBetting,
    n_max: int,
    seed: int,
) -> int | None:
    if method in ("vovk_sr_ctm", "vovk_cusum_ctm"):
        log_w = ctm_log_wealth(p, CTM_THETA, "left")
        stat = (
            vovk_sr_stat(power_log_bets(p, CTM_THETA, "left"))
            if method == "vovk_sr_ctm"
            else vovk_cusum_stat(log_w)
        )
        return first_crossing(stat, gamma)
    if method in ("vovk_sr_cmm", "vovk_cusum_cmm"):
        log_m = cmm_log_wealth(p, betting)
        if method == "vovk_cusum_cmm":
            return first_crossing(vovk_cusum_stat(log_m), gamma)
        # SR over the mixture path (their Eq. (10) applied to the CMM M_n of
        # Eq. (11), the comparator of their Fig. 2 / Eq. (13) up to the i = 0
        # term): R_n = sum_{i=1}^{n-1} M_n/M_i telescopes out of the increment
        # ratios r_j = M_j/M_{j-1} via R_n = r_n (1 + R_{n-1}).
        ratios = np.empty_like(log_m)
        ratios[0] = log_m[0]
        ratios[1:] = log_m[1:] - log_m[:-1]
        return first_crossing(vovk_sr_stat(ratios), gamma)
    if method == "repo_mixture_sr":
        gdet = EDetectorGaussian(sigma=1.0, alpha=1.0 / gamma)
        return run_detector(gdet, scores, 1.0 / gamma).alarm_time
    if method in ("optimal_sum", "optimal_max"):
        rdet = conformal_e_detector(
            gamma=gamma,
            n_max=n_max,
            betting=betting,
            stat="sum" if method == "optimal_sum" else "max",
            rho=rho if method == "optimal_sum" else 0.0,
            seed=seed,
        )
        return rdet.run_p_values(p, stop_on_alarm=True).alarm_index
    raise ValueError(f"unknown ARL method {method!r}")


def pfa_delay_study(
    changepoint: int,
    *,
    cap_multiple: float = 5.0,
    n_reps: int = 30,
    alpha: float = 0.05,
    shift: float = 1.2,
    seed: int = 0,
    methods: Sequence[str] = PFA_METHODS,
    betting: PowerBetting | None = None,
    eta: float = DEFAULT_ETA,
) -> list[DelayStudyRow]:
    """Matched-PFA delay comparison at a planted changepoint (their Fig. 1/5 protocol).

    Every method alarms at the SAME Ville threshold 1/alpha, so all are
    PFA-valid at level alpha under the exchangeable null and the delays are
    matched-false-alarm comparisons. SYNTHETIC N(0,1) -> N(shift,1) streams,
    monitored until T + floor(cap_multiple*T); all methods consume the
    IDENTICAL seeded p-value sequence per replicate. Deterministic in ``seed``.
    """
    t0 = int(changepoint)
    cm = float(cap_multiple)
    if not np.isfinite(cm) or cm < 1.0:
        raise ValueError("cap_multiple must be finite and >= 1")
    cap = int(np.floor(cm * t0))
    a = _check_alpha(alpha)
    _check_study_params(t0, cap, n_reps, methods, PFA_METHODS)
    bet = betting if betting is not None else power_betting_class()
    horizon = t0 + cap
    alarms: dict[str, list[int | None]] = {m: [] for m in methods}
    for rep in range(int(n_reps)):
        scores = synthetic_changepoint_scores(t0, cap, shift=shift, seed=seed + rep)
        p = conformal_p_values(scores, seed=seed + _RANDOMIZER_SEED_OFFSET + rep)
        for m in methods:
            alarms[m].append(
                _pfa_method_alarm(
                    m, p, alpha=a, betting=bet, eta=eta, n_max=horizon, seed=seed + rep
                )
            )
    return [summarize_delays(m, "pfa", alarms[m], t0, cap) for m in methods]


def arl_delay_study(
    changepoint: int,
    *,
    gamma: float = 1.0e4,
    cap: int = 1500,
    n_reps: int = 25,
    shift: float = 1.2,
    seed: int = 0,
    methods: Sequence[str] = ARL_METHODS,
    rho: float = DEFAULT_RHO,
    betting: PowerBetting | None = None,
) -> list[DelayStudyRow]:
    """Matched-ARL delay comparison (their Fig. 2 protocol: gamma = 1e4, cap 1500).

    Every method alarms at the SAME threshold gamma and is ARL-valid at level
    gamma (Vovk's SR/CUSUM inherit E tau >= gamma; the optimal e-detectors
    from Thm. 2.7(2); the repo parametric mixture-SR from its Ville bound).
    ``repo_mixture_sr`` is NOT distribution-free — it knows sigma = 1 and the
    mean-shift direction — and serves as a parametric reference, not a
    fairness-matched competitor. SYNTHETIC, seeded, deterministic.
    """
    t0 = int(changepoint)
    g = _check_gamma(gamma)
    c = int(cap)
    _check_study_params(t0, c, n_reps, methods, ARL_METHODS)
    r = float(rho)
    if not np.isfinite(r) or not 0.0 <= r < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    bet = betting if betting is not None else power_betting_class()
    alarms: dict[str, list[int | None]] = {m: [] for m in methods}
    for rep in range(int(n_reps)):
        scores = synthetic_changepoint_scores(t0, c, shift=shift, seed=seed + rep)
        p = conformal_p_values(scores, seed=seed + _RANDOMIZER_SEED_OFFSET + rep)
        for m in methods:
            alarms[m].append(
                _arl_method_alarm(
                    m,
                    p,
                    scores,
                    gamma=g,
                    rho=r,
                    betting=bet,
                    n_max=t0 + c,
                    seed=seed + rep,
                )
            )
    return [summarize_delays(m, "arl", alarms[m], t0, c) for m in methods]


def delay_growth_study(
    changepoints: Sequence[int] = (200, 400, 800),
    *,
    cap_multiple: float = 5.0,
    n_reps: int = 20,
    alpha: float = 0.05,
    shift: float = 1.2,
    seed: int = 0,
    methods: Sequence[str] = ("vovk_ctm", "cmm_grid", "optimal_sum", "optimal_max"),
    betting: PowerBetting | None = None,
    eta: float = DEFAULT_ETA,
) -> list[DelayStudyRow]:
    """Delay-vs-changepoint sweep exhibiting the polynomial-vs-log gap (their Fig. 5).

    Runs :func:`pfa_delay_study` at each changepoint in ``changepoints``.
    Expected finite-sample pattern (ILLUSTRATES, does not prove, the rates):
    existing CTM/CMM median capped delay grows ~ linearly in T (their
    Examples 2.2/2.4), while the optimal restarted statistics stay on a
    ~ log T scale (their Cor. 3.6), so ``median_delay_over_log_t`` diverges
    for the former and stays flat for the latter.
    """
    ts = [int(t) for t in changepoints]
    if len(ts) == 0 or any(t < 1 for t in ts):
        raise ValueError("changepoints must be a nonempty sequence of positive ints")
    rows: list[DelayStudyRow] = []
    for t0 in ts:
        rows.extend(
            pfa_delay_study(
                t0,
                cap_multiple=cap_multiple,
                n_reps=n_reps,
                alpha=alpha,
                shift=shift,
                seed=seed,
                methods=methods,
                betting=betting,
                eta=eta,
            )
        )
    return rows


def false_alarm_study(
    *,
    regime: Literal["pfa", "arl"] = "pfa",
    n_reps: int = 100,
    horizon: int = 1000,
    alpha: float = 0.05,
    gamma: float = 100.0,
    rho: float = 0.1,
    null: Literal["gaussian", "uniform", "ties"] = "gaussian",
    seed: int = 0,
    methods: Sequence[str] | None = None,
    betting: PowerBetting | None = None,
    eta: float = DEFAULT_ETA,
) -> dict[str, float]:
    """Empirical finite-horizon false-alarm rates under an exchangeable null.

    SYNTHETIC i.i.d. no-change streams (``null``: N(0,1), U(0,1), or a
    3-valued heavily-tied discrete stream exercising the p-value randomizer);
    the reported rate is P(tau <= horizon) — the EVER-CROSSING event, which is
    exactly the any-stopping-time guarantee (a crossing-based alarm stopped at
    any data-dependent time can only fire less often). Regime "pfa" rates must
    stay <= alpha (+ MC slack); regime "arl" rates must stay <=
    horizon * ||w||_inf / gamma (+ slack) by the linear early-alarm bound
    (their Eq. (19)); both are asserted in the tests, never assumed here.
    """
    h = _check_n_max(horizon)
    reps = int(n_reps)
    if reps < 1:
        raise ValueError("n_reps must be >= 1")
    if regime == "pfa":
        a = _check_alpha(alpha)
        valid = _PFA_NULL_METHODS
        ms = tuple(valid if methods is None else methods)
        _check_study_params(1, h - 1, reps, ms, valid)
    elif regime == "arl":
        g = _check_gamma(gamma)
        valid = _ARL_NULL_METHODS
        ms = tuple(valid if methods is None else methods)
        _check_study_params(1, h - 1, reps, ms, valid)
    else:
        raise ValueError("regime must be 'pfa' or 'arl'")
    bet = betting if betting is not None else power_betting_class()
    alarms: dict[str, int] = {m: 0 for m in ms}
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        if null == "gaussian":
            scores = np.asarray(rng.standard_normal(h), dtype=float)
        elif null == "uniform":
            scores = np.asarray(rng.random(h), dtype=float)
        elif null == "ties":
            scores = np.asarray(rng.integers(0, 3, h), dtype=float)
        else:
            raise ValueError("null must be one of 'gaussian', 'uniform', 'ties'")
        p = conformal_p_values(scores, seed=seed + _RANDOMIZER_SEED_OFFSET + rep)
        for m in ms:
            if regime == "pfa":
                idx = _pfa_method_alarm(
                    m, p, alpha=a, betting=bet, eta=eta, n_max=h, seed=seed + rep
                )
            else:
                idx = _arl_method_alarm(
                    m, p, scores, gamma=g, rho=rho, betting=bet, n_max=h, seed=seed + rep
                )
            alarms[m] += int(idx is not None)
    return {m: float(alarms[m]) / float(reps) for m in ms}


def arl_calibration_study(
    *,
    gamma: float = 100.0,
    rho: float = 0.1,
    n_reps: int = 100,
    early_ms: Sequence[int] = (5, 10, 25),
    null: Literal["gaussian", "uniform", "ties"] = "gaussian",
    seed: int = 0,
    betting: PowerBetting | None = None,
) -> dict[str, object]:
    """ARL-regime calibration of the optimal e-detector under the null.

    Runs the unit-weight sum e-detector with null-bet ``rho`` (their Thm.
    3.10 construction) on SYNTHETIC no-change streams, horizon
    ceil(gamma/rho) (sure stopping by Prop. 4.10). Returns: ``empirical_arl``
    (mean tau; theory: >= gamma), ``max_tau`` and ``sure_stop_bound`` (prop.
    4.10: max_tau <= ceil(gamma/rho)), and ``early_alarms`` — a list of
    (m, empirical P(tau <= m), bound m/gamma) checking the linear
    early-alarm inequality of their Eq. (19)/(25).
    """
    g = _check_gamma(gamma)
    r = float(rho)
    if not np.isfinite(r) or not 0.0 < r < 1.0:
        raise ValueError("rho must lie in the open interval (0, 1) for sure stopping")
    reps = int(n_reps)
    if reps < 1:
        raise ValueError("n_reps must be >= 1")
    horizon = int(np.ceil(g / r))
    if horizon > _N_MAX_LIMIT:
        raise ValueError(f"sure-stop horizon {horizon} exceeds the {_N_MAX_LIMIT} limit")
    ms = [int(m) for m in early_ms]
    if any(m < 1 or m >= horizon for m in ms):
        raise ValueError("early_ms must lie in [1, horizon)")
    bet = betting if betting is not None else power_betting_class()
    taus: list[int] = []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        if null == "gaussian":
            scores = np.asarray(rng.standard_normal(horizon), dtype=float)
        elif null == "uniform":
            scores = np.asarray(rng.random(horizon), dtype=float)
        elif null == "ties":
            scores = np.asarray(rng.integers(0, 3, horizon), dtype=float)
        else:
            raise ValueError("null must be one of 'gaussian', 'uniform', 'ties'")
        p = conformal_p_values(scores, seed=seed + _RANDOMIZER_SEED_OFFSET + rep)
        det = conformal_e_detector(
            gamma=g, n_max=horizon, betting=bet, stat="sum", rho=r, seed=seed + rep
        )
        res = det.run_p_values(p, stop_on_alarm=True)
        taus.append(res.alarm_index + 1 if res.alarm_index is not None else horizon)
    tau_arr = np.asarray(taus, dtype=float)
    early = [(m, float(np.mean(tau_arr <= float(m))), float(m) / g) for m in ms]
    return {
        "empirical_arl": float(np.mean(tau_arr)),
        "gamma": g,
        "rho": r,
        "n_reps": reps,
        "horizon": horizon,
        "sure_stop_bound": horizon,
        "max_tau": int(np.max(tau_arr)),
        "early_alarms": early,
    }
