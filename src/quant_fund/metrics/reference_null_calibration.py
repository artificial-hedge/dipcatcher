"""Reference-null calibrated rejection thresholds for e-processes.

Ding, Wei, Zhu & Dai (2026), "Reference-Null Calibrated Thresholds for
E-Processes with Applications to Conformal Martingales", arXiv:2609.32678
(45 pp., math.ST). Ville's inequality gives the universal e-process boundary
P_0(sup_t M_t >= 1/alpha) <= alpha, valid for EVERY nonnegative e-process but
calibrated against the least favorable class (Markov/Ville worst case). The
paper's reference-null calibration replaces it, at a FIXED finite horizon T,
by the empirical upper quantile of the e-process path maximum under
independent null reference trajectories:

* Static e-values (their Sec. 2.1, Theorem 2.1): with B reference e-values
  exchangeable with the test e-value, k_{alpha,B} = ceil((1-alpha)(B+1)) and
  c_hat = E^ref_(k) reject when E^test > c_hat (strict crossing) with
  E_{P0}[phi] <= alpha, and >= alpha - 1/(B+1) when the common null law is
  continuous. c*_alpha <= 1/alpha, so the calibrated rule is sharper than
  Markov whenever the null law is not least favorable.
* Finite-horizon e-processes (their Sec. 2.2, Theorem 2.2, Algorithm 1): the
  statistic is the normalized path maximum R_g = max_{0<=t<=T} M_t / g_t for
  a deterministic positive boundary template g fixed independently of the
  calibration sample; c_hat is the k-th order statistic of B reference R_g
  values (convention R_(B+1) = +inf, i.e. "never reject", when k = B+1) and
  the rule rejects at nu = inf{t : M_t > c_hat * g_t}. If the test trajectory
  is exchangeable with the B reference trajectories under P_0, then
  P_0(nu <= T) <= alpha — a purely rank-based, finite-sample guarantee that
  includes the calibration randomness and needs no analytic null law. The
  constant template g_t = 1 is their recommended default (Remark 2.5); a
  nonconstant template pays the aspect-ratio factor K_g = max g / min g in
  the high-probability reduction bound of their Theorem 2.4, and the
  empirical multiplier tracks the oracle quantile at the O(B^{-1/2}) DKW
  rate (their Prop. 2.6).
* Conformal specialization (their Sec. 3): under ANY exchangeable score law
  the sequential smoothed conformal p-values (their Eq. 2, exactly the output
  of quant_fund.metrics.conformal_martingale.conformal_p_values) are i.i.d.
  Unif(0,1) — the pivotal null law of their Sec. 3.1 — so the reference bank
  is B streams of T i.i.d. uniforms passed through the SAME construction.
  Constructions implemented here: (i) the adaptive Krichevsky-Trofimov
  histogram betting martingale of their Sec. 3.2 / Algorithm 2 with betting
  factor f_t = J (N_{j_t}(t-1) + 1/2) / (t - 1 + J/2) on J equal-width bins
  (Prop. 3.2: a nonnegative martingale under the exchangeability null; the
  half-count smoothing makes a first visit to an empty bin contribute a
  factor <= 1, so zero-count bins can never explode the wealth); and (ii)
  the restart mixture of their Sec. 3.3 / Eq. (5) / Algorithm 3,
  M_t = sum_{s<t} pi_s M_t^(s) + sum_{s>=t} pi_s, where component s restarts
  both wealth and KT counts at time s and the trailing weight tail keeps
  M_0 = 1 (Prop. 3.3: again a nonnegative martingale, hence an e-process).
  The weights pi_s are ANY deterministic nonnegative sequence summing to one;
  Remark 3.4 recommends the finite-horizon uniform choice pi_s = 1/T, which
  uniquely minimizes the worst-case aggregation penalty log(1/pi_tau) = log T
  at an unknown change point tau (a truncated geometric alternative is
  provided for early-shift priors). Their Sec. 4 quantifies the gain: the
  rejection boundary enters the power bounds only through log B_T, so a
  calibrated boundary c_hat = rho/alpha shrinks the evidence requirement by
  lambda log(1/rho) (Thm 4.1) and buys a detection-delay gain of order
  (log(1/rho) - overshoot) / growth-rate (Thms 4.9, 4.11, Eqs. 30).

HONESTY — the guarantee is only as good as the reference null. Theorem 2.2
assumes EXCHANGEABILITY of the test trajectory with the B reference
trajectories under the null; for the conformal specialization this holds
when the test p-values come from an exchangeable score stream and the
reference bank is i.i.d. uniforms of the SAME horizon T, generated
independently of the test data, with every design choice (T, J, construction,
weights) fixed before calibration (their Sec. 3.4). The paper's own stated
limitation (their Sec. 7): "The present framework requires a reference
sample that accurately represents the null distribution relevant to the
prospective test, and the finite-horizon calibrated boundary depends on the
monitoring horizon and other design choices fixed before calibration." A
contaminated or mis-scoped reference bank silently destroys the type-I
guarantee; the input validation below is fail-closed on shape/finiteness and
on the documented minimum B, but it CANNOT certify null quality. The
calibrated boundary is horizon-specific, not anytime-valid: it controls
crossing over monitoring windows of length <= T (shorter windows are
conservative because the path maximum over a prefix is smaller); longer
monitoring requires a fresh calibration, and the universal 1/alpha Ville
boundary remains the anytime-valid fallback. This module composes with — and
never modifies — the Ville-thresholded alarms in
quant_fund.metrics.conformal_martingale / watch / e_detectors: calibrated and
Ville rules are applied to the SAME e-process path and differ only in the
crossing boundary (their Sec. 5.1 paired design).

All Monte-Carlo evidence in the accompanying tests uses seeded SYNTHETIC
streams and is a correctness check of the error control, never market
evidence. Type-I rates, threshold sharpness, and detection delays are proper
diagnostics; no Sharpe / P&L.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "CalibratedThreshold",
    "calibrate_conformal_martingale",
    "calibrate_e_process",
    "conformal_null_path_maxima",
    "crossing_time",
    "detection_delay_summary",
    "kt_betting_factors",
    "kt_histogram_martingale",
    "min_reference_nulls",
    "normalized_path_maxima",
    "reference_null_threshold",
    "restart_mixture_kt_martingale",
    "restart_weights",
]

# Same wealth cap convention as quant_fund.metrics.watch (_E_MAX): log-wealth
# is clipped before exponentiating so strongly-drifting paths saturate at a
# finite value instead of overflowing to inf. Any realistic boundary is far
# below the cap, so crossing times are unaffected.
_E_MAX = 1e300
_LOG_E_MAX = math.log(_E_MAX)

# Floating-point guard for ceil((1-alpha)*(B+1)): exact-integer products can
# land a few ulps off, and the paper's own experiments use the exact integer
# rank (B=4999, alpha=0.05 -> k=4750, with exactly 249 reference maxima above
# the threshold under strict crossing, their Sec. 6.6). A product within this
# relative tolerance of an integer is snapped to it; otherwise plain ceil.
# Snapping DOWN can only occur when the true product is less than the
# tolerance above the integer, which changes the crossing bound by at most
# 1/(B+1) — negligible and documented.
_RANK_TOL = 1e-9

_CONSTRUCTIONS = ("kt", "restart_mixture")


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    return a


def _rank_k(alpha: float, b: int) -> int:
    """k_{alpha,B} = ceil((1-alpha)(B+1)) with an integer-snap float guard."""
    x = (1.0 - alpha) * (b + 1)
    r = float(round(x))
    if abs(x - r) <= _RANK_TOL * max(1.0, x):
        return int(r)
    return int(math.ceil(x))


def min_reference_nulls(alpha: float) -> int:
    """Documented minimum reference-bank size B for a finite calibrated threshold.

    k_{alpha,B} = ceil((1-alpha)(B+1)) <= B holds exactly when B >= 1/alpha - 1,
    so the minimum is ceil(1/alpha - 1) (19 at alpha=0.05, 99 at alpha=0.01).
    Below it, Algorithm 1 lines 9-10 of Ding et al. (2026) set the threshold
    to +inf (the rule never rejects); :func:`reference_null_threshold`
    fail-closes with a ValueError instead of returning a vacuous boundary.
    """
    a = _check_alpha(alpha)
    return max(1, int(math.ceil(1.0 / a - 1.0 - _RANK_TOL)))


@dataclass(frozen=True)
class CalibratedThreshold:
    """Reference-null calibrated rejection threshold (Ding et al. 2026, Alg. 1).

    ``threshold`` is c_hat_{alpha,B}: the k-th smallest of B exchangeable
    reference-null path statistics, k = k_order. Reject when the test
    statistic STRICTLY exceeds ``threshold`` (their Sec. 3.4 / 5.1 strict
    crossing rule). ``sharpness`` is the boundary-reduction ratio
    c_hat / (1/alpha) reported in their Table 1 ("threshold/Ville"); the
    threshold is NOT truncated at 1/alpha (their Sec. 5.1), so ``sharpness``
    can in principle exceed 1 on an unlucky reference bank — validity never
    relies on sharpness < 1, only on exchangeability (their Theorem 2.2).
    """

    threshold: float
    alpha: float
    n_reference: int
    k_order: int
    ville_threshold: float
    method: str

    @property
    def sharpness(self) -> float:
        """c_hat * alpha = c_hat / (1/alpha); values < 1 are sharper than Ville."""
        return float(self.threshold * self.alpha)


def reference_null_threshold(
    reference_statistics: Array | Sequence[float], alpha: float
) -> CalibratedThreshold:
    """Calibrated sharper threshold from independent null reference statistics.

    ``reference_statistics`` are B values of the path statistic computed from
    B INDEPENDENT null reference trajectories by applying the identical
    e-process construction: the reference e-values themselves for a static
    e-value test (their Sec. 2.1), or the (normalized) finite-horizon path
    maxima max_t M_t/g_t for an e-process (their Sec. 2.2; see
    :func:`normalized_path_maxima`). Returns c_hat = the k-th order statistic
    with k = ceil((1-alpha)(B+1)).

    Guarantee (their Theorems 2.1/2.2): if the test statistic is exchangeable
    with the B reference statistics under the null, the strict-crossing rule
    "reject when test statistic > c_hat" has type-I error <= alpha at ANY B >=
    :func:`min_reference_nulls` — a finite-sample rank guarantee that
    includes the calibration randomness, needs no analytic null law, and is
    near-exact (>= alpha - 1/(B+1)) for continuous null laws. Fail-closed:
    raises on empty input, negative/non-finite statistics, invalid alpha, or
    B below the documented minimum (the regime where Algorithm 1 would set
    the threshold to +inf and never reject).
    """
    a = _check_alpha(alpha)
    stats = np.asarray(reference_statistics, dtype=float).ravel()
    if stats.size == 0:
        raise ValueError("reference_statistics must be non-empty")
    if not bool(np.all(np.isfinite(stats))):
        raise ValueError("reference_statistics must be finite (NaN/inf rejected)")
    if bool(np.any(stats < 0.0)):
        raise ValueError("reference_statistics must be nonnegative (e-values / path maxima)")
    b = int(stats.size)
    b_min = min_reference_nulls(a)
    if b < b_min:
        raise ValueError(
            f"too few null reference samples: B={b} < min_reference_nulls(alpha={a})={b_min}; "
            "below the minimum the calibrated threshold is +inf (Algorithm 1 never rejects)"
        )
    k = _rank_k(a, b)
    if k > b:
        # Unreachable given the B >= 1/alpha - 1 check; kept fail-closed.
        raise ValueError(f"rank k={k} exceeds B={b}; increase the reference bank")
    ordered = np.sort(stats)
    return CalibratedThreshold(
        threshold=float(ordered[k - 1]),
        alpha=a,
        n_reference=b,
        k_order=k,
        ville_threshold=1.0 / a,
        method="reference-null:order-statistic",
    )


def normalized_path_maxima(paths: Array, template: Array | Sequence[float] | None = None) -> Array:
    """Normalized reference path maxima R_g = max_{0<=t<=T} M_t / g_t (their Sec. 2.2).

    ``paths`` is (B, L) reference e-process trajectories (L = T+1 values,
    column 0 the initial wealth M_0 = 1). ``template`` is the deterministic
    positive boundary template g of length L, fixed independently of the
    calibration sample; the default is the constant template g_t = 1 that
    Remark 2.5 recommends (aspect ratio K_g = 1 — an unnecessarily small g_t
    at one time point inflates the whole calibrated boundary by K_g, their
    Theorem 2.4). Returns the (B,) row maxima for
    :func:`reference_null_threshold`.
    """
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("paths must be a nonempty 2-D (B, L) array of reference trajectories")
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr < 0.0)):
        raise ValueError("reference paths must be finite and nonnegative")
    if template is None:
        return np.asarray(np.max(arr, axis=-1), dtype=np.float64)
    g = np.asarray(template, dtype=float).ravel()
    if g.shape[0] != arr.shape[1]:
        raise ValueError("template must have the same length as each reference path")
    if not bool(np.all(np.isfinite(g))) or bool(np.any(g <= 0.0)):
        raise ValueError("template must be finite and strictly positive (their Sec. 2.2)")
    return np.asarray(np.max(arr / g[None, :], axis=-1), dtype=np.float64)


def calibrate_e_process(
    reference_null_data: Array,
    e_process: Callable[[Array], Array],
    alpha: float,
    *,
    template: Array | Sequence[float] | None = None,
) -> CalibratedThreshold:
    """Full Algorithm 1 for a generic e-process construction on null reference data.

    ``reference_null_data`` is (B, T) independent null reference trajectories;
    ``e_process`` maps one (T,) trajectory to its e-process path (1-D, length
    >= 2, entry 0 the initial wealth M_0 = 1). The SAME construction is
    applied to every trajectory, the normalized path maxima are taken, and
    :func:`reference_null_threshold` returns c_hat. Design choices must be
    fixed independently of the calibration sample, and the test trajectory
    must be exchangeable with the reference trajectories under the null
    (their Theorem 2.2) — the guarantee is void otherwise (see module
    docstring). Deterministic given the inputs.
    """
    data = np.asarray(reference_null_data, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("reference_null_data must be a nonempty 2-D (B, T) array")
    if not bool(np.all(np.isfinite(data))):
        raise ValueError("reference_null_data must be finite")
    maxima = np.empty(data.shape[0], dtype=np.float64)
    for b in range(data.shape[0]):
        path = np.asarray(e_process(data[b]), dtype=float).ravel()
        if path.size < 2 or not bool(np.all(np.isfinite(path))) or bool(np.any(path < 0.0)):
            raise ValueError(
                "e_process must return a finite nonnegative path of length >= 2 (M_0 first)"
            )
        maxima[b] = float(normalized_path_maxima(path[None, :], template)[0])
    out = reference_null_threshold(maxima, alpha)
    return CalibratedThreshold(
        threshold=out.threshold,
        alpha=out.alpha,
        n_reference=out.n_reference,
        k_order=out.k_order,
        ville_threshold=out.ville_threshold,
        method="reference-null:e-process-algorithm1",
    )


def _check_p_values(p_values: Array | Sequence[float]) -> Array:
    p = np.asarray(p_values, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("p_values must be non-empty")
    if not bool(np.all(np.isfinite(p))) or bool(np.any(p < 0.0)) or bool(np.any(p > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    return p


def _check_n_bins(n_bins: int) -> int:
    j = int(n_bins)
    if j < 2:
        raise ValueError("n_bins must be >= 2 (their Sec. 3.2 fixes J >= 2)")
    return j


def _bin_indices(p: Array, n_bins: int) -> NDArray[np.int64]:
    """Bin I_j = [(j-1)/J, j/J), last bin closed: p = 1.0 maps to bin J-1."""
    idx = np.floor(p * n_bins).astype(np.int64)
    return np.asarray(np.clip(idx, 0, n_bins - 1), dtype=np.int64)


def kt_betting_factors(p_values: Array | Sequence[float], n_bins: int = 20) -> Array:
    """Predictable KT histogram betting factors f_t(p_t), t = 1..T (their Alg. 2).

    f_t(p_t) = J * (N_{j_t}(t-1) + 1/2) / (t - 1 + J/2) with N_j(t-1) the
    count of p_1..p_{t-1} in bin I_j and j_t the bin of p_t. The
    Krichevsky-Trofimov half-count smoothing (Krichevsky & Trofimov 1981;
    their Sec. 3.2) keeps every factor strictly positive: a FIRST visit to a
    previously empty bin contributes J / (2(t-1) + J) <= 1, so zero-count
    bins can never explode the martingale, and the first factor (t=1, all
    counts zero) is identically 1 — the KT predictor starts exactly uniform.
    """
    p = _check_p_values(p_values)
    j = _check_n_bins(n_bins)
    bins = _bin_indices(p, j)
    n = int(p.size)
    onehot = np.zeros((n, j), dtype=np.float64)
    onehot[np.arange(n), bins] = 1.0
    counts_before = np.empty((n, j), dtype=np.float64)
    counts_before[0] = 0.0
    np.cumsum(onehot[:-1], axis=0, out=counts_before[1:])
    tt = np.arange(n, dtype=np.float64)  # t - 1 for t = 1..T
    own = counts_before[np.arange(n), bins]
    return np.asarray(j * (own + 0.5) / (tt + j / 2.0), dtype=np.float64)


def kt_histogram_martingale(p_values: Array | Sequence[float], n_bins: int = 20) -> Array:
    """Unrestarted KT histogram conformal martingale path M_0..M_T (their Sec. 3.2).

    Returns the length-(T+1) path with M_0 = 1 and M_t = prod_{i<=t} f_i(p_i)
    for the KT betting factors of :func:`kt_betting_factors`. Under every
    exchangeable null this is a nonnegative martingale in the p-value
    filtration — hence an e-process (their Prop. 3.2) — so Ville gives
    P(sup_t M_t >= 1/alpha) <= alpha (the universal boundary used by
    quant_fund.metrics.conformal_martingale.martingale_alarm), while
    :func:`calibrate_conformal_martingale` builds the sharper horizon-T
    boundary from i.i.d.-uniform reference streams (their Sec. 3.4). Input
    is any conformal p-value stream — e.g. the output of
    quant_fund.metrics.conformal_martingale.conformal_p_values, whose
    sequential smoothed p-values match the paper's Eq. (2) exactly. Log-wealth
    saturates at log(1e300) (see _E_MAX); crossings of realistic boundaries
    are unaffected.
    """
    factors = kt_betting_factors(p_values, n_bins)
    log_wealth = np.minimum(np.cumsum(np.log(factors)), _LOG_E_MAX)
    return np.asarray(np.exp(np.concatenate([[0.0], log_wealth])), dtype=np.float64)


def restart_weights(horizon: int, kind: str = "uniform", gamma: float | None = None) -> Array:
    """Deterministic restart weights pi_s over s = 0..T-1, summing to 1 (their Sec. 3.3).

    "uniform": pi_s = 1/T — the finite-horizon recommendation of their
    Remark 3.4, which uniquely minimizes the worst-case aggregation penalty
    max_s log(1/pi_s) = log T at an unknown change point. "geometric":
    pi_s proportional to gamma^s for gamma in (0, 1), truncated at the
    horizon and renormalized — a valid deterministic weighting (Prop. 3.3
    allows any pi with sum 1) that front-loads restarts when early changes
    are expected, at the price of a larger penalty log(1/pi_tau) for late
    tau. NOTE: the brief for this lane said "geometric restart epochs"; the
    paper itself specifies general deterministic weights with the uniform
    recommendation, so uniform is the default here and geometric is opt-in.
    """
    t = int(horizon)
    if t < 1:
        raise ValueError("horizon must be >= 1")
    kind_s = str(kind)
    if kind_s == "uniform":
        return np.asarray(np.full(t, 1.0 / t), dtype=np.float64)
    if kind_s == "geometric":
        if gamma is None:
            raise ValueError("geometric restart weights require gamma in (0, 1)")
        g = float(gamma)
        if not np.isfinite(g) or not 0.0 < g < 1.0:
            raise ValueError("gamma must be finite and in (0, 1)")
        w = np.power(g, np.arange(t, dtype=np.float64))
        return np.asarray(w / w.sum(), dtype=np.float64)
    raise ValueError("kind must be 'uniform' or 'geometric'")


def _resolve_weights(
    weights: Array | Sequence[float] | str | None, horizon: int, gamma: float | None
) -> Array:
    """Restart weights from None/"uniform"/"geometric"/explicit vector; normalized."""
    if weights is None:
        return restart_weights(horizon, "geometric" if gamma is not None else "uniform", gamma)
    if isinstance(weights, str):
        return restart_weights(horizon, weights, gamma)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != horizon:
        raise ValueError(f"weights must have exactly len(p_values)={horizon} entries (s=0..T-1)")
    if not bool(np.all(np.isfinite(w))) or bool(np.any(w < 0.0)):
        raise ValueError("weights must be finite and nonnegative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return np.asarray(w / total, dtype=np.float64)


def restart_mixture_kt_martingale(
    p_values: Array | Sequence[float],
    n_bins: int = 20,
    weights: Array | Sequence[float] | str | None = None,
    gamma: float | None = None,
) -> Array:
    """Restart-mixture KT histogram e-process path M_0..M_T (their Eq. 5, Alg. 3).

    M_t = sum_{s=0}^{t-1} pi_s M_t^(s) + sum_{s=t}^{T-1} pi_s, where the
    component restarted at s keeps its OWN KT bin counts over p_{s+1..t} and
    wealth M_t^(s) = prod_{i=s+1}^t f_i^(s) (M_t^(s) = 1 for t <= s; their
    Eq. 4). The trailing weight tail is the capital of not-yet-started
    components and keeps M_0 = 1. Under every exchangeable null the mixture
    is a nonnegative martingale — hence an e-process (their Prop. 3.3) — and
    for any candidate change point tau the pathwise aggregation cost is at
    most log(1/pi_tau): log M_t >= log M_t^(tau) - log(1/pi_tau). Restarting
    the learner (not only the wealth) removes the pre-change dilution factor
    (t-1-tau)/(t-1+J/2) of their Sec. 3.3. ``weights``: None (uniform, or
    geometric when ``gamma`` is given), "uniform", "geometric", or an
    explicit length-T nonnegative vector (renormalized). O(T^2) time.
    """
    p = _check_p_values(p_values)
    j = _check_n_bins(n_bins)
    n = int(p.size)
    pi = _resolve_weights(weights, n, gamma)
    # log(0) = -inf for zero-weight components is exactly right (they drop out
    # of the mixture); compute it without tripping a divide-by-zero warning.
    log_pi = np.full(n, -np.inf, dtype=np.float64)
    np.log(pi, out=log_pi, where=pi > 0.0)
    bins = _bin_indices(p, j)
    # cc[u, b] = #{1 <= l <= u : bin(p_l) = b}; cc[0] = 0 (labels are 1-indexed).
    onehot = np.zeros((n, j), dtype=np.float64)
    onehot[np.arange(n), bins] = 1.0
    cc = np.empty((n + 1, j), dtype=np.float64)
    cc[0] = 0.0
    np.cumsum(onehot, axis=0, out=cc[1:])
    # tail[t] = sum_{s>=t} pi_s (weight of components not yet started at time t).
    tail = np.concatenate([np.cumsum(pi[::-1])[::-1], [0.0]])
    log_wealth = np.full(n, -np.inf, dtype=np.float64)  # component s = 0..T-1
    path = np.empty(n + 1, dtype=np.float64)
    path[0] = 1.0
    for t in range(1, n + 1):
        jb = int(bins[t - 1])
        log_wealth[t - 1] = 0.0  # start component s = t-1 at wealth 1
        s = np.arange(t, dtype=np.float64)
        counts = cc[t - 1, jb] - cc[:t, jb]  # #{s+1 <= l <= t-1 : bin_l = j_t}
        factors = j * (counts + 0.5) / (t - 1.0 - s + j / 2.0)
        log_wealth[:t] += np.log(factors)
        terms = log_pi[:t] + log_wealth[:t]
        mix = np.logaddexp.reduce(terms)
        log_mix = np.logaddexp(mix, math.log(tail[t]) if tail[t] > 0.0 else -np.inf)
        path[t] = math.exp(min(float(log_mix), _LOG_E_MAX))
    return np.asarray(path, dtype=np.float64)


def conformal_null_path_maxima(
    n_reference: int,
    horizon: int,
    *,
    construction: str = "kt",
    n_bins: int = 20,
    weights: Array | Sequence[float] | str | None = None,
    gamma: float | None = None,
    seed: int = 0,
) -> Array:
    """Reference-bank path maxima from i.i.d. Unif(0,1) streams (their Sec. 3.4).

    The pivotal null law (their Sec. 3.1): under EVERY exchangeable score law
    P in P_0 the sequential conformal p-values are i.i.d. Unif(0,1), and a
    p-value-adaptive betting rule sees the same null trajectory law for all
    P in P_0 — so B independent reference streams of ``horizon`` i.i.d.
    uniforms, passed through the identical construction, are exchangeable
    with the test path maximum under the null. ``construction`` is "kt"
    (Algorithm 2) or "restart_mixture" (Algorithm 3). The bank must be
    generated independently of the test data, with all design choices fixed
    beforehand; ``seed`` pins determinism. Returns the (B,) array of path
    maxima S_T^(b) = max_{0<=t<=T} M_t^(b) for
    :func:`reference_null_threshold`.
    """
    b = int(n_reference)
    if b < 1:
        raise ValueError("n_reference must be >= 1")
    t = int(horizon)
    if t < 1:
        raise ValueError("horizon must be >= 1")
    kind = str(construction)
    if kind not in _CONSTRUCTIONS:
        raise ValueError(f"construction must be one of {_CONSTRUCTIONS}")
    _check_n_bins(n_bins)
    rng = np.random.default_rng(seed)
    maxima = np.empty(b, dtype=np.float64)
    for i in range(b):
        u = rng.random(t)
        path = (
            kt_histogram_martingale(u, n_bins)
            if kind == "kt"
            else restart_mixture_kt_martingale(u, n_bins, weights, gamma)
        )
        maxima[i] = float(np.max(path))
    return np.asarray(maxima, dtype=np.float64)


def calibrate_conformal_martingale(
    alpha: float,
    *,
    horizon: int,
    construction: str = "kt",
    n_bins: int = 20,
    n_reference: int = 499,
    weights: Array | Sequence[float] | str | None = None,
    gamma: float | None = None,
    seed: int = 0,
) -> CalibratedThreshold:
    """Calibrated boundary c_hat_{alpha,B,T} for KT conformal martingales (Sec. 3.4).

    Generates the i.i.d.-uniform reference bank (:func:`conformal_null_path_maxima`),
    applies :func:`reference_null_threshold`, and returns the
    :class:`CalibratedThreshold`. Guarantee: sup over exchangeable nulls of
    P(max_{0<=t<=T} M_t > c_hat) <= alpha, with strict crossing and with the
    probability including the calibration randomness (their Sec. 3.4 via
    Theorem 2.2, constant template). Default B = 499 is the paper's
    simulation bank size K (their Sec. 5.1; k = 475 at alpha = 0.05); their
    watermark application uses B = 4999 (k = 4750). The boundary is valid
    for monitoring windows of length <= ``horizon`` on p-value streams from
    the SAME construction (n_bins, weights); longer monitoring requires a
    fresh calibration. Raises when n_reference < min_reference_nulls(alpha).
    """
    a = _check_alpha(alpha)
    maxima = conformal_null_path_maxima(
        n_reference,
        horizon,
        construction=construction,
        n_bins=n_bins,
        weights=weights,
        gamma=gamma,
        seed=seed,
    )
    out = reference_null_threshold(maxima, a)
    return CalibratedThreshold(
        threshold=out.threshold,
        alpha=out.alpha,
        n_reference=out.n_reference,
        k_order=out.k_order,
        ville_threshold=out.ville_threshold,
        method=f"reference-null:conformal-{construction}",
    )


def crossing_time(
    path: Array | Sequence[float], threshold: float, *, strict: bool = False
) -> int | None:
    """First boundary-crossing index of an e-process path, or None.

    ``strict=False``: first t with M_t >= threshold — the Ville convention of
    the existing lab modules (conformal_martingale.martingale_alarm and
    e_detectors alarm at 1/alpha with >=). ``strict=True``: first t with
    M_t > threshold — the paper's calibrated crossing rule (their Sec. 3.4 /
    5.1: "the calibrated crossing rule is strict"; strictness matters when
    the boundary is an attainable reference order statistic). Returns the
    0-based index into ``path``. Fail-closed on empty/non-finite/negative
    paths and non-positive or non-finite thresholds.
    """
    m = np.asarray(path, dtype=float).ravel()
    if m.size == 0:
        raise ValueError("path must be non-empty")
    if not bool(np.all(np.isfinite(m))) or bool(np.any(m < 0.0)):
        raise ValueError("path must be finite and nonnegative")
    c = float(threshold)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("threshold must be finite and > 0")
    hits = np.flatnonzero(m > c) if strict else np.flatnonzero(m >= c)
    return int(hits[0]) if hits.size else None


def detection_delay_summary(
    paths: Array,
    *,
    threshold: float,
    change_time: int,
    strict: bool = False,
) -> dict[str, object]:
    """Paired detection diagnostics of their Sec. 5.1 for a batch of e-process paths.

    ``paths`` is (R, L) with L = T+1 path values (index t <-> time t, index 0
    the initial wealth). ``change_time`` is tau: scores X_1..X_tau are
    pre-change, so a crossing at path index D <= tau counts as a PRE-CHANGE
    FALSE ALARM and D > tau as a detection. Keys: ``detection_times`` (R,)
    int array, -1 for no crossing by T; ``fa_rate_prechange`` = Pr(D <= tau);
    ``conditional_power`` = Pr(D <= T | D > tau) over paths surviving to the
    change; ``rmdd`` = restricted mean detection delay, the mean of D - tau
    among survivors with non-detection assigned T - tau + 1 (their Sec. 5.1
    censoring convention); ``n_reps``. RMDD comparisons between the
    calibrated and Ville boundaries on the SAME paths isolate the boundary
    effect (their paired C-vs-V design). Fail-closed on invalid shapes,
    non-finite paths, or change_time outside [1, T-1].
    """
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        raise ValueError("paths must be a nonempty 2-D (R, T+1) array with T >= 2")
    if not bool(np.all(np.isfinite(arr))) or bool(np.any(arr < 0.0)):
        raise ValueError("paths must be finite and nonnegative")
    tau = int(change_time)
    horizon = int(arr.shape[1] - 1)
    if not 1 <= tau <= horizon - 1:
        raise ValueError("change_time must satisfy 1 <= change_time <= T-1")
    times = np.full(arr.shape[0], -1, dtype=np.int64)
    for r in range(arr.shape[0]):
        d = crossing_time(arr[r], threshold, strict=strict)
        if d is not None:
            times[r] = d
    pre = times[(times >= 0) & (times <= tau)]
    # Survivors: paths NOT already lost to a pre-change false alarm, i.e.
    # D > tau or no detection by T (censored, delay value T - tau + 1).
    survivors = ~((times >= 0) & (times <= tau))
    detected = times[survivors]
    delays = np.where(
        detected >= 0,
        detected - tau,
        horizon - tau + 1,  # right-censoring value T - tau + 1 (their Sec. 5.1)
    ).astype(np.float64)
    n_surv = int(survivors.sum())
    return {
        "detection_times": times,
        "fa_rate_prechange": float(pre.size) / float(arr.shape[0]),
        "conditional_power": (float(np.count_nonzero(detected >= 0)) / n_surv) if n_surv else 1.0,
        "rmdd": float(delays.mean()) if n_surv else 0.0,
        "n_reps": int(arr.shape[0]),
    }
