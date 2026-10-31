"""Inference for realized conformal coverage under temporal dependence.

Numerical port of Zhai, Cheng & Wu (2026), "Conformal Coverage of Time
Series: Validity and Inference", arXiv:2609.33868 (math.ST), with the
functional-dependence device of Wu (2005a), "Nonlinear system theory: another
look at dependence", PNAS 102(40):14150-14154, and the block-sampling method
of Zhang, Ho, Wendler & Wu (2013), "Block sampling under strong dependence",
Stochastic Processes and their Applications 123(6):2323-2339. The companion
paper Zhai, Cheng & Wu (2026), "Valid and Efficient Split Conformal
Regression for Time Series", arXiv:2609.33866, adds interval-LENGTH accuracy
rates for conformalized quantile/median regression; length-accuracy theory is
not ported here (this layer is coverage-inference only).

Setting (their Sec. 2). Split conformal prediction with a fixed predictor
mu_hat and absolute-error nonconformity scores V_t = |Y_t - mu_hat(X_t)|: the
cutoff q_hat_n is the k_n-th order statistic of the calibration scores
V_1..V_n with k_n = ceil((n+1)(1-alpha)) (their Eq. (1), q_hat_n = inf when
k_n > n), and realized coverage over the ADJACENT test window n+1..n+m is
Cov_{n,m} = m^-1 sum 1{V_t <= q_hat_n} (their Eq. (2)).

Ported results:

1. Functional dependence measure (FDM). delta_r(k) = ||V_k - V_k*||_r is the
   L^r PREDICTION ERROR at horizon k when the single innovation eps_0 is
   replaced by an independent copy (Wu 2005a; their Eq. (3)); the indicator
   FDM is theta_k = sup_v ||1{V_k <= v} - 1{V_k* <= v}||_2 with
   Theta = sum_k theta_k (their Eq. (4)). Estimator choice: coupled
   Monte-Carlo, the standard numerical estimator of Wu's FDM — the caller
   supplies paired draws (V_k, V_k*) sharing all innovations except eps_0
   (:func:`fdm_coupling_norms`, :func:`indicator_fdm`, whose sup over v is
   evaluated exactly on the MC empirical measure). Assumptions: iid
   innovations, a causal stationary representation V_t = H(..., eps_t), and
   simulation access to the coupled pair; Lemma 8 (their Eq. (18)) transfers
   score-level FDM to indicator FDM (:func:`fdm_transfer_bound`).
2. Non-asymptotic marginal coverage bounds (their Theorem 1, Eqs. (5)-(6)):
   |P(cover) - (1-alpha)| <= 2/n + 3(Theta^2/(4n))^(1/3) density-free, and an
   O(n^-1/2) bound under a local conditional-density condition
   (:func:`marginal_coverage_bound`). No mixing assumption is needed — mixing
   can fail even for elementary short-memory recursions (Andrews 1984).
3. CLT for realized coverage (their Theorem 3, Eq. (9)): under Assumptions
   1-2 (summable indicator FDM + quantile regularity), via the Bahadur
   representation of the cutoff (their Theorem 2; Wu 2005b, Ann. Statist.
   33(4):1934-1963),
       sqrt(nm/(n+m)) {Cov_{n,m} - (1-alpha)} -> N(0, sigma_cov^2),
   sigma_cov^2 = sum_h Cov{1(V_0<=q), 1(V_h<=q)}. The feasible SE is
   sigma_hat_cov * sqrt(1/n + 1/m) with sigma_hat_cov^2 from nonoverlapping
   blocks of calibration indicators at the estimated cutoff (their Eq. (10),
   consistent under their Proposition 4; batch-means estimation as in Flegal
   & Jones 2010, Ann. Statist. 38(2):1034-1070). Moving-block and
   Politis-Romano (1994, JASA 89(428):1303-1313) stationary-bootstrap
   variants are provided (reusing quant_fund.metrics.inference); only the
   nonoverlapping member carries their Proposition 4 proof. Block length
   defaults to floor(n^(2/3)) — their Study 2 choice, satisfying l -> inf and
   l/n -> 0 as their Proposition 4 requires.
4. coverage_ztest: the asymptotic two-sided test of H0: realized coverage
   center = nominal, rejecting at |z| > z_{1-eta/2} (their Sec. 3), with a
   Wald CI for the marginal coverage probability and their pre-test
   prediction interval [(1-alpha) +- z SE] cap [0, 1] for future realized
   coverage. For a small fixed number of blocks B >= 2 the corrected
   reference sqrt(B/(B-1)) * t_{B-1} replaces the normal quantiles (their
   Eq. (24); Jones, Haran, Caffo & Neath 2006, JASA 101(476):1537-1547).
5. Long-memory block sampling (their Sec. 4 and App. C.1, Theorems 6-7). For
   Gaussian linear residuals with coefficients a_j ~ c j^-beta, 1/2 < beta <
   1, realized coverage obeys a Gaussian CLT for beta > 3/4, a
   sqrt(n/log n)-normalized Gaussian CLT at beta = 3/4 (their Eq. (11) rate
   r_n), and a NON-GAUSSIAN Rosenblatt limit for 1/2 < beta < 3/4 (Taqqu
   1975; Dehling & Taqqu 1989; their Theorem 6, via Hermite-rank arguments).
   :func:`block_sampling_inference` implements their Appendix C.1 procedure:
   within the calibration set, every overlapping pair (calibration block of
   length b = floor(n^zeta), adjacent evaluation block of length
   floor(b*m/n)) is re-calibrated and re-evaluated; memory is estimated from
   the SIGNED calibration residuals by the two-scale block-variance rule
   beta_hat = 3/2 - log(Q_hat_{2h}/Q_hat_h)/(2 log 2), h = floor(sqrt(n))
   (their Eq. (37); Zhang et al. 2013, with their stated beta_hat = 3/4
   fallback on zero block variance); the normalization
   R_N(u) = {(1 + int_1^N x^(2-4u) dx)/N}^(1/2) (their Eq. (38)) is evaluated
   in closed form with expm1 near u = 3/4 exactly as their Lemma 10
   prescribes; and the normalized block CDF G_hat_{n,b} (their Eq. (39))
   supplies quantile-based rejection, a two-sided p-value, and a prediction
   interval for realized coverage that is valid in all three regimes without
   a Gaussian approximation.

Fail-closed edges: empty/NaN streams, invalid alpha/eta/zeta, calibration
streams too short for a finite cutoff, fewer than two complete blocks,
degenerate (constant) indicator streams, and block-sampling windows that
cannot fit the paired blocks all raise ValueError rather than returning
silent fallbacks.

Honesty: everything here is an inferential object (proper coverage-error
diagnostics) — no Sharpe/Sortino/P&L content, no live-trading capability.
All Monte-Carlo evidence in the accompanying tests uses seeded SYNTHETIC
streams with documented tolerances and is a correctness check of the ported
theory, never market evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.stats import t as student_t

from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices

Array = NDArray[np.float64]

__all__ = [
    "BlockSamplingInference",
    "BlockVariance",
    "CoverageZTest",
    "MarginalCoverageBound",
    "block_sampling_inference",
    "coverage_normalization",
    "coverage_ztest",
    "default_block_length",
    "fdm_coupling_norms",
    "fdm_transfer_bound",
    "indicator_fdm",
    "marginal_coverage_bound",
    "memory_exponent",
    "realized_coverage",
    "sigma_cov_squared",
    "split_conformal_cutoff",
]

# Minimum calibration length for any block-based variance estimate: with the
# default block floor(n^(2/3)) capped at n//2, n >= 4 guarantees B >= 2.
_MIN_N_BLOCKS = 4
# Minimum number of overlapping block statistics for quantile inference in
# block_sampling_inference (below this the eta/2 empirical quantile is pure
# noise; fail closed instead of reporting an unresolvable p-value).
_MIN_BLOCK_STATS = 10
# Guard on the O(n_starts * b) re-calibration loop of block_sampling_inference.
_MAX_BLOCK_STARTS = 200_000
# Tolerance on ceil((n+1)(1-alpha)) absorbing float representation error only;
# a mathematical integer value of (n+1)(1-alpha) is never within this distance.
_RANK_EPS = 1e-9


def _check_level(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or not 0.0 < v < 1.0:
        raise ValueError(f"{name} must be finite and in the open interval (0, 1)")
    return v


def _check_finite_1d(x: Iterable[float] | Array, name: str) -> Array:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    return arr


def _conformal_rank(n: int, alpha_coverage: float) -> int:
    """k_n = ceil((n+1)(1-alpha)) (their Eq. (1)), clamped to >= 1."""
    return int(max(1, np.ceil((n + 1) * (1.0 - alpha_coverage) - _RANK_EPS)))


def split_conformal_cutoff(scores: Iterable[float] | Array, alpha_coverage: float) -> float:
    """Split-conformal cutoff q_hat_n, the k_n-th order statistic (Eq. (1)).

    k_n = ceil((n+1)(1-alpha)); q_hat_n = +inf when k_n > n exactly as their
    Eq. (1) prescribes (the prediction set is then all of R and realized
    coverage is trivially 1 — :func:`coverage_ztest` refuses to test such a
    degenerate calibration). Raises ValueError on empty/non-finite scores or
    an invalid level.
    """
    a = _check_level(alpha_coverage, "alpha_coverage")
    v = _check_finite_1d(scores, "scores")
    n = int(v.size)
    k = _conformal_rank(n, a)
    if k > n:
        return float("inf")
    return float(np.partition(v, k - 1)[k - 1])


def realized_coverage(test_scores: Iterable[float] | Array, cutoff: float) -> float:
    """Realized coverage Cov_{n,m} = m^-1 sum 1{V_t <= q_hat} (their Eq. (2)).

    ``cutoff`` must be a finite number or +inf (the Eq. (1) infinite-cutoff
    convention, which returns 1.0). NaN or -inf cutoffs raise ValueError.
    """
    v = _check_finite_1d(test_scores, "test_scores")
    c = float(cutoff)
    if np.isnan(c) or c == -np.inf:
        raise ValueError("cutoff must be finite or +inf")
    if c == np.inf:
        return 1.0
    return float(np.mean(v <= c))


# ---------------------------------------------------------------------------
# 1. Functional dependence measure (Wu 2005a; their Sec. 2, Eqs. (3)-(4))
# ---------------------------------------------------------------------------


def _check_coupled_pair(
    v_orig: Iterable[float] | Array, v_coupled: Iterable[float] | Array
) -> tuple[Array, Array]:
    a = np.asarray(v_orig, dtype=float)
    b = np.asarray(v_coupled, dtype=float)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("coupled samples must be a nonempty 2-D (n_mc, K) array")
    if b.shape != a.shape:
        raise ValueError("v_orig and v_coupled must have identical (n_mc, K) shape")
    if not bool(np.all(np.isfinite(a))) or not bool(np.all(np.isfinite(b))):
        raise ValueError("coupled samples must be finite (NaN/inf rejected)")
    return a, b


def fdm_coupling_norms(
    v_orig: Iterable[float] | Array,
    v_coupled: Iterable[float] | Array,
    *,
    r: float = 2.0,
) -> Array:
    """Coupled Monte-Carlo estimate of Wu's FDM delta_r(k) (their Eq. (3)).

    delta_r(k) = ||V_k - V_k*||_r is the L^r prediction error at horizon k
    when the innovation eps_0 is replaced by an independent copy eps_0'
    (Wu 2005a): rows of ``v_orig`` / ``v_coupled`` are paired draws sharing
    every innovation except eps_0, columns are horizons k = 0..K-1. The
    estimator is delta_hat_r(k) = (n_mc^-1 sum_i |V_k^(i) - V_k*^(i)|^r)^(1/r)
    — consistent for delta_r(k) by the law of large numbers, and unbiased for
    delta_r(k)^r. Assumes iid innovations and a causal stationary
    representation V_t = H(..., eps_t) with simulation access to the coupled
    pair; a single observed stream does not identify the coupling without a
    model.
    """
    rr = float(r)
    if not np.isfinite(rr) or rr < 1.0:
        raise ValueError("r must be finite and >= 1")
    a, b = _check_coupled_pair(v_orig, v_coupled)
    d = np.abs(a - b)
    return np.asarray(np.mean(d**rr, axis=0) ** (1.0 / rr), dtype=np.float64)


def indicator_fdm(v_orig: Iterable[float] | Array, v_coupled: Iterable[float] | Array) -> Array:
    """Coupled Monte-Carlo estimate of the indicator FDM theta_k (Eq. (4)).

    theta_k = sup_v ||1{V_k <= v} - 1{V_k* <= v}||_2. The sup over v is
    evaluated EXACTLY on the MC empirical measure: the indicator difference
    is nonzero iff v lies in the half-open interval between the paired draws,
    so theta_hat_k^2 = max_v (1/M) #{i: min_i <= v < max_i}, a step function
    maximized at an interval left endpoint (sweep-line over the 2M endpoints).
    Values lie in [0, 1]; Theta_hat = sum_k theta_hat_k is the plug-in
    estimate of the summability quantity in their Assumption 1.
    """
    a, b = _check_coupled_pair(v_orig, v_coupled)
    m = int(a.shape[0])
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    out = np.empty(a.shape[1], dtype=np.float64)
    ones = np.ones(m, dtype=np.float64)
    for k in range(a.shape[1]):
        coords = np.concatenate([lo[:, k], hi[:, k]])
        deltas = np.concatenate([ones, -ones])
        order = np.argsort(coords, kind="stable")
        run = np.cumsum(deltas[order])
        sorted_coords = coords[order]
        last_of_tie = np.empty(run.size, dtype=bool)
        last_of_tie[:-1] = sorted_coords[1:] != sorted_coords[:-1]
        last_of_tie[-1] = True
        out[k] = np.sqrt(max(float(run[last_of_tie].max()), 0.0) / m)
    return np.asarray(out, dtype=np.float64)


def fdm_transfer_bound(
    delta_r_v: Iterable[float] | Array, m_bound: float, *, r: float = 2.0
) -> Array:
    """Lemma 8 transfer bound theta_k <= min{1, sqrt(1+2M) delta_r^V(k)^(r/(2r+2))}.

    Their Eq. (18): M is a global bound on the marginal nonconformity-score
    density and delta_r^V(k) the score-level FDM. Turns score-level coupling
    estimates into indicator-FDM bounds usable in :func:`marginal_coverage_bound`.
    """
    d = np.asarray(delta_r_v, dtype=float)
    if d.size == 0 or not bool(np.all(np.isfinite(d))) or bool(np.any(d < 0.0)):
        raise ValueError("delta_r_v must be nonempty, finite, and nonnegative")
    mm = float(m_bound)
    if not np.isfinite(mm) or mm < 0.0:
        raise ValueError("m_bound must be finite and >= 0")
    rr = float(r)
    if not np.isfinite(rr) or rr < 1.0:
        raise ValueError("r must be finite and >= 1")
    expo = rr / (2.0 * rr + 2.0)
    bound = np.sqrt(1.0 + 2.0 * mm) * np.asarray(d, dtype=np.float64) ** expo
    return np.asarray(np.minimum(1.0, bound), dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. Non-asymptotic marginal coverage bounds (their Theorem 1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarginalCoverageBound:
    """Output of :func:`marginal_coverage_bound`; None = precondition unmet."""

    density_free: float | None
    local_density: float | None
    n: int
    theta: float


def marginal_coverage_bound(
    n: int,
    theta: float,
    *,
    alpha_coverage: float,
    c_f: float | None = None,
    l_lipschitz: float | None = None,
    e: float | None = None,
) -> MarginalCoverageBound:
    """Non-asymptotic marginal coverage-error bounds (their Theorem 1).

    ``n`` is the calibration size and ``theta`` the (estimated) total
    indicator FDM Theta = sum_k theta_k. Part (i), Eq. (5), is density-free:
    |P(cover) - (1-alpha)| <= 2/n + 3(Theta^2/(4n))^(1/3), certified only
    when its precondition 2/n + (2 Theta^2/n)^(1/3) < min{alpha, 1-alpha}
    holds. Part (ii), Eq. (6), needs the local density constants: marginal
    score density f >= ``c_f`` and conditional density <= ``l_lipschitz`` on
    [q-e, q+e], giving 2L/(c_f n) + 2*sqrt(2)*L*Theta/(c_f sqrt(n)) +
    8 Theta^2/(n c_f^2 e^2) under the precondition 2/n <= c_f*e/2. A bound
    whose precondition fails is reported as None (fail-closed: the theorem
    does not certify a number), and partial density-constant triples raise.
    """
    a = _check_level(alpha_coverage, "alpha_coverage")
    n_i = int(n)
    if n_i < 1:
        raise ValueError("n must be >= 1")
    th = float(theta)
    if not np.isfinite(th) or th < 0.0:
        raise ValueError("theta must be finite and >= 0")
    density_free: float | None = None
    if 2.0 / n_i + (2.0 * th * th / n_i) ** (1.0 / 3.0) < min(a, 1.0 - a):
        density_free = float(2.0 / n_i + 3.0 * (th * th / (4.0 * n_i)) ** (1.0 / 3.0))
    consts = (c_f, l_lipschitz, e)
    if any(c is not None for c in consts) and not all(c is not None for c in consts):
        raise ValueError("c_f, l_lipschitz, and e must be supplied together")
    local_density: float | None = None
    if all(c is not None for c in consts):
        cf = float(c_f)  # type: ignore[arg-type]
        ll = float(l_lipschitz)  # type: ignore[arg-type]
        ee = float(e)  # type: ignore[arg-type]
        if not np.isfinite(cf) or cf <= 0.0:
            raise ValueError("c_f must be finite and > 0")
        if not np.isfinite(ll) or ll <= 0.0:
            raise ValueError("l_lipschitz must be finite and > 0")
        if not np.isfinite(ee) or ee <= 0.0:
            raise ValueError("e must be finite and > 0")
        if 2.0 / n_i <= cf * ee / 2.0:
            local_density = float(
                2.0 * ll / (cf * n_i)
                + 2.0 * np.sqrt(2.0) * ll * th / (cf * np.sqrt(n_i))
                + 8.0 * th * th / (n_i * cf * cf * ee * ee)
            )
    return MarginalCoverageBound(
        density_free=density_free, local_density=local_density, n=n_i, theta=th
    )


# ---------------------------------------------------------------------------
# 3. Block-based SE estimation for realized coverage (their Eq. (10), Prop. 4)
# ---------------------------------------------------------------------------


def default_block_length(n: int) -> int:
    """Default block length floor(n^(2/3)), capped at n//2 so that B >= 2.

    Their Study 2 uses l = floor(n^(2/3)), which satisfies l -> inf and
    l/n -> 0 as their Proposition 4 consistency requires.
    """
    n_i = int(n)
    if n_i < _MIN_N_BLOCKS:
        raise ValueError(f"n must be >= {_MIN_N_BLOCKS} for block-based variance estimation")
    return int(max(1, min(int(np.floor(n_i ** (2.0 / 3.0))), n_i // 2)))


@dataclass(frozen=True)
class BlockVariance:
    """Output of :func:`sigma_cov_squared`: the long-run-variance estimate.

    ``n_blocks`` is the number of complete nonoverlapping blocks B (their
    Eq. (10) member, the divisor in the fixed-B t reference), the number of
    overlapping batches (moving member), or n_boot (stationary member, where
    the fixed-B reference does NOT apply).
    """

    sigma2: float
    block_length: int
    n_blocks: int
    method: str

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma2))


_VARIANCE_METHODS = ("nonoverlapping", "moving", "stationary")


def sigma_cov_squared(
    indicators: Iterable[float] | Array,
    *,
    method: str = "nonoverlapping",
    block: int | None = None,
    n_boot: int = 999,
    seed: int = 0,
) -> BlockVariance:
    """Block estimate of sigma_cov^2 = sum_h Cov{1(V_0<=q), 1(V_h<=q)}.

    ``indicators`` are the calibration-set coverage indicators 1{V_t <= q_hat_n}
    (strict 0/1 values required). Members:

    - "nonoverlapping": their Eq. (10) exactly — B = floor(n/l) complete
      nonoverlapping blocks, sigma_hat^2 = (l/B) sum_j (Ibar_j - Ibar)^2;
      consistent under their Proposition 4 (with the tail condition
      sum_k (sum_{j>=k} theta_j)^(1/2) < inf, l -> inf, l/n -> 0).
    - "moving": overlapping batches of the same length,
      sigma_hat^2 = l * (n-l+1)^-1 sum_i (Ibar_i - Ibar)^2 — the overlapping
      batch-means LR-variance estimator (Flegal & Jones 2010); uses more data
      at the same bias order but carries no Proposition 4 proof.
    - "stationary": Politis-Romano (1994) stationary bootstrap via
      quant_fund.metrics.inference.stationary_bootstrap_indices;
      sigma_hat^2 = n * Var_boot(resample mean), with mean block length from
      Politis-White automatic selection (inference.optimal_block_length)
      unless ``block`` is given. Requires ``seed`` for determinism.

    ``block`` defaults to :func:`default_block_length`. Fail-closed: fewer
    than two blocks, block >= n, constant streams are NOT rejected here (a
    zero sigma_hat^2 is legitimate output; :func:`coverage_ztest` refuses to
    studentize by it).
    """
    kind = str(method)
    if kind not in _VARIANCE_METHODS:
        raise ValueError(f"method must be one of {_VARIANCE_METHODS}")
    x = _check_finite_1d(indicators, "indicators")
    n = int(x.size)
    if n < _MIN_N_BLOCKS:
        raise ValueError(f"need at least n={_MIN_N_BLOCKS} indicators for block variance")
    if not bool(np.all((x == 0.0) | (x == 1.0))):
        raise ValueError("indicators must be strict 0/1 values")
    if block is not None:
        ell = int(block)
        if ell < 1:
            raise ValueError("block must be >= 1")
        if ell >= n:
            raise ValueError("block must be < n (need at least two blocks)")
    else:
        ell = default_block_length(n)
    if kind == "nonoverlapping":
        b_count = n // ell
        if b_count < 2:
            raise ValueError("degenerate blocks: floor(n/block) must be >= 2")
        means = x[: b_count * ell].reshape(b_count, ell).mean(axis=1)
        sigma2 = ell * float(np.mean((means - means.mean()) ** 2))
        return BlockVariance(
            sigma2=max(sigma2, 0.0), block_length=ell, n_blocks=b_count, method=kind
        )
    if kind == "moving":
        batches = sliding_window_view(x, ell).mean(axis=1)
        if batches.size < 2:
            raise ValueError("degenerate blocks: need at least two overlapping batches")
        sigma2 = ell * float(np.mean((batches - batches.mean()) ** 2))
        return BlockVariance(
            sigma2=max(sigma2, 0.0), block_length=ell, n_blocks=int(batches.size), method=kind
        )
    n_boot_i = int(n_boot)
    if n_boot_i < 2:
        raise ValueError("n_boot must be >= 2 for the stationary bootstrap")
    if block is None:
        auto = float(optimal_block_length(x))
        mean_block = auto if np.isfinite(auto) else float(max(1, int(np.ceil(n ** (1.0 / 3.0)))))
        mean_block = float(min(max(mean_block, 1.0), float(n)))
        ell = int(max(1, min(int(np.ceil(mean_block)), n - 1)))
    else:
        mean_block = float(ell)
    rng = np.random.default_rng(int(seed))
    idx = stationary_bootstrap_indices(n, n_boot_i, mean_block, rng)
    means = x[idx].mean(axis=1)
    sigma2 = n * float(np.var(means, ddof=1))
    return BlockVariance(sigma2=max(sigma2, 0.0), block_length=ell, n_blocks=n_boot_i, method=kind)


# ---------------------------------------------------------------------------
# 4. Asymptotic coverage z-test (their Sec. 3, Theorem 3 + Proposition 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageZTest:
    """Output of :func:`coverage_ztest` (all intervals already clipped to [0, 1])."""

    realized_coverage: float
    nominal: float
    n_calibration: int
    m_test: int
    sigma_cov: float
    se: float
    block_length: int
    n_blocks: int
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    pi_low: float
    pi_high: float
    reference: str
    method: str


_REFERENCES = ("normal", "student_t")


def coverage_ztest(
    calibration_scores: Iterable[float] | Array,
    test_scores: Iterable[float] | Array,
    *,
    alpha_coverage: float = 0.1,
    eta: float = 0.05,
    method: str = "nonoverlapping",
    block: int | None = None,
    reference: str = "normal",
    n_boot: int = 999,
    seed: int = 0,
) -> CoverageZTest:
    """Asymptotic test of H0: realized-coverage center = nominal (their Sec. 3).

    Statistic {Cov_{n,m} - (1-alpha)}/SE_hat with the feasible SE
    sigma_hat_cov * sqrt(1/n + 1/m) of their Theorem 3 + Proposition 4; the
    two-sided diagnostic rejects at level ``eta`` when |z| exceeds the
    reference critical value. ``reference="normal"`` uses z_{1-eta/2};
    ``reference="student_t"`` uses the fixed-B corrected reference
    sqrt(B/(B-1)) * t_{B-1,1-eta/2} of their Eq. (24) (Jones et al. 2006),
    valid only for method="nonoverlapping" (B is their number of complete
    blocks) — other methods raise.

    ``ci_low``/``ci_high`` form the Wald interval Cov_{n,m} +- c*SE_hat for
    the marginal coverage probability P(V_t <= q_hat_n): by test/CI duality
    it contains the nominal 1-alpha exactly when the two-sided test does not
    reject. ``pi_low``/``pi_high`` are their PRE-test prediction interval
    [(1-alpha) +- c*SE_hat] cap [0,1] for the random realized coverage of a
    future adjacent window. Validity is asymptotic under their Assumptions
    1-2 (summable indicator FDM + quantile regularity); it is a coverage
    diagnostic, never market evidence or a performance claim.

    Fail-closed: an infinite cutoff (k_n > n), a zero block variance
    (constant calibration indicators — the test is undefined, not "passing"),
    or an invalid reference/method combination raise ValueError.
    """
    a = _check_level(alpha_coverage, "alpha_coverage")
    eta_l = _check_level(eta, "eta")
    ref = str(reference)
    if ref not in _REFERENCES:
        raise ValueError(f"reference must be one of {_REFERENCES}")
    cal = _check_finite_1d(calibration_scores, "calibration_scores")
    n = int(cal.size)
    q_hat = split_conformal_cutoff(cal, a)
    if not np.isfinite(q_hat):
        raise ValueError(
            f"calibration stream too short: k_n > n={n} at level {a}, cutoff is infinite"
        )
    bv = sigma_cov_squared(
        (cal <= q_hat).astype(float), method=method, block=block, n_boot=n_boot, seed=seed
    )
    if ref == "student_t":
        if bv.method != "nonoverlapping":
            raise ValueError(
                "student_t reference requires method='nonoverlapping' (their Eq. (24))"
            )
        if bv.n_blocks < 2:
            raise ValueError("student_t reference requires B >= 2 complete blocks")
    if bv.sigma2 <= 0.0:
        raise ValueError("degenerate calibration indicators: block variance is zero")
    m_arr = _check_finite_1d(test_scores, "test_scores")
    m = int(m_arr.size)
    cov = realized_coverage(m_arr, q_hat)
    se = bv.sigma * float(np.sqrt(1.0 / n + 1.0 / m))
    z = (cov - (1.0 - a)) / se
    if ref == "normal":
        crit = float(norm.ppf(1.0 - eta_l / 2.0))
        p = float(2.0 * norm.sf(abs(z)))
    else:
        b = bv.n_blocks
        scale = float(np.sqrt(b / (b - 1.0)))
        crit = scale * float(student_t.ppf(1.0 - eta_l / 2.0, b - 1))
        p = float(2.0 * student_t.sf(abs(z) / scale, b - 1))
    nominal = 1.0 - a
    return CoverageZTest(
        realized_coverage=cov,
        nominal=nominal,
        n_calibration=n,
        m_test=m,
        sigma_cov=bv.sigma,
        se=se,
        block_length=bv.block_length,
        n_blocks=bv.n_blocks,
        statistic=z,
        p_value=p,
        ci_low=float(min(max(cov - crit * se, 0.0), 1.0)),
        ci_high=float(min(max(cov + crit * se, 0.0), 1.0)),
        pi_low=float(min(max(nominal - crit * se, 0.0), 1.0)),
        pi_high=float(min(max(nominal + crit * se, 0.0), 1.0)),
        reference=ref,
        method=bv.method,
    )


# ---------------------------------------------------------------------------
# 5. Long memory: memory exponent, normalization, block sampling (App. C.1)
# ---------------------------------------------------------------------------


def memory_exponent(
    residuals: Iterable[float] | Array, *, h: int | None = None
) -> tuple[float, int]:
    """Two-scale block-variance memory estimate beta_hat (their Eq. (37)).

    Applied to the SIGNED calibration residuals W_t (mean zero under their
    Assumption 3), with Q_hat_l = (n-l+1)^-1 sum_{i=0}^{n-l} (sum_{t=i+1}^{i+l}
    W_t)^2 at l = h, 2h and beta_hat = 3/2 - log(Q_hat_{2h}/Q_hat_h)/(2 log 2)
    (Zhang et al. 2013, two-scale method; their Lemma 10 gives
    beta_hat - beta = o_P(1/log n) for Gaussian linear coefficients
    a_j = j^-beta{c + O(j^-phi)} with h = floor(sqrt(n))). Default
    h = floor(sqrt(n)). Their stated convention: a zero block variance at
    either scale returns beta_hat = 3/4 (asymptotically irrelevant). Returns
    (beta_hat, h_used). Fail-closed: h < 1 or 2h > n raises ValueError.
    """
    w = _check_finite_1d(residuals, "residuals")
    n = int(w.size)
    h_used = int(np.floor(np.sqrt(n))) if h is None else int(h)
    if h_used < 1 or 2 * h_used > n:
        raise ValueError(f"h must satisfy 1 <= h and 2h <= n={n}")
    cs = np.concatenate([np.zeros(1, dtype=float), np.cumsum(w)])

    def _q_hat(length: int) -> float:
        sums = cs[length:] - cs[:-length]
        return float(np.mean(sums * sums))

    q_h = _q_hat(h_used)
    q_2h = _q_hat(2 * h_used)
    if q_h <= 0.0 or q_2h <= 0.0:
        return 0.75, h_used
    beta_hat = 1.5 - float(np.log(q_2h / q_h)) / (2.0 * float(np.log(2.0)))
    return beta_hat, h_used


def coverage_normalization(beta: float, n: int) -> float:
    """R_N(u) = {(1 + int_1^N x^(2-4u) dx)/N}^(1/2), their Eq. (38).

    Closed form J_N(u) = 1 + (N^(3-4u) - 1)/(3-4u) for u != 3/4 and
    1 + log N at u = 3/4, evaluated as log(N) * expm1(t)/t with
    t = (3-4u) log N — exactly the cancellation-avoiding prescription of
    their Lemma 10, continuous across the critical exponent. Valid for every
    real u (short-memory u >= 1 degrades gracefully toward the root-n scale:
    R_N(1) ~ sqrt(2/N)); r_hat_N = R_N(beta_hat) is asymptotically
    proportional to their rate r_N in Eq. (11), and the unknown proportionality
    constant cancels from the block-to-full ratio r_hat_n/r_hat_b (their
    Lemma 10, Eq. (44)).
    """
    n_i = int(n)
    if n_i < 2:
        raise ValueError("n must be >= 2 for the normalization integral")
    u = float(beta)
    if not np.isfinite(u):
        raise ValueError("beta must be finite")
    log_n = float(np.log(n_i))
    t = (3.0 - 4.0 * u) * log_n
    inner = log_n if t == 0.0 else log_n * float(np.expm1(t)) / t
    j_n = 1.0 + inner
    return float(np.sqrt(j_n / n_i))


@dataclass(frozen=True)
class BlockSamplingInference:
    """Output of :func:`block_sampling_inference` (their App. C.1, Theorem 7)."""

    realized_coverage: float
    nominal: float
    n_calibration: int
    m_test: int
    beta_hat: float
    regime: str
    h_memory: int
    r_block: float
    r_full: float
    block_length: int
    eval_length: int
    n_block_stats: int
    statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    reject: bool
    zeta: float


def _regime_label(beta_hat: float) -> str:
    """Informational label for their Eq. (11) regimes (the formula is uniform)."""
    if beta_hat >= 1.0:
        return "short_memory"
    if beta_hat > 0.75:
        return "gaussian"
    if beta_hat == 0.75:
        return "gaussian_critical"
    if beta_hat > 0.5:
        return "rosenblatt"
    return "ultra_persistent"


def block_sampling_inference(
    calibration_scores: Iterable[float] | Array,
    test_scores: Iterable[float] | Array,
    residuals: Iterable[float] | Array,
    *,
    alpha_coverage: float = 0.1,
    eta: float = 0.05,
    zeta: float = 2.0 / 3.0,
    h: int | None = None,
) -> BlockSamplingInference:
    """Block-sampling inference for realized coverage (their App. C.1, Thm 7).

    Valid across all three long-memory regimes of their Theorem 6 WITHOUT a
    Gaussian approximation, including the Rosenblatt (non-Gaussian) limit for
    1/2 < beta < 3/4, where moving-block-bootstrap and HAC benchmarks
    dramatically over-reject (their Table 2). Procedure, exactly as App. C.1:

    - b = floor(n^zeta) (their b = floor(n^zeta), 0 < zeta < 1; default 2/3
      matches their experiments), evaluation length l_b = floor(b*m/n)
      (matching the target calibration-test length ratio), block rank
      k_b = ceil((b+1)(1-alpha)).
    - For every start i = 0..n-b-l_b: re-calibrate on V_{i+1..i+b} and
      evaluate Cov_{i,b} on the adjacent V_{i+b+1..i+b+l_b}.
    - beta_hat from the SIGNED calibration ``residuals`` via
      :func:`memory_exponent`; normalizations r_hat_b = R_b(beta_hat),
      r_hat_n = R_n(beta_hat) via :func:`coverage_normalization`.
    - Normalized block CDF G_hat_{n,b} of (Cov_{i,b} - (1-alpha))/r_hat_b
      (their Eq. (39)); the target statistic is
      (Cov_{n,m} - (1-alpha))/r_hat_n. Reject H0 at level eta when the
      statistic falls outside [c_hat_{eta/2}, c_hat_{1-eta/2}] (linear-
      interpolated empirical quantiles); the CI rescales those endpoints by
      r_hat_n and shifts by 1-alpha (their prediction interval for realized
      coverage), clipped to [0, 1]. The p-value is the two-sided empirical
      tail 2*min(G_hat(T), 1 - G_hat(T-)), resolved on the 1/n_block_stats
      grid.

    ``residuals`` must align with ``calibration_scores`` (W_t with
    V_t = |W_t|); it is required, not inferred — sign information is not
    recoverable from the scores. Deterministic given inputs (no resampling
    randomness). Fail-closed: k_b > b, l_b < 1, b + l_b > n, 2h > n, fewer
    than 10 block statistics, more than 200k starts (loop-cost guard), or an
    infinite full-sample cutoff all raise ValueError.
    """
    a = _check_level(alpha_coverage, "alpha_coverage")
    eta_l = _check_level(eta, "eta")
    z = float(zeta)
    if not np.isfinite(z) or not 0.0 < z < 1.0:
        raise ValueError("zeta must be finite and in the open interval (0, 1)")
    v = _check_finite_1d(calibration_scores, "calibration_scores")
    w = _check_finite_1d(residuals, "residuals")
    n = int(v.size)
    if int(w.size) != n:
        raise ValueError("residuals must align with calibration_scores")
    test = _check_finite_1d(test_scores, "test_scores")
    m = int(test.size)
    b = int(np.floor(n**z))
    ell_b = int(np.floor(b * m / n)) if n > 0 else 0
    k_b = _conformal_rank(b, a)
    if b < 2:
        raise ValueError(f"n={n} too short: block length b={b} must be >= 2")
    if k_b > b:
        raise ValueError(f"k_b={k_b} > b={b}: nominal level too extreme for the block length")
    if ell_b < 1:
        raise ValueError(f"evaluation length l_b={ell_b} < 1: test window too short for n={n}")
    if b + ell_b > n:
        raise ValueError(f"paired blocks do not fit: b + l_b = {b + ell_b} > n = {n}")
    n_starts = n - b - ell_b + 1
    if n_starts < _MIN_BLOCK_STATS:
        raise ValueError(
            f"only {n_starts} block statistics < {_MIN_BLOCK_STATS}: stream too short "
            "for block-sampling quantile inference"
        )
    if n_starts > _MAX_BLOCK_STARTS:
        raise ValueError(f"{n_starts} block starts exceeds the {_MAX_BLOCK_STARTS} cost guard")
    q_hat = split_conformal_cutoff(v, a)
    if not np.isfinite(q_hat):
        raise ValueError(f"calibration stream too short: k_n > n={n}, cutoff is infinite")
    beta_hat, h_used = memory_exponent(w, h=h)
    if max(b + ell_b, 2 * h_used) > n:
        raise ValueError(f"max(b + l_b, 2h) = {max(b + ell_b, 2 * h_used)} > n = {n}")
    stats = np.empty(n_starts, dtype=np.float64)
    for i in range(n_starts):
        window = v[i : i + b]
        q_i = float(np.partition(window, k_b - 1)[k_b - 1])
        stats[i] = float(np.mean(v[i + b : i + b + ell_b] <= q_i))
    r_b = coverage_normalization(beta_hat, b)
    r_n = coverage_normalization(beta_hat, n)
    nominal = 1.0 - a
    block_stats = (stats - nominal) / r_b
    cov = realized_coverage(test, q_hat)
    stat = (cov - nominal) / r_n
    c_lo = float(np.quantile(block_stats, eta_l / 2.0))
    c_hi = float(np.quantile(block_stats, 1.0 - eta_l / 2.0))
    g_le = float(np.mean(block_stats <= stat))
    g_lt = float(np.mean(block_stats < stat))
    p_value = float(min(1.0, 2.0 * min(g_le, 1.0 - g_lt)))
    return BlockSamplingInference(
        realized_coverage=cov,
        nominal=nominal,
        n_calibration=n,
        m_test=m,
        beta_hat=beta_hat,
        regime=_regime_label(beta_hat),
        h_memory=h_used,
        r_block=r_b,
        r_full=r_n,
        block_length=b,
        eval_length=ell_b,
        n_block_stats=n_starts,
        statistic=stat,
        p_value=p_value,
        ci_low=float(min(max(nominal + r_n * c_lo, 0.0), 1.0)),
        ci_high=float(min(max(nominal + r_n * c_hi, 0.0), 1.0)),
        reject=bool(stat < c_lo or stat > c_hi),
        zeta=z,
    )
