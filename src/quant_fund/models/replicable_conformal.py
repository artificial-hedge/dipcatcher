"""Replicable conformal calibration (ReCal) — Papamichalis, Ruane & Papamichalis 2026.

Implements the ReCal construction of "Replicable Conformal Prediction"
(arXiv:2608.23638v1 [stat.ML], submitted 23 Aug 2026; Marios Papamichalis,
Regina Ruane, Theofanis Papamichalis). Two analysts who calibrate the same
frozen score artifact on independent samples deploy different classifiers with
probability one under standard split conformal (Proposition 1: the threshold
is an order statistic of continuous scores). ReCal resolves this with shared
randomness plus coarse rounding — Algorithm 1:

    shared: score artifact s, level alpha, grid width beta, one seed giving
            the offset u ~ Unif[0, beta)
    (1) tau_tilde = S_(k),  k = ceil((1 - alpha) * (n + 1))
    (2) tau = u + beta * ceil((tau_tilde - u) / beta)   (round UP to the grid)
    (3) deploy C_tau = {y : s(., y) <= tau}

Guarantees implemented and regression-locked here (all from the fetched paper):

* Identity: with the plug-in width beta = 2 * kappa_hat * B_n / (f_hat * rho),
  B_n = sqrt(2 alpha (1 - alpha) / n) + sqrt(2) / n (Propositions 2 and 3), two
  analysts sharing (beta, u) deploy the identical classifier with probability
  >= 1 - rho, given O(kappa^2 alpha (1 - alpha) / (eps^2 rho^2)) calibration
  points (Corollary 1); Theorem 2(i) bounds the mismatch by
  B_n / (f_min beta) + 4 exp(-n f_min^2 Delta^2 / 2) under the local margin
  condition (Assumption 1). Proposition 2's guarantee averages over the shared
  offset u; the Monte-Carlo harnesses below redraw u once per replicate and
  share it across the analysts of that replicate, exactly as the protocol does.
* Validity: rounding UP only enlarges sets, so marginal coverage >= k/(n+1)
  >= 1 - alpha survives unconditionally for every fixed u (Theorem 2(ii));
  the training-conditional band is 1 - alpha - e_n(delta) <= F(tau) <=
  1 - alpha + e_n(delta) + f_max beta with e_n(delta) =
  sqrt(log(2/delta)/(2n)) + 2/n.
* Price of replicability: expected coverage inflation (1+o(1)) B_n / rho at
  kappa_hat = 1 (Proposition 3(ii)); matching lower bounds — every threshold
  calibrator needs n >= (9/16384) alpha (1 - alpha) / (eps^2 rho^2) samples
  (Theorem 3) and pays coverage inflation >= (3/70) sqrt(alpha (1 - alpha)/n)
  / rho (Corollary 4). No threshold method pays less.
* Anti-gaming (Corollary 3): with a pre-registered seed, an adversary who
  redraws the calibration set M times and deploys the most favorable draw
  changes a ReCal classifier with probability at most c_M = min(1, (M-1) rho),
  and every selection rule retains expected coverage >= 1 - alpha - c_M. The
  same selection on standard split conformal silently undercovers: choosing
  the smallest of M thresholds (equivalently the narrowest sets, by nesting)
  has exact expected population coverage int_0^1 {1 - B_{k,n+1-k}(t)}^M dt
  = mu_n - a_M sigma_n + o(sigma_n), with mu_n = k/(n+1),
  sigma_n^2 = mu_n (1 - mu_n) / (n + 2), a_M = E max_{j<=M} Z_j; the selected
  threshold satisfies H_n(T_min) ~ Beta(1, M), total-variation distance
  (M-1)/M * M^{-1/(M-1)} from the honest law, so a support-only audit has zero
  power (Appendix B, eqs. 19-22).
* Seedless (Theorem 4): with the deterministic grid u = 0 and
  beta = eps / (2 f_max), every analyst lands, with probability >= 1 - delta,
  in a common sample-independent two-element list {g, g + beta} of adjacent
  classifiers, at sample rate O(kappa^2 log(1/delta) / eps^2) — no rho^-2
  factor; list size two is optimal for delta < 1/2.

Exact replication is impossible (Theorem 1): a (1, delta)-list-replicable,
uniformly eps-valid procedure cannot exist when eps + delta < min(alpha,
1 - alpha), and with a shared seed an exactly replicable procedure is
data-oblivious. Replicability is therefore always reported at a target rho
with its documented sample cost, never as an absolute.

REPO RELEVANCE — receipts culture. AGENTS.md rule 4: receipts are immutable
evidence and every research claim must be reproducible from a receipt hash. A
replicable calibration is an auditable calibration: the deployed threshold is
a bit-identical artifact across analysts and re-runs, so two sites hashing
their calibration receipt get the same digest, and a regulator re-running the
vendor's calibration lands on the same classifier (the paper's audit
motivation). The selection attack demonstrated here is the same threat model
as ``quant_fund.validation.leakage_redteam``: there, a leaky oracle survives
DSR/PBO gates because the gates are blind to selection multiplicity; here,
min-of-M recalibration silently breaks standard conformal coverage while
passing any support check, and replicability bounds the damage by c_M. That
module is cross-referenced only — this lane does not modify it.

HONESTY STAMPS. Every experiment below runs on SYNTHETIC fixture data (the
paper's rank-transformed / PIT-uniform score testbed: scores iid Unif[0, 1],
so the score law is F(t) = t, density f == 1 and kappa = 1 exactly; the
working quantile q = 1 - alpha sits in the interior). These are correctness
tests of the construction, never market evidence. Metrics are proper research
scores only — coverage, set size, agreement/identity rates — and never
Sharpe/Sortino/Calmar/P&L/NAV (AGENTS.md honesty contract;
``quant_fund.research.catalog`` FORBIDDEN_RESEARCH_METRIC_KEYS).

Fail-closed edges: alpha outside (0, 1), non-positive/non-finite grid width,
offset outside [0, beta), unattainable level (alpha < 1/(n+1), i.e. k > n),
empty/all-non-finite calibration scores, odd or degenerate pilot samples,
rho outside (0, 1], f_hat <= 0, kappa_hat < 1, and M < 2 for the selection
adversary all raise ValueError. Deterministic jitter tie-breaking (Remark 1)
is the score artifact's responsibility and is out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import norm

from quant_fund.metrics.conformal import conformal_quantile

Array = NDArray[np.float64]

GridRule = Literal["prop2", "appendix_g"]

# Paper anchors used by tests and docstrings.
# Remark 2: exact-Beta (Corollary 1) requirement at (alpha, eps, rho, delta)
# = (0.1, 0.02, 0.1, 0.05), kappa = 1 is 7.2e5 calibration points.
# Corollary 3: c_M = min(1, (M - 1) rho).
# Theorem 3: n >= (9/16384) alpha (1 - alpha) / (eps^2 rho^2) at delta = 0.


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def _validate_grid(beta: float, offset: float) -> None:
    b = float(beta)
    u = float(offset)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("beta must be positive and finite")
    if not np.isfinite(u) or u < 0.0 or u >= b:
        raise ValueError("offset must lie in [0, beta)")


def _finite_scores(scores: Array) -> Array:
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0 or not np.all(np.isfinite(s)):
        raise ValueError("calibration scores must be non-empty and finite")
    return s


def _check_level_attainable(alpha: float, n: int) -> int:
    """k = ceil((1 - alpha)(n + 1)) must satisfy k <= n (paper's alpha >= 1/(n+1))."""
    if float(alpha) < 1.0 / (float(n) + 1.0):
        raise ValueError(
            f"level unattainable: need alpha >= 1/(n+1) = {1.0 / (n + 1.0):.6g} "
            f"for n = {n} calibration scores"
        )
    return int(np.ceil((1.0 - float(alpha)) * (float(n) + 1.0)))


def _kth_order_stats(matrix: Array, alpha: float) -> Array:
    """Row-wise k-th smallest score, k = ceil((1 - alpha)(n + 1)), for (m, n) input."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-d (draws x calibration scores)")
    n = int(mat.shape[1])
    k = _check_level_attainable(alpha, n)
    k = min(max(k, 1), n)
    return np.partition(mat, k - 1, axis=1)[:, k - 1].astype(float)


# ---------------------------------------------------------------------------
# Core construction (Algorithm 1)
# ---------------------------------------------------------------------------


def b_n_spread(alpha: float, n: int) -> float:
    """B_n = sqrt(2 alpha (1 - alpha) / n) + sqrt(2) / n (Theorem 2).

    The exact finite-sample spread constant governing both the disagreement
    probability and the coverage inflation of the grid-rounded threshold.
    """
    _validate_alpha(alpha)
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    a = float(alpha)
    nn = float(n)
    return float(np.sqrt(2.0 * a * (1.0 - a) / nn) + np.sqrt(2.0) / nn)


def round_up_to_grid(values: Array | float, beta: float, offset: float = 0.0) -> Array:
    """tau = u + beta * ceil((v - u) / beta): round UP onto the shared grid u + beta*Z.

    Pure arithmetic (Algorithm 1 step 2); NaNs propagate. Both analysts run
    the identical expression, so equal grid cells give bit-identical floats.
    Rounding up can only enlarge prediction sets, which is why the marginal
    coverage guarantee survives unconditionally (Theorem 2(ii)). In floating
    point a value sitting exactly on a grid point may round one cell up
    (representation noise in the ceil argument); under continuous scores this
    is a measure-zero event and still only enlarges sets.
    """
    _validate_grid(beta, offset)
    v = np.asarray(values, dtype=float)
    out = float(offset) + float(beta) * np.ceil((v - float(offset)) / float(beta))
    return np.asarray(out, dtype=np.float64)


def replicable_conformal_threshold(
    scores: Array, alpha: float, beta: float, offset: float = 0.0
) -> float:
    """ReCal threshold (Algorithm 1 steps 1-2) from raw calibration scores.

    Step 1 is the standard finite-sample split-conformal quantile
    ``conformal_quantile`` (k-th smallest score, k = ceil((1 - alpha)(n + 1)));
    step 2 rounds it UP to the shared grid ``offset + beta * Z``. Fail-closed:
    alpha must lie in (0, 1), the level must be attainable (alpha >=
    1/(n + 1), else k > n and no finite threshold covers at 1 - alpha),
    scores must all be finite, beta > 0 finite, and
    offset in [0, beta).
    """
    _validate_alpha(alpha)
    s = _finite_scores(np.asarray(scores, dtype=float))
    _check_level_attainable(alpha, int(s.size))
    _validate_grid(beta, offset)
    tau_tilde = conformal_quantile(s, float(alpha))
    return float(round_up_to_grid(tau_tilde, beta, offset))


def shared_offset(seed: int, beta: float) -> float:
    """u = beta * V, V ~ Unif[0, 1) from the single shared seed (Algorithm 1).

    Two analysts passing the same seed obtain bit-identical offsets; this is
    the pre-registered shared randomness the replicability guarantee averages
    over (Proposition 2, eq. 40).
    """
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be positive and finite")
    return float(float(beta) * np.random.default_rng(int(seed)).random())


def grid_width(
    alpha: float,
    n: int,
    rho: float,
    f_hat: float,
    kappa_hat: float = 1.5,
    rule: GridRule = "prop2",
) -> float:
    """Shared grid width beta for target non-identity probability rho.

    ``prop2`` (Propositions 2 and 3): beta = 2 * kappa_hat * B_n / (f_hat * rho),
    the theory-backed plug-in width; ``f_hat`` estimates the score density at
    the working quantile (see :func:`pilot_density`), ``kappa_hat >= kappa`` is
    the pre-registered safety factor (the paper's recommended deployment
    default is 1.5; kappa_hat = 1 is asymptotically exact, Proposition 3).
    ``appendix_g`` (the paper's experimental grid rule): beta = kappa_hat *
    sqrt(2 alpha (1 - alpha) / n) / (f_hat * rho) — the leading term, with
    kappa_hat absorbing both the factor 2 and the pilot margin.
    """
    _validate_alpha(alpha)
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    fh = float(f_hat)
    if not np.isfinite(fh) or fh <= 0.0:
        raise ValueError("f_hat must be positive and finite")
    kh = float(kappa_hat)
    if not np.isfinite(kh) or kh < 1.0:
        raise ValueError("kappa_hat must be >= 1 (it upper-bounds kappa = f_max / f_min)")
    a = float(alpha)
    nn = float(n)
    if rule == "prop2":
        return float(2.0 * kh * b_n_spread(a, int(n)) / (fh * float(rho)))
    if rule == "appendix_g":
        return float(kh * np.sqrt(2.0 * a * (1.0 - a) / nn) / (fh * float(rho)))
    raise ValueError(f"unknown grid rule: {rule!r}")


def pilot_density(
    pilot_scores: Array, alpha: float, window: float, accuracy: float = 0.25
) -> float:
    """Safeguarded pilot density estimate f_hat_L at the working quantile (Proposition 2).

    The public pilot of n_p = 2m scores (disjoint from all calibration data)
    is split in halves: q_hat_p is the k_p-th order statistic of the first
    half with k_p = ceil((1 - alpha)(m + 1)); N_W counts second-half scores in
    W = [q_hat_p - h, q_hat_p + h]; and f_hat_L = max(N_W, 1) / ((1 + c) 2 h m)
    for accuracy c in [0, 1). On the good pilot event (probability >= 1 -
    delta_p under the paper's side conditions) f_hat_L lower-bounds f_min up to
    (1 - c)/(1 + c) and never exceeds f_max, so beta = 2 kappa_hat B_n /
    (f_hat_L rho) inherits the rho-replicability guarantee. c = 0 gives the
    plain point estimate used in the paper's experimental protocol.
    """
    _validate_alpha(alpha)
    s = _finite_scores(np.asarray(pilot_scores, dtype=float))
    if s.size < 2 or s.size % 2 != 0:
        raise ValueError("pilot must hold an even count >= 2 of finite scores (n_p = 2m)")
    h = float(window)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("window must be positive and finite")
    c = float(accuracy)
    if not 0.0 <= c < 1.0:
        raise ValueError("accuracy must be in [0, 1)")
    m = int(s.size) // 2
    k_p = int(np.ceil((1.0 - float(alpha)) * (m + 1)))
    if k_p > m:
        raise ValueError(
            f"pilot half too small for alpha: need k_p = {k_p} <= m = {m} "
            "(level unattainable from the pilot)"
        )
    first, second = s[:m], s[m:]
    q_p = float(np.partition(first, k_p - 1)[k_p - 1])
    n_w = int(np.count_nonzero(np.abs(second - q_p) <= h))
    return float(max(n_w, 1) / ((1.0 + c) * 2.0 * h * m))


def epsilon_protocol_grid_width(eps: float, f_max: float) -> float:
    """beta = eps / (2 f_max): the (eps, rho) protocol width (Corollary 1, Theorem 4).

    Also the seedless deterministic-grid width; satisfies beta <= Delta / 2
    whenever eps <= f_max * Delta.
    """
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not np.isfinite(float(f_max)) or float(f_max) <= 0.0:
        raise ValueError("f_max must be positive and finite")
    return float(float(eps) / (2.0 * float(f_max)))


# ---------------------------------------------------------------------------
# Quantified guarantees (Theorems 2-4, Corollaries 1-4, Proposition 3)
# ---------------------------------------------------------------------------


def disagreement_bound(alpha: float, n: int, beta: float, f_min: float, margin: float) -> float:
    """Theorem 2(i): P[tau != tau'] <= B_n / (f_min beta) + 4 exp(-n f_min^2 Delta^2 / 2).

    ``margin`` is Delta > 0 of Assumption 1 (the score density is bounded in
    [f_min, f_max] on the window [q - Delta, q + Delta] around the working
    quantile q). Clipped to 1 (a probability).
    """
    _validate_alpha(alpha)
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be positive and finite")
    fm = float(f_min)
    if not np.isfinite(fm) or fm <= 0.0:
        raise ValueError("f_min must be positive and finite")
    d = float(margin)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("margin must be positive and finite")
    tail = 4.0 * float(np.exp(-float(n) * fm * fm * d * d / 2.0))
    return float(min(1.0, b_n_spread(alpha, int(n)) / (fm * float(beta)) + tail))


@dataclass(frozen=True)
class ConditionalBand:
    """Theorem 2(ii) training-conditional coverage band for F(tau)."""

    e_n: float
    lower: float
    upper: float
    feasible: bool


def conditional_coverage_band(
    alpha: float,
    n: int,
    delta: float,
    beta: float,
    f_max: float,
    f_min: float,
    margin: float,
) -> ConditionalBand:
    """Two-sided band 1 - alpha - e_n <= F(tau) <= 1 - alpha + e_n + f_max * beta.

    Holds with probability >= 1 - delta over the calibration sample and then
    simultaneously for every shared offset u in [0, beta), provided
    e_n(delta) = sqrt(log(2/delta)/(2n)) + 2/n <= f_min * (Delta - beta)
    (``feasible``); Theorem 2(ii). Marginal coverage >= 1 - alpha needs no
    margin condition at all.
    """
    _validate_alpha(alpha)
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be positive and finite")
    fmx = float(f_max)
    fmn = float(f_min)
    if not np.isfinite(fmx) or fmx <= 0.0 or not np.isfinite(fmn) or fmn <= 0.0:
        raise ValueError("f_max and f_min must be positive and finite")
    if fmx < fmn:
        raise ValueError("f_max must be >= f_min")
    d = float(margin)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError("margin must be positive and finite")
    e_n = float(np.sqrt(np.log(2.0 / float(delta)) / (2.0 * float(n))) + 2.0 / float(n))
    feasible = bool(e_n <= fmn * (d - float(beta)))
    return ConditionalBand(
        e_n=e_n,
        lower=1.0 - float(alpha) - e_n,
        upper=1.0 - float(alpha) + e_n + fmx * float(beta),
        feasible=feasible,
    )


def protocol_sample_requirement(
    alpha: float, eps: float, rho: float, kappa: float = 1.0, delta: float = 0.05
) -> int:
    """Corollary 1 certified calibration size for the (eps, rho, delta) protocol.

    n >= max{32 kappa^2 alpha (1 - alpha) / (eps^2 rho^2), 8 log(2/delta) / eps^2}
    makes ReCal rho-replicable with marginal coverage >= 1 - alpha and
    F(tau) in [1 - alpha - eps/2, 1 - alpha + eps] with probability 1 - delta,
    at beta = eps / (2 f_max) and eps <= f_max * Delta. The paper's explicit
    finite-sample remainder n_0 = o((eps rho)^-2) (Appendix B, eq. 12) is not
    reproduced here; the leading terms are already conservative — Remark 2
    anchors 7.2e5 at (0.1, 0.02, 0.1, 0.05), kappa = 1, about 5x the measured
    frontier.
    """
    _validate_alpha(alpha)
    e = float(eps)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    kh = float(kappa)
    if not np.isfinite(kh) or kh < 1.0:
        raise ValueError("kappa must be >= 1")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    a = float(alpha)
    rep = 32.0 * kh * kh * a * (1.0 - a) / (e * e * float(rho) ** 2)
    band = 8.0 * np.log(2.0 / float(delta)) / (e * e)
    return int(max(np.ceil(rep), np.ceil(band)))


def sample_lower_bound(alpha: float, eps: float, rho: float, delta: float = 0.0) -> int:
    """Theorem 3: no threshold calibrator beats this sample cost.

    Any jointly measurable rho-replicable calibrator whose output satisfies
    P[F_P(A(D; r)) in [1 - alpha +/- eps]] >= 1 - delta for every atomless
    score law needs n >= 9 (1 - 8 delta)^2 / 4096 * alpha (1 - alpha) /
    (eps^2 rho^2) >= (9/16384) alpha (1 - alpha) / (eps^2 rho^2). Requires
    eps <= min(alpha, 1 - alpha)/4 and delta <= 1/16 as in the theorem.
    """
    _validate_alpha(alpha)
    e = float(eps)
    a = float(alpha)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError("eps must be positive and finite")
    if e > min(a, 1.0 - a) / 4.0:
        raise ValueError("Theorem 3 requires eps <= min(alpha, 1 - alpha) / 4")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    d = float(delta)
    if not 0.0 <= d <= 1.0 / 16.0:
        raise ValueError("Theorem 3 requires delta in [0, 1/16]")
    const = 9.0 * (1.0 - 8.0 * d) ** 2 / 4096.0
    return int(np.ceil(const * a * (1.0 - a) / (e * e * float(rho) ** 2)))


def replicability_size_floor(alpha: float, n: int, rho: float) -> float:
    """Corollary 4: unavoidable coverage inflation of any replicable calibrator.

    Every rho-replicable threshold calibrator with the validity and inflation
    side conditions (delta <= 1/32) inflates coverage by at least
    (3/70) sqrt(alpha (1 - alpha) / n) / rho; ReCal attains this frontier up
    to kappa^2 (Theorem 2(ii) with the protocol width).
    """
    _validate_alpha(alpha)
    if int(n) < 1:
        raise ValueError("n must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    a = float(alpha)
    return float((3.0 / 70.0) * np.sqrt(a * (1.0 - a) / float(n)) / float(rho))


def asymptotic_mismatch_limit(rho: float) -> float:
    """Proposition 3(i): plug-in mismatch limit E[min(rho |Z| / 2, 1)] <= rho sqrt(2/pi)/2.

    With the consistent plug-in width at kappa_hat = 1, the achieved
    non-identity probability converges to this value — strictly below the
    target rho; a width inflated by sqrt(pi/2) makes it exactly rho.
    """
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    r = float(rho)
    z_star = 2.0 / r
    return float(r * (norm.pdf(0.0) - norm.pdf(z_star)) + 2.0 * norm.sf(z_star))


def selective_recalibration_bound(m: int, rho: float) -> float:
    """Corollary 3: c_M = min(1, (M - 1) rho) bounds any selection attack.

    With a pre-registered shared offset, M i.i.d. calibration redraws and an
    arbitrary data-dependent selection deploy a classifier different from the
    first draw's with probability <= c_M, and every selection rule retains
    expected coverage >= 1 - alpha - c_M.
    """
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    return float(min(1.0, (int(m) - 1) * float(rho)))


def expected_min_coverage(alpha: float, n: int, m: int) -> float:
    """Exact expected coverage of the min-of-M selection attack, Appendix B eq. (19).

    For independent standard split-conformal thresholds with continuous score
    law F, C_j = F(tau_tilde_j) ~ iid Beta(k, n + 1 - k) and
    E min_j C_j = int_0^1 {1 - B_{k, n+1-k}(t)}^M dt. At M = 1 this equals
    mu_n = k/(n+1) exactly; as n -> infinity it is mu_n - a_M sigma_n +
    o(sigma_n) (eq. 20). Numerical: adaptive quadrature over the transition
    region plus the exact unit-integrand tail below it.
    """
    _validate_alpha(alpha)
    k = _check_level_attainable(alpha, int(n))
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    a_par = float(k)
    b_par = float(int(n) + 1 - k)
    mu = k / (float(n) + 1.0)
    sd = float(np.sqrt(mu * (1.0 - mu) / (float(n) + 2.0)))
    spread = sd * (float(np.sqrt(2.0 * np.log(max(float(m), 2.0))) + 1.0))
    lo = max(0.0, mu - 12.0 * spread)
    hi = min(1.0, mu + 6.0 * sd)
    mm = int(m)

    def integrand(t: float) -> float:
        return float((1.0 - betainc(a_par, b_par, t)) ** mm)

    if hi <= lo:
        return float(lo)
    val, _err = quad(integrand, lo, hi, limit=500)
    return float(lo + val)


def expected_max_normal(m: int) -> float:
    """a_M = E max_{j<=M} Z_j for iid standard normals (Corollary 3, eq. 20).

    a_M ~ sqrt(2 log M) only as M -> infinity; this is the exact finite-M
    value by quadrature of M x phi(x) Phi(x)^(M-1).
    """
    if int(m) < 1:
        raise ValueError("m must be >= 1")
    mm = int(m)
    if mm == 1:
        return 0.0

    def integrand(x: float) -> float:
        return float(mm * x * norm.pdf(x) * norm.cdf(x) ** (mm - 1))

    val, _err = quad(integrand, -12.0, 12.0, limit=300)
    return float(val)


def coverage_drop_approximation(alpha: float, n: int, m: int) -> float:
    """Eq. (20): E min_j C_j = mu_n - a_M sigma_n + o(sigma_n), fixed M, n -> infinity."""
    _validate_alpha(alpha)
    k = _check_level_attainable(alpha, int(n))
    mu = k / (float(n) + 1.0)
    sd = float(np.sqrt(mu * (1.0 - mu) / (float(n) + 2.0)))
    return float(mu - expected_max_normal(m) * sd)


def seedless_sample_requirement(eps: float, delta: float, kappa: float = 1.0) -> int:
    """Theorem 4(a): n >= ceil(32 kappa^2 log(2/delta) / eps^2) for the seedless 2-list.

    No rho^-2 factor appears: two answers cost neither a seed nor 1/rho^2,
    one answer costs both (the paper's list-replicability separation).
    """
    e = float(eps)
    if not np.isfinite(e) or e <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    kh = float(kappa)
    if not np.isfinite(kh) or kh < 1.0:
        raise ValueError("kappa must be >= 1")
    return int(np.ceil(32.0 * kh * kh * np.log(2.0 / float(delta)) / (e * e)))


# ---------------------------------------------------------------------------
# Monte-Carlo experiment harnesses (seeded SYNTHETIC fixtures; no market data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementReport:
    """Definition 1 replication spectrum for two independent analysts."""

    identity_rate: float
    set_identity_rate: float
    pointwise_agreement: float
    mean_churn: float
    conditional_churn: float
    mismatch_rate: float
    n_pairs: int
    beta: float
    disagreement_bound: float
    asymptotic_mismatch: float


def _validate_design(alpha: float, n: int) -> None:
    _validate_alpha(alpha)
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    _check_level_attainable(alpha, int(n))


def agreement_experiment(
    n: int = 20_000,
    alpha: float = 0.10,
    rho: float = 0.10,
    n_pairs: int = 200,
    seed: int = 23638,
    *,
    kappa_hat: float = 1.5,
    f_hat: float = 1.0,
    f_min: float = 1.0,
    margin: float | None = None,
    pool_size: int = 50,
    n_contexts: int = 500,
    beta: float | None = None,
) -> AgreementReport:
    """Two analysts, independent calibration samples, one shared seed per pair.

    SYNTHETIC fixture: scores are iid Unif[0, 1] (the paper's rank-transformed
    testbed — f == 1 and kappa == 1 exactly, population coverage of a
    threshold tau is tau itself). Per pair, one shared offset u ~ Unif[0, beta)
    is drawn from the pair's seed and both analysts round their independent
    thresholds up to the same grid u + beta*Z (Proposition 2 averages the
    guarantee over u). Prediction sets are answer sets over a candidate pool:
    C(x) = {y in pool : s(x, y) <= tau}. Reports the full Definition 1
    spectrum: identity (bit-equal thresholds AND bit-equal set maps),
    pointwise agreement E_X P[C(X) = C'(X)], and churn mass E_X |C △ C'|
    (also conditional on disagreement — Lemma 1 says conditional churn is at
    least one full grid cell).
    """
    _validate_design(alpha, n)
    if int(n_pairs) < 1:
        raise ValueError("n_pairs must be >= 1")
    if int(pool_size) < 1 or int(n_contexts) < 1:
        raise ValueError("pool_size and n_contexts must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    b = float(beta) if beta is not None else grid_width(alpha, int(n), rho, f_hat, kappa_hat)
    d = float(margin) if margin is not None else 0.9 * min(float(alpha), 1.0 - float(alpha))
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("beta must be positive and finite")
    base = np.random.SeedSequence(int(seed))
    children = base.spawn(int(n_pairs))
    ident = 0
    set_ident = 0
    pointwise: list[float] = []
    churn_all: list[float] = []
    churn_cond: list[float] = []
    for child in children:
        rng = np.random.default_rng(child)
        u = b * rng.random()  # one shared seed per pair (Algorithm 1)
        cal_a = rng.random(int(n))
        cal_b = rng.random(int(n))
        tau_a = replicable_conformal_threshold(cal_a, alpha, b, u)
        tau_b = replicable_conformal_threshold(cal_b, alpha, b, u)
        pool = rng.random((int(n_contexts), int(pool_size)))
        sets_a = pool <= tau_a
        sets_b = pool <= tau_b
        same = tau_a == tau_b
        ident += int(same)
        set_ident += int(bool(np.array_equal(sets_a, sets_b)))
        eq_rows = np.all(sets_a == sets_b, axis=1)
        pointwise.append(float(eq_rows.mean()))
        diff = np.count_nonzero(sets_a != sets_b, axis=1).astype(float)
        churn_all.append(float(diff.mean()))
        if np.any(~eq_rows):
            churn_cond.append(float(diff[~eq_rows].mean()))
    np_rep = int(n_pairs)
    mismatch = 1.0 - ident / np_rep
    return AgreementReport(
        identity_rate=ident / np_rep,
        set_identity_rate=set_ident / np_rep,
        pointwise_agreement=float(np.mean(pointwise)),
        mean_churn=float(np.mean(churn_all)),
        conditional_churn=float(np.mean(churn_cond)) if churn_cond else float("nan"),
        mismatch_rate=mismatch,
        n_pairs=np_rep,
        beta=b,
        disagreement_bound=disagreement_bound(alpha, int(n), b, f_min, d),
        # Prop 3(i) at kappa_hat = 1 is E[min(rho |Z| / 2, 1)]; widening the
        # grid by kappa_hat rescales the target to rho / kappa_hat.
        asymptotic_mismatch=asymptotic_mismatch_limit(min(float(rho) / max(kappa_hat, 1.0), 1.0)),
    )


@dataclass(frozen=True)
class ValidityCostReport:
    """Coverage validity and the price of replicability versus standard split CP."""

    mean_coverage_recal: float
    mean_coverage_standard: float
    mean_population_coverage_recal: float
    coverage_inflation: float
    nested_dominance_rate: float
    mean_size_recal: float
    mean_size_standard: float
    size_cost_ratio: float
    inflation_cap: float
    size_floor: float
    sample_requirement: int
    requirement_met: bool
    beta: float
    n_trials: int


def validity_cost_experiment(
    n: int = 20_000,
    alpha: float = 0.10,
    rho: float = 0.10,
    n_trials: int = 200,
    seed: int = 23639,
    *,
    kappa_hat: float = 1.5,
    f_hat: float = 1.0,
    f_max: float = 1.0,
    beta: float | None = None,
    offset: float | None = None,
    n_test: int = 4_000,
    pool_size: int = 50,
    n_contexts: int = 1_000,
    eps: float | None = None,
    delta: float = 0.05,
) -> ValidityCostReport:
    """Coverage survives the coarser threshold; the cost is measured, not assumed.

    SYNTHETIC PIT-uniform fixture as in :func:`agreement_experiment`. Per
    trial one calibration sample yields both the standard threshold tau_tilde
    and the ReCal threshold tau (shared offset per trial; ``offset`` pins one
    fixed u across trials instead). Reports empirical coverage on fresh test
    scores, exact population coverage (= tau under the uniform law), the
    nesting dominance rate (tau >= tau_tilde always, so ReCal sets cover at
    least as much pointwise), set sizes over a candidate pool, and:

    * ``coverage_inflation`` vs Proposition 3(ii)'s (1+o(1)) B_n / rho at
      kappa_hat = 1 (expected rounding excess is beta * f / 2 here);
    * ``inflation_cap`` = e_n(delta) + f_max * beta, the Theorem 2(ii) band;
    * ``size_floor`` = Corollary 4's unavoidable inflation for ANY replicable
      threshold calibrator;
    * ``sample_requirement`` = Corollary 1's certified n at the effective
      eps (default: the cell mass f_max * beta), with ``requirement_met``.
      The certified constants are conservative (Remark 2: ~5x the measured
      frontier); marginal coverage >= 1 - alpha holds unconditionally.
    """
    _validate_design(alpha, n)
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    if int(n_test) < 1 or int(pool_size) < 1 or int(n_contexts) < 1:
        raise ValueError("n_test, pool_size, n_contexts must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    b = float(beta) if beta is not None else grid_width(alpha, int(n), rho, f_hat, kappa_hat)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("beta must be positive and finite")
    if offset is not None:
        _validate_grid(b, float(offset))
    e = float(eps) if eps is not None else float(f_max) * b
    req = protocol_sample_requirement(alpha, e, rho, kappa=max(kappa_hat, 1.0), delta=delta)
    base = np.random.SeedSequence(int(seed))
    children = base.spawn(int(n_trials))
    cov_r: list[float] = []
    cov_s: list[float] = []
    pop_r: list[float] = []
    size_r: list[float] = []
    size_s: list[float] = []
    dom = 0
    for child in children:
        rng = np.random.default_rng(child)
        u = float(offset) if offset is not None else b * rng.random()
        cal = rng.random(int(n))
        tau_s = conformal_quantile(cal, alpha)
        tau_r = float(round_up_to_grid(tau_s, b, u))
        truth = rng.random(int(n_test))
        pool = rng.random((int(n_contexts), int(pool_size)))
        cov_s.append(float(np.mean(truth <= tau_s)))
        cov_r.append(float(np.mean(truth <= tau_r)))
        pop_r.append(float(min(tau_r, 1.0)))
        size_s.append(float(np.mean(np.count_nonzero(pool <= tau_s, axis=1))))
        size_r.append(float(np.mean(np.count_nonzero(pool <= tau_r, axis=1))))
        dom += int(tau_r >= tau_s)
    e_n = float(np.sqrt(np.log(2.0 / float(delta)) / (2.0 * float(n))) + 2.0 / float(n))
    mean_pop = float(np.mean(pop_r))
    return ValidityCostReport(
        mean_coverage_recal=float(np.mean(cov_r)),
        mean_coverage_standard=float(np.mean(cov_s)),
        mean_population_coverage_recal=mean_pop,
        coverage_inflation=mean_pop - (1.0 - float(alpha)),
        nested_dominance_rate=dom / int(n_trials),
        mean_size_recal=float(np.mean(size_r)),
        mean_size_standard=float(np.mean(size_s)),
        size_cost_ratio=float(np.mean(size_r) / np.mean(size_s)),
        inflation_cap=float(e_n + float(f_max) * b),
        size_floor=replicability_size_floor(alpha, int(n), rho),
        sample_requirement=req,
        requirement_met=bool(int(n) >= req),
        beta=b,
        n_trials=int(n_trials),
    )


@dataclass(frozen=True)
class GamingReport:
    """Corollary 3 selection attack: min-of-M recalibrations, standard vs ReCal."""

    standard_selected_coverage: float
    standard_expected_theory: float
    standard_honest_coverage: float
    standard_undercover_gap: float
    recal_selected_coverage: float
    recal_honest_coverage: float
    recal_stability: float
    recal_mean_distinct: float
    selection_bound_c_m: float
    coverage_floor: float
    m: int
    n_trials: int
    beta: float


def selective_recalibration_experiment(
    n: int = 20_000,
    alpha: float = 0.10,
    rho: float = 0.10,
    m: int = 20,
    n_trials: int = 100,
    seed: int = 23640,
    *,
    kappa_hat: float = 1.5,
    f_hat: float = 1.0,
    beta: float | None = None,
) -> GamingReport:
    """An adversary redraws the calibration set M times and keeps the best draw.

    Selection rule: the smallest threshold — under the nested family
    C_tau = {s <= tau} this is simultaneously the narrowest sets and the
    oracle-lowest-coverage draw (the paper's full-pool-coverage oracle stress
    test). Per trial the offset u is drawn once and shared by all M runs (a
    pre-registered seed, Corollary 3); across trials u is redrawn (the
    guarantee averages over u). SYNTHETIC PIT-uniform fixture, so population
    coverage of a threshold equals min(tau, 1) exactly.

    Standard split conformal undercovers: the selected coverage tracks
    eq. (19)'s exact E min_j C_j = mu_n - a_M sigma_n + o(sigma_n) below the
    nominal level, invisibly to any support check (H_n(T_min) ~ Beta(1, M)).
    ReCal barely moves: all M draws give one classifier except with
    probability <= c_M = min(1, (M - 1) rho) (``recal_stability`` measures the
    achieved rate), every candidate is an upward-rounded marginally valid
    threshold, so the selected coverage stays at or above nominal (paper F5).
    Same threat model as ``quant_fund.validation.leakage_redteam``'s
    selection-multiplicity blind spot; cross-reference only.
    """
    _validate_design(alpha, n)
    if int(m) < 2:
        raise ValueError("the selection adversary needs at least two recalibrations (m >= 2)")
    if int(n_trials) < 1:
        raise ValueError("n_trials must be >= 1")
    if not 0.0 < float(rho) <= 1.0:
        raise ValueError("rho must be in (0, 1]")
    b = float(beta) if beta is not None else grid_width(alpha, int(n), rho, f_hat, kappa_hat)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("beta must be positive and finite")
    base = np.random.SeedSequence(int(seed))
    children = base.spawn(int(n_trials))
    sel_std: list[float] = []
    sel_rec: list[float] = []
    honest_rec: list[float] = []
    stable = 0
    distinct: list[float] = []
    for child in children:
        rng = np.random.default_rng(child)
        u = b * rng.random()  # pre-registered seed, fixed across the M redraws
        cal = rng.random((int(m), int(n)))
        taus_s = _kth_order_stats(cal, alpha)
        taus_r = np.asarray(round_up_to_grid(taus_s, b, u), dtype=np.float64)
        sel_std.append(float(min(np.min(taus_s), 1.0)))
        sel_rec.append(float(min(np.min(taus_r), 1.0)))
        honest_rec.append(float(min(taus_r[0], 1.0)))
        n_distinct = int(np.unique(taus_r).size)
        distinct.append(float(n_distinct))
        stable += int(n_distinct == 1)
    k = _check_level_attainable(alpha, int(n))
    mu_n = k / (float(n) + 1.0)
    c_m = selective_recalibration_bound(m, rho)
    return GamingReport(
        standard_selected_coverage=float(np.mean(sel_std)),
        standard_expected_theory=expected_min_coverage(alpha, int(n), int(m)),
        standard_honest_coverage=float(mu_n),
        standard_undercover_gap=float((1.0 - float(alpha)) - np.mean(sel_std)),
        recal_selected_coverage=float(np.mean(sel_rec)),
        recal_honest_coverage=float(np.mean(honest_rec)),
        recal_stability=stable / int(n_trials),
        recal_mean_distinct=float(np.mean(distinct)),
        selection_bound_c_m=c_m,
        coverage_floor=float(1.0 - float(alpha) - c_m),
        m=int(m),
        n_trials=int(n_trials),
        beta=b,
    )


@dataclass(frozen=True)
class SeedlessReport:
    """Theorem 4: without a shared seed the deterministic grid confines analysts."""

    adjacency_rate: float
    identity_rate: float
    max_cell_gap: int
    n_pairs: int
    beta: float
    sample_requirement: int
    requirement_met: bool


def seedless_experiment(
    n: int = 40_000,
    alpha: float = 0.10,
    eps: float = 0.06,
    n_pairs: int = 200,
    seed: int = 23641,
    *,
    f_max: float = 1.0,
    kappa: float = 1.0,
    delta: float = 0.05,
) -> SeedlessReport:
    """Seedless variant: u = 0, beta = eps / (2 f_max) (Algorithm 1, Theorem 4).

    No shared randomness is available, so single-answer replication is
    impossible below failure probability 1/2; instead all analysts land, with
    probability >= 1 - delta per analyst at the certified sample size
    ceil(32 kappa^2 log(2/delta) / eps^2), in a common sample-independent
    two-element list {g, g + beta} of ADJACENT classifiers — a checkable
    property for a regulator who cannot coordinate seeds. SYNTHETIC
    PIT-uniform fixture (f == 1, kappa == 1 exactly); ``adjacency_rate`` is
    the fraction of independent analyst pairs with |cell_A - cell_B| <= 1,
    ``identity_rate`` the fraction landing on the very same grid point (not
    guaranteed seedlessly — paper F7 measured 1.000 adjacency with within-pair
    identity varying by track).
    """
    _validate_design(alpha, n)
    if int(n_pairs) < 1:
        raise ValueError("n_pairs must be >= 1")
    if not 0.0 < float(eps) < min(float(alpha), 1.0 - float(alpha)):
        raise ValueError("Theorem 4 requires 0 < eps < min(alpha, 1 - alpha)")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1)")
    b = epsilon_protocol_grid_width(eps, f_max)
    base = np.random.SeedSequence(int(seed))
    children = base.spawn(int(n_pairs))
    adj = 0
    ident = 0
    max_gap = 0
    for child in children:
        rng = np.random.default_rng(child)
        cal_a = rng.random(int(n))
        cal_b = rng.random(int(n))
        tau_a = conformal_quantile(cal_a, alpha)
        tau_b = conformal_quantile(cal_b, alpha)
        cell_a = int(np.ceil(tau_a / b))
        cell_b = int(np.ceil(tau_b / b))
        gap = abs(cell_a - cell_b)
        max_gap = max(max_gap, gap)
        adj += int(gap <= 1)
        ident += int(gap == 0)
    req = seedless_sample_requirement(eps, delta, kappa)
    return SeedlessReport(
        adjacency_rate=adj / int(n_pairs),
        identity_rate=ident / int(n_pairs),
        max_cell_gap=max_gap,
        n_pairs=int(n_pairs),
        beta=b,
        sample_requirement=req,
        requirement_met=bool(int(n) >= req),
    )


def bench_replicable_conformal(
    n: int = 8_000,
    alpha: float = 0.10,
    rho: float = 0.10,
    seed: int = 23638,
    *,
    m: int = 10,
    n_pairs: int = 100,
    n_trials: int = 100,
    eps: float = 0.06,
    kappa_hat: float = 1.0,
) -> dict[str, float | str]:
    """SYNTHETIC fixture bench for the ReCal lane. Proper scores only; no Sharpe.

    Correctness bench on the PIT-uniform testbed (never market evidence):
    agreement identity rate vs the rho target, coverage validity, the set-size
    price of replicability, the selection-attack contrast, and the seedless
    two-list confinement. kappa_hat defaults to 1 (the paper's main synthetic
    protocol; 1.5 is its recommended deployment default).
    """
    agree = agreement_experiment(n, alpha, rho, n_pairs, seed, kappa_hat=kappa_hat)
    valid = validity_cost_experiment(n, alpha, rho, n_trials, seed + 1, kappa_hat=kappa_hat)
    gaming = selective_recalibration_experiment(
        n, alpha, rho, m, n_trials, seed + 2, kappa_hat=kappa_hat
    )
    seedless = seedless_experiment(n, alpha, eps, n_pairs, seed + 3)
    return {
        "synthetic_repcon_agreement_rate": agree.identity_rate,
        "synthetic_repcon_mismatch_rate": agree.mismatch_rate,
        "synthetic_repcon_disagreement_bound": agree.disagreement_bound,
        "synthetic_repcon_pointwise_agreement": agree.pointwise_agreement,
        "synthetic_repcon_coverage": valid.mean_coverage_recal,
        "synthetic_repcon_standard_coverage": valid.mean_coverage_standard,
        "synthetic_repcon_coverage_inflation": valid.coverage_inflation,
        "synthetic_repcon_size_cost_ratio": valid.size_cost_ratio,
        "synthetic_repcon_sample_requirement": float(valid.sample_requirement),
        "synthetic_repcon_gaming_undercover_standard": gaming.standard_undercover_gap,
        "synthetic_repcon_gaming_selected_standard": gaming.standard_selected_coverage,
        "synthetic_repcon_gaming_selected_recal": gaming.recal_selected_coverage,
        "synthetic_repcon_gaming_stability": gaming.recal_stability,
        "synthetic_repcon_gaming_bound_c_m": gaming.selection_bound_c_m,
        "synthetic_repcon_seedless_adjacency_rate": seedless.adjacency_rate,
        "synthetic_repcon_seedless_identity_rate": seedless.identity_rate,
        "synthetic_n": float(n),
        "synthetic_alpha": float(alpha),
        "synthetic_rho": float(rho),
        "synthetic_m": float(m),
        "synthetic_n_pairs": float(n_pairs),
        "synthetic_n_trials": float(n_trials),
        "synthetic_beta": agree.beta,
        "synthetic_seed": float(seed),
        "synthetic_dgp": "fixture",
        "synthetic_claim": "research_metric_only",
    }
