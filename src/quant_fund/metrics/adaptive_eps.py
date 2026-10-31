"""e-PS: sample-efficient multiple testing with adaptive data collection.

Implements the e-value-based posterior sampling (e-PS) framework of Lin, Ma,
Ren & Wei (2026), "Sample-Efficient Multiple Testing with Adaptive Data
Collection", arXiv:2609.26651 [stat.ME] (fetched and verified 2026-09-30).

Setting (their Section 2.1): K hypotheses H_k^(0): P_k in P_k vs
H_k^(1): P_k in Q_k; at each time t the experimenter picks ONE hypothesis
A_t to sample (non-anticipating, possibly data-dependent) and observes only
Y_t = Y_t(A_t). Goal: discover ALL nonnulls with a minimal sampling budget
while controlling FDR at ARBITRARY (data-dependent) stopping times
(their Definition 2.1).

Ingredients, composed with :mod:`quant_fund.metrics.anytime_fdr`:

1. Product-form e-processes (their Eq. 3-4): increments e_{k,t} > 0 a.s.,
   E_{H_k^0}[e_{k,t} | F_{t-1}^-] <= 1 (conditionally valid given the chosen
   arm), and e_{k,t} = 1 when A_t != k. The unsampled-arm convention makes
   {E_{k,t}} a nonnegative supermartingale on the GLOBAL filtration F_t under
   H_k^0 (their Section 3.1 verification), hence a genuine e-process: plain
   e-BH on the current values is valid at any global stopping time and NO
   e-lifting adjuster is needed (contrast ``anytime_fdr.stopped_e_bh``, which
   corrects LOCAL e-processes stopped at global times, Wang,
   Dandapanthula & Ramdas 2025, arXiv:2502.08539).
2. Rejection sets R_t = e-BH(E_{1,t}, ..., E_{K,t}; alpha), reusing
   ``anytime_fdr.e_bh`` (Wang & Ramdas 2022, JRSS-B 84(3):822-852): FDR <=
   alpha under arbitrary dependence, hence at arbitrary stopping times.
3. Sampling policy (their Algorithm 1 / Eq. 6): posterior sampling over the
   MEAN LOG E-VALUE INCREMENT. Per hypothesis k maintain
   m-hat_{k,t} = log(E_{k,t}) / (n_{k,t} v 1) (empirical average of the log
   e-increments collected from k) and a variance proxy v_{k,t}; for t > K
   draw, independently for each not-yet-rejected k,

       mu-tilde_{k,t} ~ N(m-hat_{k,t-1}, v_{k,t-1} / (n_{k,t-1} v 1)),

   and sample A_t = argmax_{k not in R_{t-1}} mu-tilde_{k,t} (ties broken by
   lowest index). NOTE: this is a Gaussian/Thompson-style posterior over the
   increment mean under a flat normal prior motivated by the CLT — NOT an
   exponential-tilt or Gibbs-weight posterior over hypotheses. For t <= K the
   initialization A_t = t samples every hypothesis once (their line 4); since
   an unsampled arm has E = 1 < 1/alpha it can never already be rejected, so
   the initialization respects the support condition automatically.
4. Nested rejection sets (their Proposition 3.1): sampling ONLY from
   [K] \\ R_{t-1} (support condition) freezes rejected arms' e-values, and
   e-BH's critical value can then only loosen, so R_{t-1} subset R_{t} for
   all t ("carefree", Tavyrikov, Goeman & de Heide 2026, EJS 20(2):3178-3189).
   :class:`EPsSelector` enforces the support condition and fails closed on
   any nesting violation.
5. Sample complexity (their Theorem 3.2 and Eq. 8): with variance proxies
   v_{k,t} = rho_{n}(V_{k,n}, c_k)^2 v kappa^2,
   rho_l(v, c) = sqrt(2(v+c)/l * log(4K/delta * sqrt(1+v/c))), e-PS rejects
   all nonnulls by time T ~ K + sum_j R_j(w_j + log(K/delta)) + max{h_j, l_j}
   + H_j + sum_k m_k with probability >= 1 - delta - (boundary-crossing
   terms), governed by the growth rates gamma (nonnulls, ~= KL(Q||P)) and
   concentration of the log-e increments. Specializations: simple-vs-simple
   likelihood-ratio increments (their Eq. 12, Theorem 4.1 + lower bound
   Theorem 4.2), composite-vs-simple log-optimal/RIPr increments (Eq. 16,
   Theorem 4.3), simple-vs-composite predictable plug-in increments (Eq. 17,
   Theorem 5.1, Gaussian Corollary 5.2). All three are provided as seeded
   SYNTHETIC Gaussian planted worlds below.

Variance proxies: the paper's simulations (Section 6, Section 7.2, D.2) use
v_{k,t} = 1.1 * sigma-hat_{k,t}^2 with sigma-hat^2 the empirical variance of
the log e-value increments collected from arm k (Section 7.2 states this
explicitly); the joke-rating experiment (Section 7.1) uses v = mean of the
squared bet sizes lambda_{k,i}^2; the theorems use the rho^2 v kappa^2 radius
above. :class:`EPsSelector` defaults to the 1.1 * sample-variance-of-log-
increments choice, falls back to ``initial_variance`` before two increments
exist, and accepts a deterministic ``variance_fn(k, n_k)`` override covering
the other two styles.

Contrast with :mod:`quant_fund.models.bandits` Thompson sampling (documented,
deliberately NOT imported): ``ThompsonGaussian`` draws N(running mean,
sigma^2/n) per arm and takes the argmax — mechanically the same posterior-
sampling step e-PS uses. The objectives are opposite in the ways that matter:
(a) bandit reward is the arm's mean outcome and the best arm is pulled
forever to minimize regret; e-PS's surrogate reward is the mean LOG E-VALUE
INCREMENT, which drifts to +KL(Q||P) > 0 under H1 and <= -KL(P||Q) < 0 under
H0 (their Section 4.1), so evidence, not payoff, is accumulated; (b) e-PS
HARD-EXCLUDES rejected hypotheses from further sampling (frozen evidence =>
nested rejection sets), while a bandit concentrates on the incumbent best
arm; (c) the output is an inferential rejection set with anytime-valid FDR
control, which requires the conditionally valid product increments of Eq. 3 —
regret-optimal bandit pulls carry no such guarantee; (d) the target
statistic is tau_* = first time all nonnulls are rejected (their Eq. 11),
not cumulative regret.

Honesty contract: everything empirical in this module runs on seeded
SYNTHETIC planted worlds and reports proper discovery/calibration quantities
only — FDR, FDP, TPR, discovery sample counts. These are correctness tests,
never market evidence. No Sharpe/Sortino/Calmar/P&L/NAV content, no
live-trading claims.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.anytime_fdr import e_bh

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

# Oracle: given an arm index and the run's seeded generator, draw ONE fresh
# observation from that arm's world and return the log e-value increment
# log e_{k,t} (finite a.s. <=> e_{k,t} > 0 a.s., their Eq. 3). Stateful
# oracles (plug-in, Eq. 17) must keep per-arm predictable state.
EIncrementOracle = Callable[[int, np.random.Generator], float]

__all__ = [
    "POLICIES",
    "EIncrementOracle",
    "EPsSelector",
    "EpsRunResult",
    "FixedDesignResult",
    "GaussianLROracle",
    "GaussianPlugInOracle",
    "GaussianWorld",
    "StepSnapshot",
    "bench_eps_efficiency",
    "eps_run",
    "fixed_design_ebh",
    "make_composite_vs_simple_world",
    "make_oracle",
    "make_simple_vs_composite_world",
    "make_simple_vs_simple_world",
]

POLICIES = ("eps", "round_robin", "uniform")
SPECIALIZATIONS = ("simple_vs_simple", "composite_vs_simple", "simple_vs_composite")

_LOG_E_MAX = float(np.log(1e300))  # exp() overflow guard, matches evalues.py
_STOP_REASONS = ("budget", "all_rejected", "stop_rule")


# ------------------------------------------------------------ validation ----


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _check_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    v = int(value)
    if v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return v


def _check_positive_float(name: str, value: float) -> float:
    x = float(value)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return x


def _as_bool_mask(name: str, value: object, n_hyp: int) -> BoolArray:
    mask = np.asarray(value).reshape(-1)
    if mask.size != n_hyp:
        raise ValueError(f"{name} must have exactly n_hyp={n_hyp} entries")
    return np.asarray(mask.astype(bool), dtype=bool)


# ------------------------------------------------------------ selector ----


class EPsSelector:
    """e-PS adaptive sampling policy (Lin, Ma, Ren & Wei 2026, Algorithm 1).

    Maintains, per hypothesis k: the cumulative log e-value log E_{k,t}
    (product-form e-process, their Eq. 4), the pull count n_{k,t} (Eq. 5),
    the empirical mean log increment m-hat_{k,t} = log(E_{k,t})/(n_{k,t} v 1),
    a variance proxy v_{k,t}, and the current rejection mask R_t (owned by the
    caller, which runs e-BH).

    Posterior construction (their Eq. 6 / Algorithm 1 line 6): for every
    not-yet-rejected k draw independently

        mu-tilde_{k,t} ~ N(m-hat_{k,t-1}, v_{k,t-1} / (n_{k,t-1} v 1)),

    and select A_t = argmax over unrejected k (ties -> lowest index). This is
    a Gaussian posterior draw for the mean log e-increment under a flat normal
    prior with CLT-scale variance — a Thompson-sampling step over HYPOTHESES
    (Thompson 1933), not a Gibbs/exponential-tilt distribution. Exploitation
    favors arms with high evidence growth (nonnulls drift at +KL); exploration
    persists while the posterior variance v/n is large.

    Sampling support is restricted to [K] \\ R_{t-1} (their Proposition 3.1
    support condition), which makes the e-BH rejection sets nested
    (R_{t-1} subset R_t): discoveries are never revoked. For t <= K the
    paper's initialization A_t = t is used (line 4); with alpha < 1 an
    unsampled arm (E = 1) can never be rejected yet, so the support condition
    holds automatically (a defensive skip keeps it true regardless).

    Baselines (their Section 6): ``policy="uniform"`` samples uniformly among
    unrejected hypotheses; ``policy="round_robin"`` cycles deterministically
    through them. Both share the initialization phase and support
    restriction, so comparisons isolate the allocation rule.

    Variance proxies v_{k,t} (see module docstring): default is
    ``variance_inflation`` x sample variance of arm k's observed log
    increments (paper default inflation 1.1); arms with < 2 increments use
    ``initial_variance``; ``variance_fn(k, n_k)`` overrides everything with a
    deterministic proxy (theorem-style rho^2 v kappa^2 of their Eq. 8, or the
    mean-lambda^2 style of their Section 7.1).

    Fail-closed: bad constructor args, non-finite log increments, out-of-range
    arms, a revoked rejection (nesting violation), or selecting when every
    hypothesis is rejected raise ValueError.
    """

    def __init__(
        self,
        n_hyp: int,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        policy: str = "eps",
        variance_inflation: float = 1.1,
        initial_variance: float = 1.0,
        variance_fn: Callable[[int, int], float] | None = None,
    ) -> None:
        self._n_hyp = _check_int("n_hyp", n_hyp, 1)
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if seed is not None and rng is not None:
            raise ValueError("pass either seed or rng, not both")
        self._rng = np.random.default_rng(rng if rng is not None else seed)
        self._policy = str(policy)
        self._variance_inflation = _check_positive_float("variance_inflation", variance_inflation)
        self._initial_variance = _check_positive_float("initial_variance", initial_variance)
        if variance_fn is not None and not callable(variance_fn):
            raise ValueError("variance_fn must be callable")
        self._variance_fn = variance_fn
        self._log_e = np.zeros(self._n_hyp, dtype=float)
        self._counts = np.zeros(self._n_hyp, dtype=np.int64)
        self._inc_sum = np.zeros(self._n_hyp, dtype=float)
        self._inc_sumsq = np.zeros(self._n_hyp, dtype=float)
        self._rejected = np.zeros(self._n_hyp, dtype=bool)
        self._rr_cursor = 0
        self.t = 1  # current time step (1-based, as in Algorithm 1)

    # -- state views ---------------------------------------------------------

    @property
    def n_hyp(self) -> int:
        return self._n_hyp

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def log_e(self) -> Array:
        """Cumulative log e-values log E_{k,t} (copy)."""
        return np.array(self._log_e, dtype=float)

    @property
    def counts(self) -> IntArray:
        """Per-hypothesis pull counts n_{k,t} (copy)."""
        return np.array(self._counts, dtype=np.int64)

    @property
    def rejected(self) -> BoolArray:
        """Current rejection mask R_{t-1} (copy)."""
        return np.array(self._rejected, dtype=bool)

    @property
    def all_rejected(self) -> bool:
        return bool(np.all(self._rejected))

    def means(self) -> Array:
        """m-hat_{k,t} = log(E_{k,t}) / (n_{k,t} v 1): mean log increments."""
        return np.asarray(self._log_e / np.maximum(self._counts.astype(float), 1.0), dtype=float)

    def variance_proxies(self) -> Array:
        """Per-arm variance proxy v_{k,t} (see class docstring)."""
        if self._variance_fn is not None:
            v = np.array(
                [float(self._variance_fn(k, int(self._counts[k]))) for k in range(self._n_hyp)],
                dtype=float,
            )
            if not bool(np.all(np.isfinite(v))) or bool(np.any(v < 0.0)):
                raise ValueError("variance_fn must return finite nonnegative values")
            return v
        n = self._counts.astype(float)
        raw = (self._inc_sumsq - self._inc_sum**2 / np.maximum(n, 1.0)) / np.maximum(n - 1.0, 1.0)
        sample_var = np.maximum(raw, 0.0)
        v = np.where(n >= 2.0, self._variance_inflation * sample_var, self._initial_variance)
        return np.asarray(v, dtype=float)

    def posterior_sds(self) -> Array:
        """Posterior sd sqrt(v_{k,t} / (n_{k,t} v 1)) of their Eq. 6 draw."""
        return np.asarray(
            np.sqrt(self.variance_proxies() / np.maximum(self._counts.astype(float), 1.0)),
            dtype=float,
        )

    def e_values(self) -> Array:
        """E_{k,t} = exp(log E_{k,t}), clipped below 1e300 for e-BH safety."""
        return np.asarray(np.exp(np.minimum(self._log_e, _LOG_E_MAX)), dtype=float)

    # -- Algorithm 1 steps -----------------------------------------------------

    def select(self) -> int:
        """Choose the next hypothesis A_t (0-based index)."""
        free = ~self._rejected
        if not bool(np.any(free)):
            raise ValueError(
                "every hypothesis is rejected; the run must stop (Algorithm 1 line 13)"
            )
        if self.t <= self._n_hyp:
            # Initialization phase (line 4): A_t = t. Defensive skip keeps the
            # Proposition 3.1 support condition even if a caller marks arms.
            arm = self.t - 1
            if self._rejected[arm]:
                arm = int(np.flatnonzero(free)[0])
            return arm
        if self._policy == "eps":
            idx = np.flatnonzero(free)
            draws = self.means()[idx] + self.posterior_sds()[idx] * self._rng.standard_normal(
                idx.size
            )
            return int(idx[int(np.argmax(draws))])
        if self._policy == "round_robin":
            return self._select_round_robin(free)
        idx = np.flatnonzero(free)
        return int(idx[int(self._rng.integers(idx.size))])

    def _select_round_robin(self, free: BoolArray) -> int:
        for step in range(self._n_hyp):
            arm = (self._rr_cursor + step) % self._n_hyp
            if bool(free[arm]):
                self._rr_cursor = (arm + 1) % self._n_hyp
                return arm
        raise ValueError("no unrejected hypothesis remains to sample")  # pragma: no cover

    def observe(self, arm: int, log_increment: float) -> None:
        """Accumulate one observed log e-increment for ``arm`` (time t -> t+1)."""
        if isinstance(arm, bool) or not isinstance(arm, int) or not 0 <= arm < self._n_hyp:
            raise ValueError(f"arm must be an integer in [0, {self._n_hyp - 1}]")
        z = float(log_increment)
        if not np.isfinite(z):
            raise ValueError("log e-value increment must be finite (e_{k,t} > 0 a.s., paper Eq. 3)")
        self._log_e[arm] += z
        self._counts[arm] += 1
        self._inc_sum[arm] += z
        self._inc_sumsq[arm] += z * z
        self.t += 1

    def mark_rejected(self, rejected: Array | BoolArray | Sequence[bool]) -> None:
        """Install the e-BH rejection mask R_t; fail closed on revocations."""
        mask = _as_bool_mask("rejected mask", rejected, self._n_hyp)
        if bool(np.any(self._rejected & ~mask)):
            raise ValueError(
                "nesting violation: rejection sets must be monotone "
                "(R_{t-1} subset R_t, paper Proposition 3.1)"
            )
        self._rejected = np.array(mask, dtype=bool)


# ----------------------------------------------------------------- runs ----


@dataclass(frozen=True)
class StepSnapshot:
    """Post-update view of one step, handed to ``eps_run`` stop rules.

    Arrays are copies: mutating them cannot corrupt the run (stop rules may
    use them freely). ``t`` is the 1-based step index.
    """

    t: int
    rejected: BoolArray
    log_e: Array
    counts: IntArray


@dataclass(frozen=True)
class EpsRunResult:
    """Output of :func:`eps_run`.

    ``discovery_time`` is tau_* = first t with H1 subset R_t (their Eq. 11;
    None when no nonnull mask was given or discovery did not happen within
    the budget). ``full_rejection_time`` is the first t with R_t = [K]
    (Algorithm 1 line 13 break; usually None — null arms are never rejected
    since e-BH thresholds exceed 1/alpha > 1 >= E for pure-null evidence).
    ``fdp_final``/``tpr_final`` are the realized FDP/TPP (their Section 2.1
    definitions) of the rejection set at the stop; NaN without a nonnull mask.
    """

    arms: IntArray
    log_increments: Array
    log_e: Array
    counts: IntArray
    rejected: BoolArray
    num_rejected: int
    total_samples: int
    stopped_by: str
    discovery_time: int | None
    full_rejection_time: int | None
    fdp_final: float
    tpr_final: float
    history: tuple[BoolArray, ...] = field(default_factory=tuple)


def eps_run(
    oracle: EIncrementOracle,
    n_hyp: int,
    alpha: float,
    max_steps: int,
    *,
    seed: int | None = None,
    policy: str = "eps",
    nonnull_mask: Array | BoolArray | None = None,
    stop_rule: Callable[[StepSnapshot], bool] | None = None,
    record_history: bool = False,
    variance_inflation: float = 1.1,
    initial_variance: float = 1.0,
    variance_fn: Callable[[int, int], float] | None = None,
) -> EpsRunResult:
    """Full adaptive e-PS loop (Algorithm 1) with e-BH rejection sets.

    At each step t: select A_t via the ``policy`` (default: e-PS posterior
    sampling), draw ONE observation from the caller-supplied e-process
    increment ``oracle(arm, rng)`` (returns the log increment; it owns the
    planted world and uses the run's seeded generator), accumulate the
    product e-process, recompute R_t = e-BH(E_{1,t}, ..., E_{K,t}; alpha),
    and optionally stop. Because increments satisfy the conditional validity
    of their Eq. 3, {E_{k,t}} is an e-process on the global filtration and
    R_t has FDR <= alpha at EVERY t and at ANY stopping time expressible as
    ``stop_rule`` (their Definition 2.1); because sampling avoids R_{t-1},
    the R_t sequence is nested (their Proposition 3.1) and this function
    fails closed if that is ever violated.

    The loop breaks when (a) ``stop_rule(snapshot)`` fires — evaluated after
    each rejection-set update, (b) R_t = [K] (Algorithm 1 line 13), or
    (c) ``max_steps`` is exhausted (``stopped_by`` records which).

    Determinism: with an integer ``seed`` and a fresh oracle, the data stream
    and the policy's posterior draws come from two independent
    ``SeedSequence`` children, so the run is exactly reproducible.

    Fail-closed: invalid scalars/masks, a non-callable oracle, or a
    non-finite increment raise ValueError.
    """
    k = _check_int("n_hyp", n_hyp, 1)
    a = _check_alpha(alpha)
    steps = _check_int("max_steps", max_steps, 1)
    if not callable(oracle):
        raise ValueError("oracle must be callable: (arm, rng) -> log e-increment")
    if stop_rule is not None and not callable(stop_rule):
        raise ValueError("stop_rule must be callable or None")
    nonnull = None
    if nonnull_mask is not None:
        nonnull = _as_bool_mask("nonnull_mask", nonnull_mask, k)
        if not bool(np.any(nonnull)):
            raise ValueError("nonnull_mask must mark at least one nonnull hypothesis")

    if seed is None:
        data_rng = np.random.default_rng(None)
        selector = EPsSelector(
            k,
            policy=policy,
            variance_inflation=variance_inflation,
            initial_variance=initial_variance,
            variance_fn=variance_fn,
        )
    else:
        children = np.random.SeedSequence(int(seed)).spawn(2)
        data_rng = np.random.default_rng(children[0])
        selector = EPsSelector(
            k,
            rng=np.random.default_rng(children[1]),
            policy=policy,
            variance_inflation=variance_inflation,
            initial_variance=initial_variance,
            variance_fn=variance_fn,
        )

    arms: list[int] = []
    log_increments: list[float] = []
    history: list[BoolArray] = []
    rejected = np.zeros(k, dtype=bool)
    discovery_time: int | None = None
    full_rejection_time: int | None = None
    stopped_by = "budget"

    for step in range(1, steps + 1):
        arm = selector.select()
        z = float(oracle(arm, data_rng))
        if not np.isfinite(z):
            raise ValueError(
                f"oracle returned non-finite log increment {z!r} for arm {arm} "
                f"at t={step} (increments must satisfy e > 0 a.s., paper Eq. 3)"
            )
        selector.observe(arm, z)
        arms.append(arm)
        log_increments.append(z)
        rejected = e_bh(selector.e_values(), a).rejected
        selector.mark_rejected(rejected)
        if record_history:
            history.append(rejected)
        if nonnull is not None and discovery_time is None and bool(np.all(rejected[nonnull])):
            discovery_time = step
        if full_rejection_time is None and bool(np.all(rejected)):
            full_rejection_time = step
        if stop_rule is not None:
            snapshot = StepSnapshot(
                t=step,
                rejected=np.array(rejected, dtype=bool),
                log_e=selector.log_e,
                counts=selector.counts,
            )
            if stop_rule(snapshot):
                stopped_by = "stop_rule"
                break
        if full_rejection_time == step:
            stopped_by = "all_rejected"
            break

    fdp = float("nan")
    tpr = float("nan")
    if nonnull is not None:
        n_r = int(np.count_nonzero(rejected))
        n_false = int(np.count_nonzero(rejected & ~nonnull))
        fdp = float(n_false) / float(max(n_r, 1))
        n_nonnull = int(np.count_nonzero(nonnull))
        tpr = float(np.count_nonzero(rejected & nonnull)) / float(max(n_nonnull, 1))

    return EpsRunResult(
        arms=np.asarray(arms, dtype=np.int64),
        log_increments=np.asarray(log_increments, dtype=float),
        log_e=selector.log_e,
        counts=selector.counts,
        rejected=np.asarray(rejected, dtype=bool),
        num_rejected=int(np.count_nonzero(rejected)),
        total_samples=len(arms),
        stopped_by=stopped_by,
        discovery_time=discovery_time,
        full_rejection_time=full_rejection_time,
        fdp_final=fdp,
        tpr_final=tpr,
        history=tuple(history),
    )


@dataclass(frozen=True)
class FixedDesignResult:
    """Output of :func:`fixed_design_ebh` (non-adaptive baseline).

    ``discovery_n`` is the smallest per-arm sample size n at which the
    single-shot e-BH rejects every nonnull (None if censored at
    ``n_per_arm_max``); ``total_samples`` = n_per_arm * n_hyp at the stop.
    """

    rejected: BoolArray
    n_per_arm: int
    total_samples: int
    discovery_n: int | None
    fdp_final: float
    tpr_final: float


def fixed_design_ebh(
    oracle: EIncrementOracle,
    n_hyp: int,
    alpha: float,
    n_per_arm_max: int,
    *,
    seed: int | None = None,
    nonnull_mask: Array | BoolArray | None = None,
) -> FixedDesignResult:
    """Fixed-design e-BH baseline: n samples per arm, then e-BH once.

    Classical design/analysis separation (the regime the paper contrasts
    against, their Sections 1 and D.2 "fixed-horizon baseline"): every
    hypothesis gets the SAME pre-specified number of samples, allocated
    round by round, with no adaptivity. The product e-values remain valid
    e-values (independent across arms here), so e-BH at level ``alpha``
    controls FDR — matched to the adaptive runs by construction. Sweeps
    n = 1..``n_per_arm_max`` and stops at the first n whose rejection set
    covers all nonnulls (the fixed-design analogue of tau_*), reporting
    total cost n * n_hyp.
    """
    k = _check_int("n_hyp", n_hyp, 1)
    a = _check_alpha(alpha)
    n_max = _check_int("n_per_arm_max", n_per_arm_max, 1)
    if not callable(oracle):
        raise ValueError("oracle must be callable: (arm, rng) -> log e-increment")
    nonnull = None
    if nonnull_mask is not None:
        nonnull = _as_bool_mask("nonnull_mask", nonnull_mask, k)
        if not bool(np.any(nonnull)):
            raise ValueError("nonnull_mask must mark at least one nonnull hypothesis")

    rng = np.random.default_rng(seed)
    log_e = np.zeros(k, dtype=float)
    rejected = np.zeros(k, dtype=bool)
    discovery_n: int | None = None
    n_used = 0
    for n in range(1, n_max + 1):
        for arm in range(k):
            z = float(oracle(arm, rng))
            if not np.isfinite(z):
                raise ValueError(
                    f"oracle returned non-finite log increment {z!r} for arm {arm} at round {n}"
                )
            log_e[arm] += z
        n_used = n
        e = np.asarray(np.exp(np.minimum(log_e, _LOG_E_MAX)), dtype=float)
        rejected = e_bh(e, a).rejected
        if nonnull is not None and bool(np.all(rejected[nonnull])):
            discovery_n = n
            break

    fdp = float("nan")
    tpr = float("nan")
    if nonnull is not None:
        n_r = int(np.count_nonzero(rejected))
        fdp = float(int(np.count_nonzero(rejected & ~nonnull))) / float(max(n_r, 1))
        tpr = float(np.count_nonzero(rejected & nonnull)) / float(
            max(int(np.count_nonzero(nonnull)), 1)
        )
    return FixedDesignResult(
        rejected=rejected,
        n_per_arm=n_used,
        total_samples=n_used * k,
        discovery_n=discovery_n,
        fdp_final=fdp,
        tpr_final=tpr,
    )


# ------------------------------------------- synthetic planted worlds ----


@dataclass(frozen=True)
class GaussianWorld:
    """Seeded SYNTHETIC planted Gaussian world for one paper specialization.

    Correctness-test fixture only — never market evidence (honesty contract).
    Fields (all length n_hyp):

    - ``nonnull``: planted H1 mask (the truth; fixed across MC repetitions,
      as in the paper's Section 6 where B_k is drawn once).
    - ``theta_true``: true mean of arm k's observations.
    - ``theta_test``: hypothesized alternative mean entering the e-increment
      (simple-vs-simple / composite-vs-simple LR target; unused by the plug-in
      oracle, kept as the planted effect size there).
    - ``theta_boundary``: null-set boundary theta_0 / theta-null (RIPr anchor
      of their Section 4.2; 0 in every world built here).
    - ``var``: known variance v_k.

    ``specialization`` is one of :data:`SPECIALIZATIONS`, mapping to their
    Eq. 12 (simple-vs-simple), Eq. 16 (composite-vs-simple), Eq. 17 +
    Corollary 5.2 (simple-vs-composite).
    """

    specialization: str
    n_hyp: int
    nonnull: BoolArray
    theta_true: Array
    theta_test: Array
    theta_boundary: Array
    var: Array

    def __post_init__(self) -> None:
        if self.specialization not in SPECIALIZATIONS:
            raise ValueError(f"specialization must be one of {SPECIALIZATIONS}")
        n = _check_int("n_hyp", self.n_hyp, 2)
        for name in ("theta_true", "theta_test", "theta_boundary", "var"):
            a = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if a.size != n or not bool(np.all(np.isfinite(a))):
                raise ValueError(f"{name} must be a finite array of length n_hyp")
            object.__setattr__(self, name, a)
        if bool(np.any(self.var <= 0.0)):
            raise ValueError("var must be strictly positive")
        m = np.asarray(self.nonnull).reshape(-1)
        if m.size != n:
            raise ValueError("nonnull must have length n_hyp")
        object.__setattr__(self, "nonnull", np.asarray(m.astype(bool), dtype=bool))


def _plant_nonnull(
    n_hyp: int, k_nonnull: int, seed: int
) -> tuple[BoolArray, IntArray, np.random.Generator]:
    n = _check_int("n_hyp", n_hyp, 2)
    k = _check_int("k_nonnull", k_nonnull, 0)
    if k > n:
        raise ValueError("k_nonnull must not exceed n_hyp")
    _check_int("seed", seed, 0)
    rng = np.random.default_rng(seed)
    idx = (
        np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
        if k > 0
        else np.zeros(0, dtype=np.int64)
    )
    nonnull = np.zeros(n, dtype=bool)
    nonnull[idx] = True
    return nonnull, idx, rng


def _staggered_effects(n_hyp: int, effect: float, stagger: float) -> Array:
    e = _check_positive_float("effect", effect)
    s = float(stagger)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("stagger must be finite and >= 0")
    return np.asarray(e + s * np.arange(n_hyp, dtype=float), dtype=float)


def make_simple_vs_simple_world(
    n_hyp: int,
    k_nonnull: int,
    *,
    effect: float = 0.8,
    stagger: float = 0.2,
    var: float = 1.0,
    seed: int,
) -> GaussianWorld:
    """Planted world for their Section 4.1 (simple-vs-simple, Eq. 12).

    H_k^0: theta = 0 vs H_k^1: theta = theta_k^alt = effect + stagger*k
    (the paper's Section 6 uses theta_k = 0.02*k with v = 1). Planted truth:
    nonnull arms draw Y ~ N(theta_k^alt, v); null arms Y ~ N(0, v). The LR
    increment log e = theta_k^alt*Y/v - (theta_k^alt)^2/(2v) drifts at
    +KL = theta^2/(2v) under H1 and -KL under H0 (their Remark 1).
    """
    nonnull, _, _ = _plant_nonnull(n_hyp, k_nonnull, seed)
    v = _check_positive_float("var", var)
    theta_test = _staggered_effects(n_hyp, effect, stagger)
    theta_true = np.where(nonnull, theta_test, 0.0)
    return GaussianWorld(
        specialization="simple_vs_simple",
        n_hyp=n_hyp,
        nonnull=nonnull,
        theta_true=np.asarray(theta_true, dtype=float),
        theta_test=theta_test,
        theta_boundary=np.zeros(n_hyp, dtype=float),
        var=np.full(n_hyp, v, dtype=float),
    )


def make_composite_vs_simple_world(
    n_hyp: int,
    k_nonnull: int,
    *,
    effect: float = 0.8,
    stagger: float = 0.2,
    null_spread: float = 0.3,
    var: float = 1.0,
    seed: int,
) -> GaussianWorld:
    """Planted world for their Section 4.2 (composite-vs-simple, Eq. 16).

    H_k^0: theta <= theta_0 = 0 (composite, monotone-likelihood-ratio family)
    vs H_k^1: theta = theta_k^alt = effect + stagger*k. The log-optimal
    e-value uses the RIPr P*_k = boundary point theta_0, so the increment is
    the same closed form as the simple case with theta_0 = 0; validity holds
    for EVERY theta <= theta_0 (their Theorem 4.3(1)). Planted null truths
    are drawn uniformly in [-null_spread, 0] (strictly inside the composite
    null), nonnull truths at theta_k^alt.
    """
    nonnull, _, rng = _plant_nonnull(n_hyp, k_nonnull, seed)
    v = _check_positive_float("var", var)
    spread = float(null_spread)
    if not np.isfinite(spread) or spread < 0.0:
        raise ValueError("null_spread must be finite and >= 0")
    theta_test = _staggered_effects(n_hyp, effect, stagger)
    theta_true = np.where(nonnull, theta_test, -spread * rng.random(n_hyp))
    return GaussianWorld(
        specialization="composite_vs_simple",
        n_hyp=n_hyp,
        nonnull=nonnull,
        theta_true=np.asarray(theta_true, dtype=float),
        theta_test=theta_test,
        theta_boundary=np.zeros(n_hyp, dtype=float),
        var=np.full(n_hyp, v, dtype=float),
    )


def make_simple_vs_composite_world(
    n_hyp: int,
    k_nonnull: int,
    *,
    effect: float = 0.9,
    stagger: float = 0.25,
    var: float = 1.0,
    seed: int,
) -> GaussianWorld:
    """Planted world for their Section 5 (simple-vs-composite, Eq. 17).

    H_k^0: theta = theta-null = 0 vs H_k^1: theta != 0 (composite, direction
    unknown; Corollary 5.2 with d = 1). Planted nonnull truths alternate in
    sign, |theta_k| = effect + stagger*k; the e-increment is the plug-in LR
    against the running mean (no known alternative needed). ``theta_test``
    stores the planted effect magnitude (unused by the plug-in oracle).
    """
    nonnull, idx, rng = _plant_nonnull(n_hyp, k_nonnull, seed)
    v = _check_positive_float("var", var)
    magnitude = _staggered_effects(n_hyp, effect, stagger)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(idx)) if len(idx) else np.zeros(0)
    theta_true = np.zeros(n_hyp, dtype=float)
    theta_true[idx] = signs * magnitude[idx]
    return GaussianWorld(
        specialization="simple_vs_composite",
        n_hyp=n_hyp,
        nonnull=nonnull,
        theta_true=theta_true,
        theta_test=np.asarray(np.abs(theta_true), dtype=float),
        theta_boundary=np.zeros(n_hyp, dtype=float),
        var=np.full(n_hyp, v, dtype=float),
    )


class GaussianLROracle:
    """Likelihood-ratio increment oracle for Eq. 12 / Eq. 16 worlds.

    Stateless: each call draws Y ~ N(theta_true[k], v_k) from the passed
    generator and returns

        log e = (a_k - b_k)*Y/v_k - (a_k^2 - b_k^2)/(2*v_k),

    with a_k = ``theta_test[k]`` and b_k = ``theta_boundary[k]`` (= 0 for the
    worlds built here). This is the Radon-Nikodym increment dQ*/dP* of their
    Eq. 12 (simple null) / Eq. 16 (RIPr-anchored composite null); it is
    conditionally valid under EVERY null in the respective null set. The
    centered increment is sub-Gaussian with variance proxy (a_k - b_k)^2 / v_k
    (their Remark 2); :meth:`variance_fn` exposes that theorem-style proxy for
    ``EPsSelector(variance_fn=...)``.
    """

    def __init__(self, world: GaussianWorld) -> None:
        if world.specialization not in ("simple_vs_simple", "composite_vs_simple"):
            raise ValueError(
                "GaussianLROracle serves simple_vs_simple and composite_vs_simple worlds"
            )
        self._world = world
        self._theta = np.asarray(world.theta_true, dtype=float)
        self._sd = np.asarray(np.sqrt(world.var), dtype=float)
        self._a = np.asarray(world.theta_test - world.theta_boundary, dtype=float)
        self._b0 = np.asarray(world.theta_test**2 - world.theta_boundary**2, dtype=float) / 2.0

    def __call__(self, arm: int, rng: np.random.Generator) -> float:
        k = int(arm)
        if not 0 <= k < self._world.n_hyp:
            raise ValueError(f"arm must be in [0, {self._world.n_hyp - 1}]")
        y = float(rng.normal(self._theta[k], self._sd[k]))
        return float(self._a[k] * y / self._sd[k] ** 2 - self._b0[k] / self._sd[k] ** 2)

    def variance_fn(self, arm: int, n_pulls: int) -> float:
        """Theorem-style constant proxy sigma_k^2 = (a_k)^2/v_k (Remark 2)."""
        k = int(arm)
        if not 0 <= k < self._world.n_hyp:
            raise ValueError(f"arm must be in [0, {self._world.n_hyp - 1}]")
        _check_int("n_pulls", n_pulls, 0)
        return float(self._a[k] ** 2 / self._sd[k] ** 2)


class GaussianPlugInOracle:
    """Plug-in increment oracle for Eq. 17 worlds (Corollary 5.2, d = 1).

    Stateful and PREDICTABLE: keeps theta-hat_{k,l} = running mean of arm k's
    l observations with theta-hat_{k,0} = theta-null_k, and on the (l+1)-th
    pull returns

        log e = -(Y - theta-hat_{k,l})^2/(2 v_k) + (Y - theta-null_k)^2/(2 v_k),

    i.e. dQ-hat_{k,l}/dP-null of their Eq. 17. Because theta-hat uses only
    arm k's PAST observations, the increment is F_{t-1}-measurable and the
    product is a valid e-process under the simple null (their Theorem 5.1(1)).
    The first pull of every arm returns exactly 0 (Q-hat_{k,0} = P-null).
    A FRESH instance must be used per run/MC repetition.
    """

    def __init__(self, world: GaussianWorld) -> None:
        if world.specialization != "simple_vs_composite":
            raise ValueError("GaussianPlugInOracle serves simple_vs_composite worlds")
        self._world = world
        self._theta = np.asarray(world.theta_true, dtype=float)
        self._sd = np.asarray(np.sqrt(world.var), dtype=float)
        self._theta0 = np.asarray(world.theta_boundary, dtype=float)
        self._sum = np.zeros(world.n_hyp, dtype=float)
        self._count = np.zeros(world.n_hyp, dtype=np.int64)

    def __call__(self, arm: int, rng: np.random.Generator) -> float:
        k = int(arm)
        if not 0 <= k < self._world.n_hyp:
            raise ValueError(f"arm must be in [0, {self._world.n_hyp - 1}]")
        v = float(self._sd[k] ** 2)
        theta_hat = (
            float(self._sum[k] / self._count[k]) if self._count[k] > 0 else float(self._theta0[k])
        )
        y = float(rng.normal(self._theta[k], self._sd[k]))
        z = (-((y - theta_hat) ** 2) + (y - float(self._theta0[k])) ** 2) / (2.0 * v)
        self._sum[k] += y
        self._count[k] += 1
        return float(z)


def make_oracle(world: GaussianWorld) -> GaussianLROracle | GaussianPlugInOracle:
    """Fresh increment oracle for ``world`` (new state per run/repetition)."""
    if not isinstance(world, GaussianWorld):
        raise ValueError("world must be a GaussianWorld")
    if world.specialization == "simple_vs_composite":
        return GaussianPlugInOracle(world)
    return GaussianLROracle(world)


# ------------------------------------------------- efficiency benchmark ----


def _mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def bench_eps_efficiency(
    world: GaussianWorld,
    alpha: float = 0.1,
    *,
    n_seeds: int = 8,
    seed: int = 0,
    budget: int = 4000,
    n_per_arm_max: int = 150,
    policies: Sequence[str] = ("eps", "round_robin"),
) -> dict[str, object]:
    """Headline SYNTHETIC efficiency bench: samples to discover all nonnulls.

    Runs each adaptive ``policy`` (stopped exactly at tau_*, their Eq. 11, via
    a legitimate data-dependent stop rule — the planted H1 is deterministic
    given the world, so {H1 subset R_t} is F_t-measurable) and the
    fixed-design baseline, over ``n_seeds`` paired seeds. Reports mean/max
    total samples to full discovery, censored counts (discovery not reached
    within ``budget`` / ``n_per_arm_max``), realized FDP/TPR at the stop, and
    the e-PS speedups. All numbers come from seeded SYNTHETIC planted worlds —
    correctness evidence only, never market evidence.
    """
    if not isinstance(world, GaussianWorld):
        raise ValueError("world must be a GaussianWorld")
    a = _check_alpha(alpha)
    reps = _check_int("n_seeds", n_seeds, 1)
    base_seed = _check_int("seed", seed, 0)
    cap = _check_int("budget", budget, world.n_hyp)
    n_max = _check_int("n_per_arm_max", n_per_arm_max, 1)
    if len(policies) == 0:
        raise ValueError("policies must be nonempty")
    for pol in policies:
        if pol not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")

    nonnull = world.nonnull

    def stop_at_discovery(snap: StepSnapshot) -> bool:
        return bool(np.all(snap.rejected[nonnull]))

    per_policy: dict[str, dict[str, list[float]]] = {
        str(pol): {"samples": [], "fdp": [], "tpr": []} for pol in policies
    }
    fixed: dict[str, list[float]] = {"samples": [], "fdp": [], "tpr": []}

    for i in range(reps):
        s = base_seed + 977 * i
        for pol in policies:
            res = eps_run(
                make_oracle(world),
                world.n_hyp,
                a,
                cap,
                seed=s,
                policy=str(pol),
                nonnull_mask=nonnull,
                stop_rule=stop_at_discovery,
            )
            bucket = per_policy[str(pol)]
            disc = float(res.discovery_time) if res.discovery_time is not None else float("nan")
            bucket["samples"].append(disc)
            bucket["fdp"].append(res.fdp_final)
            bucket["tpr"].append(res.tpr_final)
        fd = fixed_design_ebh(
            make_oracle(world),
            world.n_hyp,
            a,
            n_max,
            seed=s,
            nonnull_mask=nonnull,
        )
        fixed["samples"].append(
            float(fd.total_samples) if fd.discovery_n is not None else float("nan")
        )
        fixed["fdp"].append(fd.fdp_final)
        fixed["tpr"].append(fd.tpr_final)

    out: dict[str, object] = {
        "label": "SYNTHETIC planted-world e-PS correctness bench — not market evidence",
        "synthetic": True,
        "specialization": world.specialization,
        "n_hyp": int(world.n_hyp),
        "k_nonnull": int(np.count_nonzero(nonnull)),
        "alpha": float(a),
        "n_seeds": int(reps),
        "budget": int(cap),
        "n_per_arm_max": int(n_max),
    }
    means: dict[str, float] = {}
    for pol, bucket in per_policy.items():
        censored = int(sum(1 for v in bucket["samples"] if not np.isfinite(v)))
        out[f"{pol}_discovery_samples_mean"] = _mean_or_nan(bucket["samples"])
        out[f"{pol}_discovery_samples_max"] = (
            float(np.nanmax(bucket["samples"])) if censored < reps else float("nan")
        )
        out[f"{pol}_censored"] = censored
        out[f"{pol}_fdp_at_discovery_mean"] = _mean_or_nan(bucket["fdp"])
        out[f"{pol}_tpr_mean"] = _mean_or_nan(bucket["tpr"])
        means[pol] = float(out[f"{pol}_discovery_samples_mean"])  # type: ignore[arg-type]
    fixed_censored = int(sum(1 for v in fixed["samples"] if not np.isfinite(v)))
    out["fixed_design_discovery_samples_mean"] = _mean_or_nan(fixed["samples"])
    out["fixed_design_censored"] = fixed_censored
    out["fixed_design_fdp_at_discovery_mean"] = _mean_or_nan(fixed["fdp"])
    out["fixed_design_tpr_mean"] = _mean_or_nan(fixed["tpr"])

    eps_mean = means.get("eps", float("nan"))
    for name, other in (
        ("round_robin", means.get("round_robin", float("nan"))),
        ("fixed_design", float(out["fixed_design_discovery_samples_mean"])),  # type: ignore[arg-type]
    ):
        speedup = (
            float(other) / float(eps_mean)
            if np.isfinite(eps_mean) and eps_mean > 0.0 and np.isfinite(other)
            else float("nan")
        )
        out[f"eps_speedup_vs_{name}"] = speedup
    return out
