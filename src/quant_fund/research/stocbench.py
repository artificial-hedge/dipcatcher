"""StocBench: stochastic-sampler evaluation under fixed sample budgets.

Implements the benchmark protocol of Pfister, Holzschuh & Thuerey (2026),
"StocBench: A Benchmark for Generative Modeling of Stochastic Dynamics",
arXiv:2608.22309 [cs.LG], adapted from stochastic fluid flows to the
distributional-forecasting domain of this harness. The paper evaluates
transport-based and distilled few-step generative samplers on a
stochastically forced 2-D Kolmogorov flow with four protocol elements:

1. ONE-STEP DISTRIBUTIONAL ACCURACY: each sampler's generated ensemble is
   compared against a large simulated reference ensemble at every
   evaluation context (their Section 3). Here a context t is one
   forecast origin; the sampler draws n draws of its predictive ensemble
   and is scored by the strictly proper energy score against the realized
   outcome (:func:`sampler_context_scores`) and by energy distance against
   the reference ensemble (:func:`sampler_distributional_errors`).
2. INVARIANT-MEASURE PRESERVATION UNDER ROLLOUTS: the paper checks whether
   autoregressive rollouts preserve the invariant measure via the
   enstrophy spectrum (their Section 3.2). The fluid-specific spectrum is
   replaced by the per-horizon energy distance between the rollout marginal
   ensemble and the reference invariant-measure ensemble
   (:func:`rollout_invariant_drift`) — a scale-free distributional-drift
   diagnostic with the same role: it is zero iff the marginal laws agree.
3. ALEATORIC / EPISTEMIC CONTROL TASK: the paper's deterministic control
   task hands the model the realized forcing over the prediction interval,
   turning generation into a deterministic map; the residual error isolates
   the model's map error (epistemic component) while the stochastic-task
   score additionally contains the noise model's cost (aleatoric
   component). The paper's headline finding — performance does NOT
   translate between the two settings — is quantified here by the
   per-context Spearman rank correlation between the two task scores
   (:func:`aleatoric_epistemic_split`).
4. LIMITED INFERENCE BUDGETS: the paper's axis of comparison is the number
   of function evaluations (NFE) / sample budget; rankings shift between
   low and high budgets. This module makes the budget axis explicit:
   :func:`allocate_budget` apportions a fixed draw budget over evaluation
   contexts (equal split or Neyman variance-proportional allocation),
   :func:`compare_samplers` summarizes paired per-context score
   differentials with a variance-normalized statistic plus HAC and
   block-bootstrap standard errors, and :func:`budget_significance` runs
   an anytime-valid confidence sequence over the differential stream so
   significance claims are valid at whatever budget the evaluation stops
   at (no peeking correction needed).

Honesty: every metric produced here is a proper score (energy score /
energy distance, a divergence diagnostic that is nonnegative and zero iff
the distributions agree) or a proper inferential object (CS, HAC t,
bootstrap CI, rank correlation). No Sharpe/Sortino/Calmar/P&L/NAV headline
is produced; the module emits no ``sim_internal_*`` keys because it runs
no trading simulator. ``stocbench_benchmark`` is a seeded SYNTHETIC
correctness check on a known Gaussian DGP — never market evidence.
``budget_significance`` requires a DECLARED sub-Gaussian proxy sd
``sigma`` for the score-differential increments; overestimating ``sigma``
only widens the sequence (conservative), underestimating breaks validity —
the bench derives it as a multiple of a pilot sd and reports both.
Neyman allocation requires pilot per-context score variances, mirroring
classical stratified-sampling practice (Neyman 1934).

Composition (imports, does not reimplement): ``metrics.energy_score``
(Gneiting & Raftery 2007 strictly proper multivariate ensemble score),
``metrics.confidence_sequences.subgaussian_cs`` (Howard, Ramdas, McAuliffe
& Sekhon 2021 time-uniform CS), ``metrics.hac.dm_hac_tstat`` (Andrews-
Monahan prewhitened long-run variance, the Diebold-Mariano-style SE),
``metrics.bootstrap.circular_block_indices`` (Politis-Romano 1994 circular
block resampling), and ``utils.hashing.hash_bytes`` for the receipts-style
canonical experiment key used by ``research.verify`` (sort-keys canonical
JSON -> SHA-256).

References:
- Pfister, Holzschuh & Thuerey (2026). StocBench: A Benchmark for
  Generative Modeling of Stochastic Dynamics. arXiv:2608.22309 [cs.LG].
- Gneiting & Raftery (2007). Strictly proper scoring rules, prediction,
  and estimation. JASA 102(477), 359-378.
- Szekely & Rizzo (2013). Energy statistics: a class of statistics based
  on distances. J. Statist. Plann. Inference 143(8), 1249-1272,
  arXiv:1210.3927.
- Neyman (1934). On the two different aspects of the representative
  method. JRSS 97(4), 558-625 (optimal stratified allocation).
- Howard, Ramdas, McAuliffe & Sekhon (2021). Time-uniform, nonparametric,
  nonasymptotic confidence sequences. Annals of Statistics 49(2),
  1055-1080, arXiv:1810.08240.
- Diebold & Mariano (1995). Comparing predictive accuracy. JBES 13(3).
- Politis & Romano (1994). The stationary bootstrap. JASA 89(428).
- Spearman (1904) rank correlation for cross-task non-translation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from quant_fund.metrics.bootstrap import circular_block_indices
from quant_fund.metrics.confidence_sequences import subgaussian_cs
from quant_fund.metrics.energy_score import energy_score
from quant_fund.metrics.hac import dm_hac_tstat
from quant_fund.utils.hashing import SHA256_HEX_LENGTH, hash_bytes

__all__ = [
    "ALLOCATION_RULES",
    "BudgetSignificance",
    "EvaluationTask",
    "SamplerComparison",
    "aleatoric_epistemic_split",
    "allocate_budget",
    "budget_significance",
    "compare_samplers",
    "control_task_scores",
    "ensemble_energy_distance",
    "experiment_key",
    "rollout_invariant_drift",
    "sampler_context_scores",
    "sampler_distributional_errors",
    "stocbench_benchmark",
]

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: A stochastic sampler: (rng, context_index, n_draws) -> (n_draws, d) ensemble.
SamplerFn = Callable[[np.random.Generator, int, int], Array]

#: A deterministic control map: (realized_innovation_t, context_index) -> (d,).
ControlFn = Callable[[Array, int], Array]

#: Budget allocation rules for :func:`allocate_budget`.
ALLOCATION_RULES: tuple[str, str] = ("equal", "neyman")

#: Minimum draws per (context, replicate): the composed energy score needs
#: n_samples >= 2 (its ensemble X-X' term), so allocations floor at 2.
_MIN_DRAWS = 2


def _check_int(value: int, name: str, lo: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(value)
    if v < lo:
        raise ValueError(f"{name} must be >= {lo}")
    return v


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return v


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not np.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in the open interval (0, 1)")
    return a


def _as_ensemble(values: object, name: str, *, min_n: int = 2) -> Array:
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape[0] < min_n or a.shape[1] < 1:
        raise ValueError(f"{name} must be a (n, d) array with n >= {min_n}, d >= 1")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    return a


def _as_stream(values: object, name: str, *, min_len: int = 1) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size < min_len:
        raise ValueError(f"{name} must contain at least {min_len} observations")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


@dataclass(frozen=True)
class EvaluationTask:
    """A stochastic-sampler evaluation task: T contexts with references.

    ``references[t]`` is the large simulated reference ensemble for context
    t (the paper's per-step truth ensemble, shape (n_ref, d)); ``outcomes``
    is the realized observation per context used by the proper-score leg;
    ``innovations`` (optional) is the realized forcing per context that
    unlocks the control task. All arrays finite; n_ref >= 8 (the reference
    ensemble must dominate the sampler ensemble budget), T >= 4.
    """

    references: Array
    outcomes: Array
    innovations: Array | None = None
    name: str = "task"

    def __post_init__(self) -> None:
        refs = np.asarray(self.references, dtype=float)
        if refs.ndim != 3 or refs.shape[0] < 4 or refs.shape[1] < 8 or refs.shape[2] < 1:
            raise ValueError("references must be (T, n_ref, d) with T >= 4, n_ref >= 8, d >= 1")
        if not bool(np.all(np.isfinite(refs))):
            raise ValueError("references must be finite")
        obs = np.asarray(self.outcomes, dtype=float)
        if obs.ndim != 2 or obs.shape != (refs.shape[0], refs.shape[2]):
            raise ValueError("outcomes must be (T, d) matching references")
        if not bool(np.all(np.isfinite(obs))):
            raise ValueError("outcomes must be finite")
        innov = self.innovations
        if innov is not None:
            iv = np.asarray(innov, dtype=float)
            if iv.ndim != 2 or iv.shape[0] != refs.shape[0] or iv.shape[1] < 1:
                raise ValueError("innovations must be (T, k) matching references")
            if not bool(np.all(np.isfinite(iv))):
                raise ValueError("innovations must be finite")
            object.__setattr__(self, "innovations", iv)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "references", refs)
        object.__setattr__(self, "outcomes", obs)

    @property
    def n_contexts(self) -> int:
        return int(self.references.shape[0])

    @property
    def dim(self) -> int:
        return int(self.references.shape[2])


def allocate_budget(
    total_budget: int,
    n_contexts: int,
    n_replicates: int = 1,
    *,
    rule: str = "equal",
    pilot_variance: Array | None = None,
) -> IntArray:
    """Allocate a fixed total draw budget across contexts, per replicate.

    Returns ``m[t]``, the number of ensemble draws spent on context t in
    EACH replicate; total consumed is ``n_replicates * sum(m) <=
    total_budget`` (the remainder ``total_budget - n_replicates*C`` with
    ``C = total_budget // n_replicates`` is unspent). Every context gets at
    least ``_MIN_DRAWS = 2`` draws (the composed energy score needs
    n >= 2), so ``total_budget < 2 * n_contexts * n_replicates`` raises.

    Rules (the paper evaluates methods at fixed NFE budgets; this makes the
    spend rule explicit and deterministic):

    - ``"equal"``: uniform split ``C // T`` with the leftover ``C % T``
      distributed +1 to the first contexts (deterministic).
    - ``"neyman"``: Neyman (1934) optimal allocation for a stratified
      mean — every context keeps its mandatory 2-draw floor and the
      surplus ``C - 2T`` is split proportionally to ``sqrt(v_t)`` with
      ``v_t`` the pilot per-context score variance, so contexts whose
      scores fluctuate more earn more draws. ``pilot_variance`` must be a
      finite nonnegative (T,) vector, not all zero. Largest-remainder
      apportionment in descending fractional-part order (ties by lowest
      index) hits ``C`` exactly; ``sum(m) == C <= per_rep`` always holds.
    """
    budget = _check_int(total_budget, "total_budget", 1)
    n_t = _check_int(n_contexts, "n_contexts", 1)
    n_r = _check_int(n_replicates, "n_replicates", 1)
    if budget < _MIN_DRAWS * n_t * n_r:
        raise ValueError(
            f"total_budget {budget} cannot cover {_MIN_DRAWS} draws on "
            f"{n_t} contexts x {n_r} replicates"
        )
    kind = str(rule)
    if kind not in ALLOCATION_RULES:
        raise ValueError(f"rule must be one of {ALLOCATION_RULES}")
    per_rep = budget // n_r
    if kind == "equal":
        base, rem = divmod(per_rep, n_t)
        m = np.full(n_t, base, dtype=np.int64)
        m[:rem] += 1
        return m
    v = np.asarray(pilot_variance, dtype=float).reshape(-1) if pilot_variance is not None else None
    if v is None:
        raise ValueError("pilot_variance is required for rule='neyman'")
    if v.size != n_t or not bool(np.all(np.isfinite(v))):
        raise ValueError("pilot_variance must be a finite (n_contexts,) array")
    if bool(np.any(v < 0.0)):
        raise ValueError("pilot_variance must be nonnegative")
    w = np.sqrt(v)
    if float(w.sum()) <= 0.0:
        raise ValueError("pilot_variance must not be all zero")
    surplus = per_rep - _MIN_DRAWS * n_t  # >= 0 by the fail-closed check above
    extra = surplus * (w / float(w.sum()))
    m = _MIN_DRAWS + np.floor(extra).astype(np.int64)
    frac = extra - np.floor(extra)
    order = np.lexsort((np.arange(n_t), -frac))  # descending frac, ties by index
    remaining = per_rep - int(m.sum())  # in [0, n_t]: floors sum <= surplus
    for k in range(remaining):
        m[order[k]] += 1
    return m


def ensemble_energy_distance(x: Array, y: Array) -> float:
    """Energy distance between two ensembles (Szekely & Rizzo 2013).

    ``ED(P_n, Q_m) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` evaluated on
    the empirical measures (V-statistic form, all pairs including the
    diagonal). This is the energy distance BETWEEN the two empirical
    distributions, hence always >= 0 with equality iff the ensembles are
    identical multisets — unlike the unbiased U-statistic plug-in, which
    can dip slightly below 0. This is the distributional-accuracy metric the
    paper applies to generated ensembles vs large simulated reference
    ensembles (their Section 3); here it also serves as the
    invariant-measure drift diagnostic.
    """
    a = _as_ensemble(x, "x")
    b = _as_ensemble(y, "y")
    if a.shape[1] != b.shape[1]:
        raise ValueError("x and y must share the same dimension d")
    cross = float(np.mean(cdist(a, b, metric="euclidean")))
    dx = float(np.mean(cdist(a, a, metric="euclidean")))
    dy = float(np.mean(cdist(b, b, metric="euclidean")))
    return 2.0 * cross - dx - dy


def _check_draws(draws_per_context: object, n_contexts: int) -> IntArray:
    d = np.asarray(draws_per_context, dtype=np.int64).reshape(-1)
    if d.size != n_contexts:
        raise ValueError("draws_per_context must have length n_contexts")
    if bool(np.any(d < _MIN_DRAWS)):
        raise ValueError(f"every context needs >= {_MIN_DRAWS} draws")
    return d


def _check_ensemble_draw(ens: object, want_n: int, dim: int, r: int, t: int) -> Array:
    a = np.asarray(ens, dtype=float)
    if a.ndim == 1 and dim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2 or a.shape != (want_n, dim):
        raise ValueError(
            f"sampler returned shape {a.shape} at (replicate {r}, context {t}); "
            f"expected ({want_n}, {dim})"
        )
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"sampler returned non-finite draws at (replicate {r}, context {t})")
    return a


def _replicate_runs(
    task: EvaluationTask,
    sampler: SamplerFn,
    draws: IntArray,
    n_replicates: int,
    seed: int,
) -> list[list[Array]]:
    """Draw per-(replicate, context) ensembles with deterministic seeds.

    Replicate r runs a dedicated ``default_rng`` spawned from
    ``SeedSequence(seed)``; within a replicate the contexts share the rng
    stream in index order, so repeated calls are bit-identical and
    replicates are independent.
    """
    r_seeds = np.random.SeedSequence(seed).spawn(n_replicates)
    runs: list[list[Array]] = []
    for r in range(n_replicates):
        rng_r = np.random.default_rng(r_seeds[r])
        row = [
            _check_ensemble_draw(sampler(rng_r, t, int(draws[t])), int(draws[t]), task.dim, r, t)
            for t in range(task.n_contexts)
        ]
        runs.append(row)
    return runs


def sampler_context_scores(
    task: EvaluationTask,
    sampler: SamplerFn,
    draws_per_context: Array | IntArray,
    *,
    n_replicates: int = 1,
    seed: int = 0,
) -> Array:
    """(R, T) energy scores: sampler ensemble vs the realized outcome.

    Per (replicate r, context t) the sampler draws ``draws_per_context[t]``
    samples and is scored by the strictly proper energy score
    (``metrics.energy_score``, composed) against ``task.outcomes[t]`` — the
    paper's one-step evaluation with the realized observation as target.
    Lower is better. Deterministic given ``seed``.
    """
    draws = _check_draws(draws_per_context, task.n_contexts)
    n_r = _check_int(n_replicates, "n_replicates", 1)
    runs = _replicate_runs(task, sampler, draws, n_r, int(seed))
    out = np.empty((n_r, task.n_contexts), dtype=float)
    for r in range(n_r):
        for t in range(task.n_contexts):
            out[r, t] = energy_score(runs[r][t], task.outcomes[t])
    return out


def sampler_distributional_errors(
    task: EvaluationTask,
    sampler: SamplerFn,
    draws_per_context: Array | IntArray,
    *,
    n_replicates: int = 1,
    seed: int = 0,
) -> Array:
    """(R, T) energy distances: sampler ensemble vs the reference ensemble.

    The paper's headline one-step distributional-accuracy leg: per
    (replicate, context) the ensemble is compared against
    ``task.references[t]`` by :func:`ensemble_energy_distance` — >= 0,
    0 iff the laws agree. Lower is better. Deterministic given ``seed``.
    """
    draws = _check_draws(draws_per_context, task.n_contexts)
    n_r = _check_int(n_replicates, "n_replicates", 1)
    runs = _replicate_runs(task, sampler, draws, n_r, int(seed))
    out = np.empty((n_r, task.n_contexts), dtype=float)
    for r in range(n_r):
        for t in range(task.n_contexts):
            out[r, t] = ensemble_energy_distance(runs[r][t], task.references[t])
    return out


def control_task_scores(task: EvaluationTask, control_map: ControlFn) -> Array:
    """(T,) control-task errors: deterministic map vs realized outcome.

    The paper's deterministic control task hands the model the realized
    forcing; the sampler degenerates to a point map, whose energy score
    reduces to the L2 distance ``||control_map(eps_t, t) - outcome_t||``
    (the singleton ensemble has E||X - X'|| = 0). Requires
    ``task.innovations`` — without the realized forcing the control task is
    undefined and this raises (fail-closed).
    """
    if task.innovations is None:
        raise ValueError("task.innovations is required for the control task")
    out = np.empty(task.n_contexts, dtype=float)
    for t in range(task.n_contexts):
        g = np.asarray(control_map(task.innovations[t], t), dtype=float).reshape(-1)
        if g.shape != (task.dim,) or not bool(np.all(np.isfinite(g))):
            raise ValueError(f"control_map must return a finite (d,) output at context {t}")
        out[t] = float(np.linalg.norm(g - task.outcomes[t]))
    return out


@dataclass(frozen=True)
class SamplerComparison:
    """Paired comparison of two samplers' per-(replicate, context) scores.

    ``diffs = scores_a - scores_b`` flattened in budget-accrual order
    (replicate-major, contexts within): ``mean_diff`` > 0 means sampler B is
    better on this proper score (lower is better). ``var_normalized`` =
    mean/sd (dimensionless signal-to-noise; 0.0 when the stream is exactly
    constant). ``hac_t`` / ``hac_pvalue`` are the Diebold-Mariano-style
    statistic with the Andrews-Monahan prewhitened long-run variance
    (composed from ``metrics.hac``). ``boot_lo`` / ``boot_hi`` are the
    percentile circular-block-bootstrap CI of the mean difference
    (Politis-Romano indices composed from ``metrics.bootstrap``).
    """

    n: int
    mean_diff: float
    sd_diff: float
    var_normalized: float
    hac_t: float
    hac_pvalue: float
    boot_lo: float
    boot_hi: float
    alpha: float
    block_len: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the bootstrap CI excludes 0 (a real directional edge)."""
        return bool(self.boot_lo > 0.0 or self.boot_hi < 0.0)

    def as_dict(self) -> dict[str, float]:
        """Flat float mapping for receipts (no str fields)."""
        return {
            "n": float(self.n),
            "mean_diff": self.mean_diff,
            "sd_diff": self.sd_diff,
            "var_normalized": self.var_normalized,
            "hac_t": self.hac_t,
            "hac_pvalue": self.hac_pvalue,
            "boot_lo": self.boot_lo,
            "boot_hi": self.boot_hi,
            "excludes_zero": float(self.excludes_zero),
            "block_len": float(self.block_len),
        }


def compare_samplers(
    scores_a: Array,
    scores_b: Array,
    *,
    alpha: float = 0.05,
    n_boot: int = 400,
    seed: int = 0,
    block_len: int | None = None,
) -> SamplerComparison:
    """Variance-normalized paired comparison on a score-differential stream.

    ``scores_a`` / ``scores_b`` are matching-shape (R, T) or (N,) proper
    score arrays; the paired differential ``d = a - b`` is flattened in
    budget-accrual order. Requires n >= 8 (the composed HAC estimator's
    floor). The variance-normalized statistic ``mean(d)/sd(d)`` is the
    scale-free comparison number — it ranks sampler gaps independently of
    the score's units; ``sd = 0`` (identical streams) maps to 0.0 with a
    NaN HAC statistic surfaced, never hidden.
    """
    if np.shape(scores_a) != np.shape(scores_b):
        raise ValueError("scores_a and scores_b must have identical shape")
    a = _as_stream(scores_a, "scores_a", min_len=8)
    b = _as_stream(scores_b, "scores_b", min_len=8)
    a_lvl = _check_alpha(alpha)
    n_b = _check_int(n_boot, "n_boot", 20)
    d = a - b
    n = int(d.size)
    bl = (
        _check_int(block_len, "block_len", 1)
        if block_len is not None
        else max(2, int(math.floor(math.sqrt(n))))
    )
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    vnorm = mean_d / sd_d if sd_d > 0.0 else 0.0
    hac = dm_hac_tstat(d)
    idx = circular_block_indices(n, bl, n_b, seed=int(seed))
    boot_means = np.asarray(d[idx].mean(axis=1), dtype=np.float64)
    lo = float(np.quantile(boot_means, a_lvl / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - a_lvl / 2.0))
    return SamplerComparison(
        n=n,
        mean_diff=mean_d,
        sd_diff=sd_d,
        var_normalized=float(vnorm),
        hac_t=float(hac["t"]),
        hac_pvalue=float(hac["pvalue"]),
        boot_lo=lo,
        boot_hi=hi,
        alpha=a_lvl,
        block_len=bl,
    )


@dataclass(frozen=True)
class BudgetSignificance:
    """Anytime-valid significance of a score-differential stream.

    ``significant_within_budget`` is True iff the composed sub-Gaussian CS
    excludes 0 at ANY prefix of the stream — a valid claim at whatever
    budget the evaluation happened to stop at (Howard et al. 2021; no
    peeking correction). ``first_excluding`` is the smallest t (1-based
    observation count = budget consumed) whose interval excludes 0, or -1
    if the budget never sufficed. ``final_lower`` / ``final_upper`` are the
    CS endpoints at the full budget.
    """

    n: int
    alpha: float
    sigma: float
    significant_within_budget: bool
    first_excluding: int
    final_lower: float
    final_upper: float


def budget_significance(
    diffs: Array,
    *,
    sigma: float,
    alpha: float = 0.05,
) -> BudgetSignificance:
    """Budget-constrained significance via an anytime-valid CS.

    The paired score differentials accrue one observation per budget unit;
    ``subgaussian_cs`` (composed from ``metrics.confidence_sequences``,
    Howard et al. 2021 two-sided mixture boundary, rho tuned at the
    stream's own horizon) gives a time-uniform (1 - alpha) sequence, so the
    first prefix whose interval excludes 0 is a valid significance claim
    regardless of when the evaluation stopped.

    ``sigma`` is the DECLARED sub-Gaussian proxy sd of each differential —
    a design constant (like a declared boundedness constant), not a
    plug-in: validity needs the true increments to be sub-Gaussian at that
    scale. Overestimating widens the sequence (conservative); pass a pilot
    estimate inflated by a safety multiple, as ``stocbench_benchmark``
    does.
    """
    d = _as_stream(diffs, "diffs", min_len=2)
    a = _check_alpha(alpha)
    sig = _check_positive(sigma, "sigma")
    cs = subgaussian_cs(d, sigma=sig, alpha=a)
    excl = (cs.upper < 0.0) | (cs.lower > 0.0)
    first = int(np.argmax(excl) + 1) if bool(np.any(excl)) else -1
    return BudgetSignificance(
        n=int(d.size),
        alpha=a,
        sigma=sig,
        significant_within_budget=bool(np.any(excl)),
        first_excluding=first,
        final_lower=float(cs.lower[-1]),
        final_upper=float(cs.upper[-1]),
    )


def aleatoric_epistemic_split(
    stochastic_scores: Array,
    control_scores: Array,
) -> dict[str, float]:
    """Split a sampler's error into aleatoric gap vs control-task error.

    The paper's control task removes the noise model by revealing the
    realized forcing: ``control_scores`` (per context, L2 to the realized
    outcome) isolates the deterministic map's error — the epistemic
    component — while ``stochastic_scores`` additionally carry the noise
    model's cost, so ``aleatoric_gap = mean(stochastic) - mean(control)``
    estimates the aleatoric share (finite budgets can dip it below 0; the
    raw value is reported, never clipped).

    ``task_rank_corr`` is the per-context Spearman rank correlation between
    the two task scores — the paper's non-translation result (samplers that
    win the stochastic task lose the control task) reads as a low or
    negative rank correlation. Requires T >= 4 contexts.
    """
    st = np.asarray(stochastic_scores, dtype=float)
    if st.ndim == 2:
        st = st.mean(axis=0)
    st = _as_stream(st, "stochastic_scores", min_len=4)
    ct = _as_stream(control_scores, "control_scores", min_len=4)
    if st.shape != ct.shape:
        raise ValueError("stochastic and control per-context scores must align")
    mean_st = float(st.mean())
    mean_ct = float(ct.mean())
    rho_st = rankdata(st)
    rho_ct = rankdata(ct)
    sd_st, sd_ct = float(rho_st.std()), float(rho_ct.std())
    corr = float(np.corrcoef(rho_st, rho_ct)[0, 1]) if sd_st > 0.0 and sd_ct > 0.0 else 0.0
    return {
        "mean_stochastic": mean_st,
        "mean_control": mean_ct,
        "aleatoric_gap": mean_st - mean_ct,
        "task_rank_corr": corr,
        "n_contexts": float(st.size),
    }


def rollout_invariant_drift(rollouts: Array, references: Array) -> Array:
    """(H,) energy-distance drift of a rollout marginal vs the invariant law.

    The paper assesses whether autoregressive rollouts preserve the
    invariant measure via the enstrophy spectrum; here the invariant check
    is the per-horizon energy distance between the rollout ensemble
    ``rollouts[h]`` (n paths x d) and the reference invariant-measure
    ensemble ``references[h]``. ``drift[h] >= 0``, 0 iff the marginal laws
    agree; a rising curve is the distributional signature of the sampler
    leaving the invariant measure.
    """
    ro = np.asarray(rollouts, dtype=float)
    rf = np.asarray(references, dtype=float)
    if ro.ndim != 3 or rf.ndim != 3 or ro.shape[0] != rf.shape[0]:
        raise ValueError("rollouts and references must both be (H, n, d) arrays")
    if ro.shape[0] < 1 or ro.shape[1] < 2 or rf.shape[1] < 2 or ro.shape[2] != rf.shape[2]:
        raise ValueError("need H >= 1, n >= 2 per horizon, and matching d")
    if not (bool(np.all(np.isfinite(ro))) and bool(np.all(np.isfinite(rf)))):
        raise ValueError("rollouts and references must be finite")
    return np.asarray(
        [ensemble_energy_distance(ro[h], rf[h]) for h in range(ro.shape[0])],
        dtype=np.float64,
    )


def _jsonable(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def experiment_key(payload: Mapping[str, Any]) -> str:
    """Receipts-style deterministic experiment key (SHA-256 hex).

    Canonical JSON (sorted keys, tight separators, numpy normalized) hashed
    with ``utils.hashing.hash_bytes`` — the same canonicalization
    ``research.verify`` uses for receipt digests, so an experiment payload
    produces a stable 64-hex fingerprint for evidence chains.
    """
    text = json.dumps(_jsonable(dict(payload)), sort_keys=True, separators=(",", ":"))
    key = hash_bytes(text.encode())
    if len(key) != SHA256_HEX_LENGTH:
        raise ValueError("hash_bytes returned a malformed digest")
    return key


def _gaussian_context_sampler(mu: Array, sd: Array, *, shift: float, scale: float) -> SamplerFn:
    """SYNTHETIC Gaussian context sampler: draws N(mu_t + shift, (sd_t*scale)^2)."""

    def _draw(rng: np.random.Generator, t: int, n: int) -> Array:
        loc = float(mu[t]) + shift
        s = float(sd[t]) * scale
        return np.asarray(loc + s * rng.standard_normal((n, 1)), dtype=np.float64)

    return _draw


def _gaussian_control_map(mu: Array, sd: Array, *, shift: float, scale: float) -> ControlFn:
    """Deterministic control map: outcome under the OBSERVED innovation."""

    def _map(eps: Array, t: int) -> Array:
        return np.asarray([float(mu[t]) + shift + float(sd[t]) * scale * float(eps[0])])

    return _map


def stocbench_benchmark(
    seed: int = 20260823,
    n_contexts: int = 48,
    n_replicates: int = 6,
    n_reference: int = 256,
    draws_low: int = 8,
    draws_high: int = 64,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Seeded SYNTHETIC StocBench run — correctness check, not market evidence.

    DGP (all keys labeled SYNTHETIC by construction): context means
    ``mu_t = 0.8 sin(2 pi t / 16) + 0.02 t`` and a volatility regime switch
    ``sigma_t = 1.0`` for ``t < T/2`` else ``1.6``; the realized outcome is
    ``y_t = mu_t + sigma_t eps_t`` with seeded ``eps ~ N(0, 1)`` carried as
    the control-task innovation; the reference ensemble is ``n_reference``
    draws of the truth. Three samplers: ``oracle`` (exact law),
    ``shifted`` (mean bias +1.00 — epistemic error), ``misscaled``
    (dispersion x0.30 — aleatoric misfit, the under-dispersion failure
    mode of few-step distilled samplers).

    The run exercises every protocol leg: equal vs Neyman allocation of the
    per-sampler draw budget, low- vs high-budget energy scores (the paper's
    inference-budget axis), energy-distance distributional error, paired
    differentials (oracle as reference challenger) with the variance-
    normalized statistic, HAC and bootstrap CIs, anytime-valid budget
    significance with a pilot-inflated declared sigma (1.5x the low-budget
    pooled diff sd — a declared bound, not a plug-in), the control-task
    aleatoric/epistemic split with the non-translation rank correlation,
    and an AR(1) rollout invariant-measure drift curve. Returns a flat
    ``dict[str, float]``; repeated calls are bit-identical.
    """
    n_t = _check_int(n_contexts, "n_contexts", 8)
    n_r = _check_int(n_replicates, "n_replicates", 2)
    n_ref = _check_int(n_reference, "n_reference", 8)
    m_lo = _check_int(draws_low, "draws_low", _MIN_DRAWS)
    m_hi = _check_int(draws_high, "draws_high", _MIN_DRAWS)
    a_lvl = _check_alpha(alpha)
    if m_hi <= m_lo:
        raise ValueError("draws_high must exceed draws_low")

    rng = np.random.default_rng(seed)
    t_grid = np.arange(n_t, dtype=float)
    mu = 0.8 * np.sin(2.0 * np.pi * t_grid / 16.0) + 0.02 * t_grid
    sd = np.where(t_grid < n_t / 2.0, 1.0, 1.6)
    eps = rng.standard_normal((n_t, 1))
    outcomes = mu[:, None] + sd[:, None] * eps
    refs = mu[:, None, None] + sd[:, None, None] * rng.standard_normal((n_t, n_ref, 1))
    task = EvaluationTask(
        references=refs,
        outcomes=outcomes,
        innovations=eps,
        name="SYNTHETIC_stocbench_gaussian",
    )

    samplers: dict[str, SamplerFn] = {
        "oracle": _gaussian_context_sampler(mu, sd, shift=0.0, scale=1.0),
        "shifted": _gaussian_context_sampler(mu, sd, shift=1.00, scale=1.0),
        "misscaled": _gaussian_context_sampler(mu, sd, shift=0.0, scale=0.30),
    }
    controls: dict[str, ControlFn] = {
        "oracle": _gaussian_control_map(mu, sd, shift=0.0, scale=1.0),
        "shifted": _gaussian_control_map(mu, sd, shift=1.00, scale=1.0),
        "misscaled": _gaussian_control_map(mu, sd, shift=0.0, scale=0.30),
    }

    # --- budget allocation legs -------------------------------------------
    budget_per_rep = n_t * m_hi
    total_budget = budget_per_rep * n_r
    alloc_equal = allocate_budget(total_budget, n_t, n_r, rule="equal")
    pilot_draws = np.full(n_t, m_lo, dtype=np.int64)
    pilot_scores = sampler_context_scores(
        task, samplers["oracle"], pilot_draws, n_replicates=n_r, seed=seed + 101
    )
    pilot_var = pilot_scores.var(axis=0, ddof=1)
    alloc_neyman = allocate_budget(total_budget, n_t, n_r, rule="neyman", pilot_variance=pilot_var)

    # --- one-step accuracy at low and high budgets -------------------------
    draws_high_arr = np.full(n_t, m_hi, dtype=np.int64)
    scores_low = sampler_context_scores(
        task, samplers["oracle"], pilot_draws, n_replicates=n_r, seed=seed + 7
    )
    sc: dict[str, Array] = {}
    derr: dict[str, Array] = {}
    for i, name in enumerate(samplers):
        sc[name] = sampler_context_scores(
            task, samplers[name], draws_high_arr, n_replicates=n_r, seed=seed + 11 * (i + 1)
        )
        derr[name] = sampler_distributional_errors(
            task, samplers[name], draws_high_arr, n_replicates=n_r, seed=seed + 97 * (i + 1)
        )

    # --- paired differentials, oracle as reference challenger --------------
    diff_shift = sc["shifted"] - sc["oracle"]
    diff_miss = sc["misscaled"] - sc["oracle"]
    cmp_shift = compare_samplers(
        sc["shifted"], sc["oracle"], alpha=a_lvl, n_boot=400, seed=seed + 13
    )
    cmp_miss = compare_samplers(
        sc["misscaled"], sc["oracle"], alpha=a_lvl, n_boot=400, seed=seed + 17
    )
    pilot_shift = (
        sampler_context_scores(
            task, samplers["shifted"], pilot_draws, n_replicates=n_r, seed=seed + 23
        )
        - sampler_context_scores(
            task, samplers["oracle"], pilot_draws, n_replicates=n_r, seed=seed + 23
        )
    ).reshape(-1)
    pilot_miss = (
        sampler_context_scores(
            task, samplers["misscaled"], pilot_draws, n_replicates=n_r, seed=seed + 29
        )
        - sampler_context_scores(
            task, samplers["oracle"], pilot_draws, n_replicates=n_r, seed=seed + 29
        )
    ).reshape(-1)
    # Each differential stream declares its OWN sub-Gaussian bound from its
    # own pilot (inflated 1.5x — a conservative design constant, not a
    # plug-in on the evaluated stream itself).
    sig_shift = budget_significance(
        diff_shift, sigma=float(max(1.5 * pilot_shift.std(ddof=1), 0.25)), alpha=a_lvl
    )
    sig_miss = budget_significance(
        diff_miss, sigma=float(max(1.5 * pilot_miss.std(ddof=1), 0.25)), alpha=a_lvl
    )
    sig_shift_low = budget_significance(
        pilot_shift, sigma=float(max(1.5 * pilot_shift.std(ddof=1), 0.25)), alpha=a_lvl
    )

    # --- control task: aleatoric / epistemic split -------------------------
    ctrl = {name: control_task_scores(task, controls[name]) for name in controls}
    split_shift = aleatoric_epistemic_split(sc["shifted"], ctrl["shifted"])

    # --- rollout invariant-measure drift -----------------------------------
    n_path, horizon, rho, innov_sd = 256, 16, 0.9, 0.5
    stat_sd = innov_sd / math.sqrt(1.0 - rho * rho)
    rng_roll = np.random.default_rng(seed + 31)
    ref_roll = rng_roll.standard_normal((horizon, n_ref, 1)) * stat_sd
    x = np.zeros((n_path, 1))
    roll_o = np.empty((horizon, n_path, 1))
    roll_s = np.empty((horizon, n_path, 1))
    xs = np.zeros((n_path, 1))
    for h in range(horizon):
        x = rho * x + innov_sd * rng_roll.standard_normal((n_path, 1))
        xs = rho * xs + innov_sd * rng_roll.standard_normal((n_path, 1)) + 0.5 * (1.0 - rho)
        roll_o[h] = x
        roll_s[h] = xs
    drift_o = rollout_invariant_drift(roll_o, ref_roll)
    drift_s = rollout_invariant_drift(roll_s, ref_roll)

    key = experiment_key(
        {
            "bench": "stocbench",
            "seed": int(seed),
            "n_contexts": n_t,
            "n_replicates": n_r,
            "n_reference": n_ref,
            "draws_low": m_lo,
            "draws_high": m_hi,
            "alpha": a_lvl,
            "samplers": sorted(samplers),
        }
    )
    digest12 = float(int(key[:12], 16)) / float(1 << 48)

    return {
        "stocbench_alloc_equal_draws": float(alloc_equal[0]),
        "stocbench_alloc_neyman_min": float(alloc_neyman.min()),
        "stocbench_alloc_neyman_max": float(alloc_neyman.max()),
        "stocbench_alloc_neyman_spent_per_rep": float(alloc_neyman.sum()),
        "stocbench_oracle_es_low": float(scores_low.mean()),
        "stocbench_oracle_es_high": float(sc["oracle"].mean()),
        "stocbench_shifted_es_high": float(sc["shifted"].mean()),
        "stocbench_misscaled_es_high": float(sc["misscaled"].mean()),
        "stocbench_derr_oracle": float(derr["oracle"].mean()),
        "stocbench_derr_shifted": float(derr["shifted"].mean()),
        "stocbench_derr_misscaled": float(derr["misscaled"].mean()),
        "stocbench_diff_shifted_mean": cmp_shift.mean_diff,
        "stocbench_diff_shifted_vnorm": cmp_shift.var_normalized,
        "stocbench_diff_shifted_hac_t": cmp_shift.hac_t,
        "stocbench_diff_shifted_hac_p": cmp_shift.hac_pvalue,
        "stocbench_diff_shifted_boot_lo": cmp_shift.boot_lo,
        "stocbench_diff_shifted_boot_hi": cmp_shift.boot_hi,
        "stocbench_diff_misscaled_mean": cmp_miss.mean_diff,
        "stocbench_diff_misscaled_vnorm": cmp_miss.var_normalized,
        "stocbench_diff_misscaled_hac_t": cmp_miss.hac_t,
        "stocbench_sig_shifted_within_budget": float(sig_shift.significant_within_budget),
        "stocbench_sig_shifted_first_excluding": float(sig_shift.first_excluding),
        "stocbench_sig_shifted_low_within_budget": float(sig_shift_low.significant_within_budget),
        "stocbench_sig_misscaled_within_budget": float(sig_miss.significant_within_budget),
        "stocbench_sig_misscaled_first_excluding": float(sig_miss.first_excluding),
        "stocbench_declared_sigma_shifted": sig_shift.sigma,
        "stocbench_declared_sigma_misscaled": sig_miss.sigma,
        "stocbench_control_oracle": float(ctrl["oracle"].mean()),
        "stocbench_control_shifted": float(ctrl["shifted"].mean()),
        "stocbench_control_misscaled": float(ctrl["misscaled"].mean()),
        "stocbench_split_shifted_aleatoric_gap": float(split_shift["aleatoric_gap"]),
        "stocbench_split_shifted_rank_corr": float(split_shift["task_rank_corr"]),
        "stocbench_drift_oracle_h0": float(drift_o[0]),
        "stocbench_drift_oracle_last": float(drift_o[-1]),
        "stocbench_drift_shifted_h0": float(drift_s[0]),
        "stocbench_drift_shifted_last": float(drift_s[-1]),
        "stocbench_key_digest12": digest12,
    }
