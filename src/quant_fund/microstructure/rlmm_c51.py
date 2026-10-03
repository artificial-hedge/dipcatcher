"""Scenario-bandit robust fine-tuning for the C51 RL market maker (Algorithm C).

**Labeled SYNTHETIC** research infrastructure (lane B4-ii, wave 17): the
scenario-bandit layer of Moret & Lillo (2026) on top of the wave-16 C51
distributional-RL market maker inside the zero-intelligence LOB. Implements
the paper's Section 9 / Algorithm C: a finite pool of exogenous regime-flow
*scenarios* (schedules on the market-order clock), an EWMA difficulty score
per scenario fed by the penalized terminal score of each rollout, and a
standardized-softmax reweighting rule (with an epsilon-uniform mix and a
per-arm probability cap) that redirects training effort toward the
low-return tail of the scenario pool.

References:

- Moret, Lillo (2026). Deep learning of robust market making under
  regime-switching order flow. arXiv:2609.11614 (verified 2026-09-30 against
  the arXiv abstract + HTML full text). This module implements Section 9
  (Algorithm C): the scenario definition
  ``xi_i = ((tau_{i,k}, L_{i,k}, p_{i,k})_{k=1..K_i}, zeta_i)``, the
  piecewise-constant buy-MO-probability schedule on the MO clock, the
  reference generator ``P_0`` (equal-weight mixture of the random-persistence
  family ``tau ~ Uniform{15,30,60,120,240}``, ``L ~ Exp(1/tau)``,
  ``p_buy ~ U[0.20,0.80]`` and the correlated-direction family
  ``p_buy,k = 0.5 + s_k I_k`` with sign persistence ``rho_side = 0.85``,
  ``I ~ U[0,0.30]``), the penalized terminal score ``H_T`` (terminal PnL
  minus summed quadratic + inventory-wall penalties), the arm loss
  ``l_i = -H_T``, the difficulty EWMA ``d_i <- (1-eta)d_i + eta*l_i`` with
  ``eta = 0.05``, the standardized softmax weights
  ``w~_i = softmax(beta_sb * nu_i)`` over ``nu_i = (d_i - d-bar)/(s_d +
  eps_nu)``, the epsilon mix ``w_i = (1-eps) w~_i + eps/M``, the per-arm cap
  ``w_max = 0.05``, the annealing schedule (beta 0.0 -> 0.5, eps 1.0 -> 0.5
  over the first half of fine-tuning), and the periodic regeneration of the
  easiest 10% of the pool every 500 bandit updates.
- Bellemare, Dabney, Munos (2017). A distributional perspective on
  reinforcement learning. *ICML 2017*, arXiv:1707.06887 — the C51 TD update
  the bandit replays (implemented in ``rl_market_maker.C51MarketMaker``).
- Adams, MacKay (2007). Bayesian online changepoint detection.
  arXiv:0710.3742 — run-length framework behind the online flow filter used
  by the augmented state (implemented in
  ``rl_market_maker.FlowBiasFilter``; tests cross-check it against
  ``models.changepoint.bocpd_gaussian`` on a planted scenario switch).
- Avellaneda, Stoikov (2008) / Gueant, Lehalle, Fernandez-Tapia (2012) —
  closed-form baselines evaluated under the same scenario draws via
  ``zi_lob_simulator.as_policy`` / ``glft_policy``.

Composition (import, do not reimplement):

- ``microstructure.rl_market_maker`` owns the RLMM environment adapter and
  the agent: :class:`C51MarketMaker`, :class:`C51Config`,
  :class:`RLStateSpec`, :func:`run_rl_mm_session` (smart quoting, SMDP
  rewards, inventory gating), :func:`train_c51_market_maker`,
  :class:`FlowBiasFilter`, :func:`build_state_vector`,
  :func:`quote_exposure_imbalance`, :func:`paper_regime_flow`, the action
  grids, and the honesty schema. Reused as the single source of truth —
  every bandit rollout is an ordinary ``run_rl_mm_session`` training
  episode, so no simulator or agent code is duplicated here.
- ``microstructure.zi_lob_simulator`` owns the event engine and the flow
  interface: :class:`ZILobSimulator` calls ``flow.current()`` /
  ``flow.advance()`` per market order, and ``run_rl_mm_session`` requires
  ``flow`` to be a :class:`MarkovRegimeFlow`. :class:`ScheduledRegimeFlow`
  subclasses it so a stored scenario can drive a session unmodified.
- ``models.changepoint.bocpd_gaussian`` is NOT used inside this module: the
  online per-MO belief is the Beta-Bernoulli filter (paper Appendix A)
  living in ``rl_market_maker``; the batch Gaussian BOCPD appears only in
  tests as an independent cross-check that a planted scenario switch is
  detectable.
- ``models.bandits`` is NOT reused: Algorithm C's sampler is an EWMA
  difficulty score reweighted by a standardized softmax with a uniform mix,
  a per-arm cap, and periodic pool refresh — a different update law from
  UCB1 / epsilon-greedy / Thompson / EXP3 / Bayes-UCB, none of which
  implement the paper's Eq. (30) EWMA + standardized softmax + refresh
  semantics. Implemented directly here.

Deviations from the paper (deliberate, documented):

1. **Cap redistribution is uniform over uncapped arms.** The paper caps each
   sampling probability at ``w_max`` and "redistributes the excess over the
   remaining pool" without specifying the rule; we redistribute uniformly
   (iterative water-filling until no arm exceeds the cap, then a final
   renormalize). Deterministic either way.
2. **Penalty reconstruction from the decision-sampled inventory path.**
   ``H_T`` subtracts ``sum_t phi*(q_{t+1}^2 + (|q_{t+1}| - q_w)_+^2)``.
   ``run_rl_mm_session`` does not export per-decision penalties, so the sum
   is recomputed from the session's ``inventory_path`` sampled at decision
   times (post-initial records) — the same q_{t+1} grid the paper sums over,
   up to the runner's sampling cadence.
3. **Regime lengths are integer MO counts** (``L_k`` rounded from
   ``Exp(1/tau_k)``, floored at 1); the paper's continuous durations live on
   the MO clock and are integer-valued in any implementation.
4. **Pool size and budgets are configurable** (paper: M = 256, refresh every
   500 updates). Lane tests use tiny pools; defaults keep the paper's rates.
5. **A ``stationary`` scenario arm is added** for the evaluation mixture
   ({stationary, regime-switching} flow scenarios): a single-segment
   constant-p_buy schedule. The paper's ``P_0`` contains only the two
   regime-switching families; stationary enters here only as an evaluation
   arm, never inside the bandit pool.
6. **Scenario selection is index-sampled** from ``w`` by the bandit's seeded
   Generator (paper samples ``xi ~ w``; the mechanism is unchanged).
7. ``ScheduledRegimeFlow`` subclasses ``MarkovRegimeFlow`` to satisfy the
   session runner's isinstance gate; the parent's two placeholder states are
   inert because ``current``/``advance``/``expected_p_buy`` are overridden
   to follow the stored schedule deterministically (no RNG at replay time).

Honesty: every output is a SYNTHETIC correctness diagnostic on a synthetic
engine, never market evidence. All P&L-like keys (terminal MTM, the
penalized terminal score ``H_T``, arm losses) are namespaced
``sim_internal_*``, must never be headlined, and there is no broker
connectivity or live-trading claim anywhere in this module. Tiny-budget
tests assert plumbing, schedule fidelity, bandit-update math, and
determinism — NOT that fine-tuning beats the Algorithm-B controller at
paper scale (that comparison is the documented-optional bench battery).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.rl_market_maker import (
    RL_MM_REVISION,
    C51MarketMaker,
    run_rl_mm_session,
)
from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    MMState,
    QuotePolicy,
    RegimeState,
    ZILobConfig,
)

Array = NDArray[np.float64]

__all__ = [
    "PAPER_BANDIT_KWARGS",
    "RLMM_C51_REVISION",
    "SCENARIO_FAMILIES",
    "RegimeSegment",
    "Scenario",
    "ScenarioBandit",
    "ScenarioGenerator",
    "ScheduledRegimeFlow",
    "default_expected_mos",
    "evaluate_scenario_robustness",
    "fixed_offset_policy",
    "inventory_skew_policy",
    "penalized_terminal_score",
    "random_offset_policy",
    "run_scenario_bandit_finetuning",
    "stationary_scenario",
]

RLMM_C51_REVISION = "SYNTHETIC_RLMM_C51_SCENBANDIT_v1"

#: The two generative families of the paper's reference generator P_0
#: (Sec. 9). "stationary" exists only as an evaluation arm, never in the pool.
SCENARIO_FAMILIES: tuple[str, str] = ("random_persistence", "correlated_direction")

#: Paper defaults for the bandit (Sec. 9): EWMA rate 0.05, beta_sb annealed
#: 0.0 -> 0.5, eps annealed 1.0 -> 0.5 over the first half of fine-tuning,
#: per-arm cap 0.05, refresh of the easiest 10% every 500 updates.
PAPER_BANDIT_KWARGS: dict[str, Any] = {
    "eta_ewma": 0.05,
    "beta_start": 0.0,
    "beta_end": 0.5,
    "eps_start": 1.0,
    "eps_end": 0.5,
    "w_max": 0.05,
    "eps_nu": 1e-8,
    "refresh_every": 500,
    "refresh_frac": 0.10,
}


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (local; mirrors sibling-module style)
# ---------------------------------------------------------------------------


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def _prob(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be a probability in [0, 1], got {x!r}")
    return v


def _int_at_least(x: int, floor: int, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < floor:
        raise ValueError(f"{name} must be an int >= {floor}, got {x!r}")
    return x


def _seed_int(x: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ValueError(f"seed must be a non-negative int, got {x!r}")
    return int(x)


# ---------------------------------------------------------------------------
# Scenario representation (paper Sec. 9, xi_i)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeSegment:
    """One regime of a scenario schedule on the market-order clock.

    ``length_mos`` is the regime duration in market-order events (``L_k``,
    integer >= 1); ``p_buy`` the buy-MO probability; ``intensity_mult`` the
    MO-intensity multiplier (paper stress families keep it at 1.0 — the
    regimes modulate direction, not rate); ``tau_hint`` the exponential scale
    ``tau_k`` that generated ``L_k`` (retained for reproducibility only, as
    in the paper's scenario tuple).
    """

    length_mos: int
    p_buy: float
    intensity_mult: float = 1.0
    tau_hint: float = 60.0

    def __post_init__(self) -> None:
        _int_at_least(self.length_mos, 1, "length_mos")
        _prob(self.p_buy, "p_buy")
        _pos_finite(self.intensity_mult, "intensity_mult")
        _pos_finite(self.tau_hint, "tau_hint")


@dataclass(frozen=True)
class Scenario:
    """An exogenous regime plan ``xi_i`` (paper Sec. 9).

    A tuple of :class:`RegimeSegment` plus the generating ``family`` tag and
    the ``seed`` used to draw it (``zeta_i`` in the paper). The piecewise-
    constant buy-MO-probability schedule is ``p(m) = p_k`` whenever
    ``C_{k-1} <= m < C_k`` on the MO clock (``m`` zero-indexed); beyond the
    stored horizon the last regime is held (documented clamp — sessions can
    overrun the planned MO count when the simulated rate exceeds M-hat).
    """

    segments: tuple[RegimeSegment, ...]
    family: str
    seed: int

    def __post_init__(self) -> None:
        segs = tuple(self.segments)
        if len(segs) < 1:
            raise ValueError("scenario must contain at least one regime segment")
        for s in segs:
            if not isinstance(s, RegimeSegment):
                raise TypeError(f"scenario segments must be RegimeSegment, got {s!r}")
        object.__setattr__(self, "segments", segs)
        if not isinstance(self.family, str) or not self.family:
            raise ValueError(f"family must be a non-empty string, got {self.family!r}")
        _seed_int(self.seed)

    @property
    def n_regimes(self) -> int:
        return len(self.segments)

    @property
    def total_mos(self) -> int:
        return sum(s.length_mos for s in self.segments)

    @property
    def cumulative(self) -> tuple[int, ...]:
        """Cumulative boundaries ``C_k = sum_{j<=k} L_j`` (len K)."""
        out: list[int] = []
        acc = 0
        for s in self.segments:
            acc += s.length_mos
            out.append(acc)
        return tuple(out)

    def p_buy_at(self, mo_index: int) -> float:
        """Buy-MO probability at zero-indexed MO count ``m`` (paper schedule)."""
        m = _int_at_least(mo_index, 0, "mo_index")
        cum = self.cumulative
        k = 0
        while k < len(cum) - 1 and m >= cum[k]:
            k += 1
        return float(self.segments[k].p_buy)

    def covers(self, mo_index: int) -> bool:
        """Whether the stored schedule covers MO index ``m`` (pre-clamp)."""
        return 0 <= mo_index < self.total_mos


class ScheduledRegimeFlow(MarkovRegimeFlow):
    """A :class:`Scenario` executed on the market-order clock.

    Subclasses :class:`MarkovRegimeFlow` so it satisfies the
    ``run_rl_mm_session`` isinstance gate; ``current`` / ``advance`` /
    ``expected_p_buy`` are overridden to follow the stored
    ``(L_k, p_k)`` schedule — the parent's two placeholder states and its
    transition RNG are inert (replay is fully deterministic: the schedule
    was fixed when the scenario was drawn). ``transitions`` records
    ``(n_mo, regime_index)`` at each boundary crossing, and
    ``state_mo_counts`` extends to one counter per scenario regime.
    """

    def __init__(self, scenario: Scenario) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError(f"scenario must be a Scenario, got {scenario!r}")
        super().__init__(
            (
                RegimeState("sched_a", 1.0, 0.5),
                RegimeState("sched_b", 1.0, 0.5),
            ),
            (1.0, 1.0),
            seed=scenario.seed,
        )
        self._scenario = scenario
        self._cum = scenario.cumulative
        self._state_idx = 0
        self.state_mo_counts = [0] * scenario.n_regimes
        self.transitions = []

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def current(self) -> RegimeState:
        seg = self._scenario.segments[self._state_idx]
        return RegimeState(
            f"{self._scenario.family}#{self._state_idx}",
            seg.intensity_mult,
            seg.p_buy,
        )

    def advance(self) -> None:
        """One MO-clock tick along the stored schedule (deterministic)."""
        idx = self._state_idx
        self.state_mo_counts[idx] += 1
        self.n_mo += 1
        nxt = idx + 1
        if nxt < self._scenario.n_regimes and self.n_mo >= self._cum[nxt - 1]:
            self._state_idx = nxt
            self.transitions.append((self.n_mo, nxt))

    def expected_p_buy(self) -> float:
        """Visit-weighted mean buy probability realized so far (fail-closed)."""
        if self.n_mo == 0:
            raise ValueError("expected_p_buy undefined before any MO event")
        total = sum(
            self.state_mo_counts[k] * self._scenario.segments[k].p_buy
            for k in range(self._scenario.n_regimes)
        )
        return float(total / self.n_mo)


# ---------------------------------------------------------------------------
# Reference scenario generator P_0 (paper Sec. 9)
# ---------------------------------------------------------------------------


class ScenarioGenerator:
    """The paper's reference scenario generator ``P_0``.

    Equal-weight mixture over two families (paper Sec. 9):

    - ``random_persistence``: per regime, ``tau_k`` drawn uniformly from
      ``tau_grid`` (paper: ``{15,30,60,120,240}``), ``L_k ~ Exp(1/tau_k)``
      rounded to an int >= 1 MO, ``p_buy,k ~ U[p_lo, p_hi]`` (paper:
      ``[0.20, 0.80]``);
    - ``correlated_direction``: same ``(tau_k, L_k)`` mechanism, but
      ``p_buy,k = 0.5 + s_k I_k`` with the direction sign ``s_k`` retained
      across regime boundaries with probability ``rho_side`` (paper: 0.85;
      the first sign is a fair coin) and ``I_k ~ U[0, i_max]`` (paper:
      ``[0.00, 0.30]``).

    A draw first picks a family by a fair coin (unless ``family`` is given),
    then appends regimes until the cumulative length covers
    ``expected_mos`` (the paper's ``K_i`` stopping rule). Deterministic
    given ``seed``: each generator consumes one numpy Generator.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        tau_grid: tuple[int, ...] = (15, 30, 60, 120, 240),
        p_lo: float = 0.20,
        p_hi: float = 0.80,
        rho_side: float = 0.85,
        i_max: float = 0.30,
        intensity_mult: float = 1.0,
    ) -> None:
        _seed_int(seed)
        grid = tuple(int(t) for t in tau_grid)
        if not grid or any(t < 1 for t in grid):
            raise ValueError(f"tau_grid must be a non-empty tuple of ints >= 1, got {tau_grid!r}")
        lo = _prob(p_lo, "p_lo")
        hi = _prob(p_hi, "p_hi")
        if hi <= lo:
            raise ValueError(f"require p_lo < p_hi, got ({p_lo!r}, {p_hi!r})")
        rho = float(rho_side)
        if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho_side must lie in [0, 1], got {rho_side!r}")
        im = float(i_max)
        if not math.isfinite(im) or im < 0.0 or im > 0.5:
            raise ValueError(f"i_max must lie in [0, 0.5], got {i_max!r}")
        self._tau_grid = grid
        self._p_lo = lo
        self._p_hi = hi
        self._rho = rho
        self._i_max = im
        self._intensity = _pos_finite(intensity_mult, "intensity_mult")
        self._rng = np.random.default_rng(seed)

    @property
    def rho_side(self) -> float:
        return self._rho

    def _draw_segment(self, family: str, sign: int) -> tuple[RegimeSegment, int]:
        rng = self._rng
        tau = float(rng.choice(np.asarray(self._tau_grid, dtype=np.float64)))
        length = max(1, int(round(float(rng.exponential(tau)))))
        if family == "random_persistence":
            p = self._p_lo + (self._p_hi - self._p_lo) * float(rng.random())
        elif family == "correlated_direction":
            if float(rng.random()) > self._rho:
                sign = -sign
            p = 0.5 + sign * self._i_max * float(rng.random())
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unknown family {family!r}")
        return RegimeSegment(length, p, self._intensity, tau), sign

    def draw(self, expected_mos: int, *, family: str | None = None) -> Scenario:
        """Draw one scenario: append regimes until ``sum L_k >= expected_mos``."""
        m = _int_at_least(int(expected_mos), 1, "expected_mos")
        rng = self._rng
        fam = (
            family
            if family is not None
            else ("random_persistence" if float(rng.random()) < 0.5 else "correlated_direction")
        )
        if fam not in SCENARIO_FAMILIES:
            raise ValueError(f"family must be one of {SCENARIO_FAMILIES}, got {fam!r}")
        sign = 1 if float(rng.random()) < 0.5 else -1
        segs: list[RegimeSegment] = []
        total = 0
        while total < m:
            seg, sign = self._draw_segment(fam, sign)
            segs.append(seg)
            total += seg.length_mos
        return Scenario(tuple(segs), fam, seed=int(rng.integers(0, 2**31 - 1)))

    def initial_pool(self, pool_size: int, expected_mos: int) -> list[Scenario]:
        """Deterministic half-half pool initialization (paper Sec. 9)."""
        n = _int_at_least(pool_size, 2, "pool_size")
        m = _int_at_least(int(expected_mos), 1, "expected_mos")
        pool: list[Scenario] = []
        for i in range(n):
            fam = SCENARIO_FAMILIES[0] if i < n // 2 else SCENARIO_FAMILIES[1]
            pool.append(self.draw(m, family=fam))
        return pool

    def stationary(self, expected_mos: int, *, p_buy: float = 0.5) -> Scenario:
        """A single-segment stationary-flow scenario (evaluation arm only)."""
        m = _int_at_least(int(expected_mos), 1, "expected_mos")
        p = _prob(p_buy, "p_buy")
        return Scenario(
            (RegimeSegment(m, p, self._intensity, 1e12),),
            "stationary",
            seed=int(self._rng.integers(0, 2**31 - 1)),
        )


def stationary_scenario(expected_mos: int, *, p_buy: float = 0.5, seed: int = 0) -> Scenario:
    """Standalone stationary-flow scenario (constant ``p_buy``, one regime)."""
    m = _int_at_least(int(expected_mos), 1, "expected_mos")
    p = _prob(p_buy, "p_buy")
    _seed_int(seed)
    return Scenario((RegimeSegment(m, p, 1.0, 1e12),), "stationary", seed=seed)


def default_expected_mos(config: ZILobConfig, horizon: float) -> int:
    """The paper's ``M-hat``: expected market orders in an episode.

    With per-side MO intensity ``mu`` (total ``2*mu``) over ``horizon``
    seconds, ``M-hat = ceil(2 * mu * horizon)`` — the count scenario
    schedules are generated to cover (paper Eq. 27).
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    h = _pos_finite(horizon, "horizon")
    return max(1, int(math.ceil(2.0 * config.mu * h)))


# ---------------------------------------------------------------------------
# Scenario bandit (paper Sec. 9, Algorithm C)
# ---------------------------------------------------------------------------


def _cap_redistribute(w: Array, cap: float) -> Array:
    """Cap each arm at ``cap``; redistribute excess uniformly over uncapped
    arms (iterative water-filling), then renormalize. Deterministic."""
    out = np.asarray(w, dtype=np.float64).copy()
    for _ in range(4 * len(out) + 4):
        over = out > cap + 1e-15
        if not bool(over.any()):
            break
        excess = float((out[over] - cap).sum())
        out[over] = cap
        under = ~over
        n_under = int(under.sum())
        if n_under == 0:  # pragma: no cover - infeasible cap guarded upstream
            break
        out[under] += excess / n_under
    total = float(out.sum())
    if not math.isfinite(total) or total <= 0.0:  # pragma: no cover
        raise RuntimeError("bandit weight cap produced a degenerate distribution")
    return out / total


class ScenarioBandit:
    """The paper's Algorithm-C scenario bandit over a finite scenario pool.

    Each stored scenario is an arm carrying a difficulty score ``d_i``, an
    EWMA of observed arm losses ``l_i = -H_T`` (``eta_ewma = 0.05``,
    init 0). Before each rollout, difficulties are standardized
    (``nu_i = (d_i - d-bar)/(s_d + eps_nu)``; ``nu = 0`` when ``s_d`` is
    numerically zero), softmaxed at inverse temperature ``beta_sb``, mixed
    with the uniform distribution at weight ``eps``, capped per-arm at
    ``w_max`` (uniform redistribution — documented deviation), and the arm
    is drawn from ``w``. ``beta_sb`` anneals ``beta_start -> beta_end`` and
    ``eps`` anneals ``eps_start -> eps_end`` linearly over the first half of
    ``n_updates_planned`` bandit updates. Every ``refresh_every`` updates,
    :meth:`refresh` replaces the easiest ``refresh_frac`` of the pool with
    new ``P_0`` draws seeded at the pool-average difficulty.

    Deterministic given ``seed`` and identical select/record/refresh call
    sequences; pure numpy, torch-free.
    """

    def __init__(
        self,
        scenarios: list[Scenario],
        *,
        n_updates_planned: int,
        eta_ewma: float = 0.05,
        beta_start: float = 0.0,
        beta_end: float = 0.5,
        eps_start: float = 1.0,
        eps_end: float = 0.5,
        w_max: float = 0.05,
        eps_nu: float = 1e-8,
        refresh_every: int = 500,
        refresh_frac: float = 0.10,
        seed: int = 0,
    ) -> None:
        if not isinstance(scenarios, list) or len(scenarios) < 2:
            raise ValueError("scenarios must be a list of at least 2 Scenario arms")
        for s in scenarios:
            if not isinstance(s, Scenario):
                raise TypeError(f"pool entries must be Scenario, got {s!r}")
        self._pool = scenarios  # shared by the runner: refresh mutates in place
        self._m = len(scenarios)
        self._planned = _int_at_least(n_updates_planned, 1, "n_updates_planned")
        eta = float(eta_ewma)
        if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
            raise ValueError(f"eta_ewma must lie in (0, 1], got {eta_ewma!r}")
        self._eta = eta
        b0 = _nonneg_finite(beta_start, "beta_start")
        b1 = _nonneg_finite(beta_end, "beta_end")
        if b1 < b0:
            raise ValueError(f"beta_end ({beta_end!r}) must be >= beta_start ({beta_start!r})")
        self._beta0, self._beta1 = b0, b1
        e0 = _prob(eps_start, "eps_start")
        e1 = _prob(eps_end, "eps_end")
        if e1 > e0:
            raise ValueError(f"eps_end ({eps_end!r}) must be <= eps_start ({eps_start!r})")
        self._eps0, self._eps1 = e0, e1
        wm = float(w_max)
        if not math.isfinite(wm) or not 0.0 < wm <= 1.0:
            raise ValueError(f"w_max must lie in (0, 1], got {w_max!r}")
        if wm * self._m < 1.0 - 1e-12:
            raise ValueError(f"w_max ({wm}) x pool ({self._m}) < 1: no valid distribution exists")
        self._w_max = wm
        self._eps_nu = _pos_finite(eps_nu, "eps_nu")
        self._refresh_every = _int_at_least(refresh_every, 1, "refresh_every")
        rf = float(refresh_frac)
        if not math.isfinite(rf) or not 0.0 < rf <= 1.0:
            raise ValueError(f"refresh_frac must lie in (0, 1], got {refresh_frac!r}")
        self._refresh_frac = rf
        _seed_int(seed)
        self._rng = np.random.default_rng(seed)
        self._d = np.zeros(self._m, dtype=np.float64)
        self._updates = 0
        self._refreshes = 0
        self._last_refresh_update = 0

    # -- introspection --------------------------------------------------------

    @property
    def pool(self) -> list[Scenario]:
        return self._pool

    @property
    def n_arms(self) -> int:
        return self._m

    @property
    def n_updates(self) -> int:
        return self._updates

    @property
    def n_refreshes(self) -> int:
        return self._refreshes

    @property
    def difficulties(self) -> Array:
        return self._d.copy()

    def anneal(self, u: int) -> tuple[float, float]:
        """(beta_sb, eps) at bandit update ``u`` (linear over first half)."""
        _int_at_least(int(u), 0, "u")
        half = max(1.0, 0.5 * float(self._planned))
        f = min(1.0, float(u) / half)
        beta = self._beta0 + (self._beta1 - self._beta0) * f
        eps = self._eps0 + (self._eps1 - self._eps0) * f
        return beta, eps

    # -- weights / selection ----------------------------------------------------

    def weights(self) -> Array:
        """Current sampling distribution (paper: standardize, softmax, mix,
        cap). Sums to 1; every arm retains at least ``eps/M`` mass."""
        d = self._d
        mu = float(d.mean())
        sd = float(np.sqrt(np.mean((d - mu) ** 2)))  # population std (paper /M)
        if sd <= self._eps_nu:
            nu = np.zeros(self._m, dtype=np.float64)
        else:
            nu = (d - mu) / (sd + self._eps_nu)
        beta, eps = self.anneal(self._updates)
        logits = beta * nu
        logits -= float(logits.max())
        w = np.exp(logits)
        w /= float(w.sum())
        w = (1.0 - eps) * w + eps / self._m
        return _cap_redistribute(w, self._w_max)

    def select(self) -> int:
        """Draw one arm index from the current weights (seeded)."""
        return int(self._rng.choice(self._m, p=self.weights()))

    # -- learning ----------------------------------------------------------------

    def record(self, arm: int, loss: float) -> None:
        """EWMA difficulty update for the rolled-out arm (paper Eq. 30)."""
        a = int(arm)
        if isinstance(arm, bool) or not 0 <= a < self._m:
            raise ValueError(f"arm must be an int in [0, {self._m}), got {arm!r}")
        ell = float(loss)
        if not math.isfinite(ell):
            raise ValueError(f"loss must be finite, got {loss!r}")
        self._d[a] = (1.0 - self._eta) * self._d[a] + self._eta * ell
        self._updates += 1

    def needs_refresh(self) -> bool:
        """Whether a scheduled pool regeneration is due."""
        return (
            self._updates > 0
            and self._updates % self._refresh_every == 0
            and self._updates > self._last_refresh_update
        )

    def refresh(self, generator: ScenarioGenerator, expected_mos: int) -> list[int]:
        """Regenerate the easiest ``refresh_frac`` of the pool from P_0.

        Replaced arms receive the pool-average difficulty (treated as
        neutral until their first rollout, per the paper). Returns the
        replaced arm indices. Fail-closed when the generator is not a
        :class:`ScenarioGenerator`.
        """
        if not isinstance(generator, ScenarioGenerator):
            raise TypeError("generator must be a ScenarioGenerator")
        m = _int_at_least(int(expected_mos), 1, "expected_mos")
        k = max(1, int(self._m * self._refresh_frac))
        order = np.argsort(self._d, kind="stable")
        easiest = order[:k]
        mean_d = float(self._d.mean())
        for i in easiest:
            self._pool[int(i)] = generator.draw(m)
            self._d[int(i)] = mean_d
        self._refreshes += 1
        self._last_refresh_update = self._updates
        return sorted(int(i) for i in easiest)


# ---------------------------------------------------------------------------
# Penalized terminal score (paper Sec. 9, H_T)
# ---------------------------------------------------------------------------


def penalized_terminal_score(
    bundle: dict[str, Any],
    *,
    reward_phi: float,
    reward_wall_fraction: float,
    inventory_cap: int | None,
) -> dict[str, float]:
    """Penalized terminal score ``H_T`` of a ``run_rl_mm_session`` bundle.

    ``H_T = PnL_T - sum_t phi*(q_{t+1}^2 + (|q_{t+1}| - q_w)_+^2)`` where
    ``PnL_T`` is the terminal mark-to-market (``sim_internal_mtm_pnl_final``)
    and the penalty sum runs over the decision-sampled inventory path
    (post-initial records — the same post-decision inventory grid the paper
    sums over; documented deviation: the runner does not export the
    per-decision penalty directly). ``inventory_cap=None`` drops the
    inventory-wall term (uncapped sessions carry no ``q_w``). The arm loss
    is ``l = -H_T``.

    All returned keys are ``sim_internal_*`` simulator-internal diagnostics,
    never headline metrics.
    """
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a run_rl_mm_session result dict")
    phi = _nonneg_finite(reward_phi, "reward_phi")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    cap: int | None = None
    if inventory_cap is not None:
        cap = _int_at_least(inventory_cap, 1, "inventory_cap")
    mtm = float(bundle.get("sim_internal_mtm_pnl_final", float("nan")))
    if not math.isfinite(mtm):
        raise ValueError("bundle['sim_internal_mtm_pnl_final'] must be finite")
    path = bundle.get("inventory_path")
    if not isinstance(path, list) or len(path) < 2:
        raise ValueError("bundle['inventory_path'] must be a list with >= 2 entries")
    wall = float(cap) * fw if cap is not None else float("inf")
    q = np.asarray(path[1:], dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ValueError("bundle['inventory_path'] must be finite")
    wall_excess = np.maximum(np.abs(q) - wall, 0.0) ** 2
    penalty = float(phi * float(np.sum(q * q + wall_excess)))
    h_t = mtm - penalty
    return {
        "sim_internal_terminal_score": float(h_t),
        "sim_internal_arm_loss": float(-h_t),
        "sim_internal_mtm_pnl_final": float(mtm),
        "sim_internal_penalty_sum": float(penalty),
        "n_penalty_terms": float(q.size),
    }


# ---------------------------------------------------------------------------
# Torch-free numpy contrast policies (QuotePolicy adapters on the offset grid)
# ---------------------------------------------------------------------------


def _offset_quotes(state: MMState, delta: tuple[int, int], tick: float) -> tuple[float, float]:
    """Map an action-grid offset pair to prices; falls back to the touch
    when the mapped quotes would cross (same safety layer as the agent
    path of ``run_rl_mm_session``)."""
    bb, ba = state.best_bid, state.best_ask
    assert bb is not None and ba is not None  # guarded by callers
    db, da = delta
    bid = bb - db * tick
    ask = ba + da * tick
    if bid >= ask:
        bid, ask = bb, ba
    return (float(bid), float(ask))


def fixed_offset_policy(
    offset: tuple[int, int] = (0, 0),
    *,
    tick: float,
) -> QuotePolicy:
    """A constant quote-offset policy (torch-free contrast baseline).

    ``offset = (delta_bid, delta_ask)`` in ticks, interpreted exactly like a
    cell of the agent action grid: ``-1`` posts one tick inside the touch,
    ``0`` joins it, ``+1`` steps one tick deeper. Deterministic.
    """
    tk = _pos_finite(tick, "tick")
    db, da = offset
    for v in (db, da):
        if isinstance(v, bool) or not isinstance(v, int) or abs(v) > 1:
            raise ValueError(f"offset cells must be ints in [-1, 1], got {offset!r}")
    pair = (int(db), int(da))

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.best_bid is None or state.best_ask is None:
            return (None, None)
        return _offset_quotes(state, pair, tk)

    return policy


def random_offset_policy(
    grid: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
    *,
    tick: float,
    seed: int = 0,
) -> QuotePolicy:
    """Uniform-random action-grid policy (torch-free random baseline).

    Draws an offset pair uniformly from ``grid`` at each decision using a
    seeded numpy Generator — the numpy-only analogue of an untrained agent,
    for contrast sessions that must run without torch. Deterministic given
    ``seed``.
    """
    tk = _pos_finite(tick, "tick")
    _seed_int(seed)
    cells = tuple(grid)
    if len(cells) < 1:
        raise ValueError("grid must contain at least one offset pair")
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise ValueError(f"grid cells must be (delta_bid, delta_ask), got {cell!r}")
        for v in cell:
            if isinstance(v, bool) or not isinstance(v, int) or abs(v) > 1:
                raise ValueError(f"offset cells must be ints in [-1, 1], got {cell!r}")
    rng = np.random.default_rng(seed)

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.best_bid is None or state.best_ask is None:
            return (None, None)
        idx = int(rng.integers(len(cells)))
        return _offset_quotes(state, cells[idx], tk)

    return policy


def inventory_skew_policy(
    *,
    tick: float,
    inventory_cap: int,
    skew_fraction: float = 0.5,
) -> QuotePolicy:
    """Inventory-aware heuristic policy (torch-free contrast baseline).

    Joins the touch (``(0, 0)``) inside a neutral band; once
    ``|q| >= skew_fraction * inventory_cap`` it leans one tick toward
    flattening: long inventory -> ask one tick inside the touch and bid one
    tick deeper ``(+1, -1)``, short inventory -> mirrored ``(-1, +1)``.
    Deterministic.
    """
    tk = _pos_finite(tick, "tick")
    cap = _int_at_least(inventory_cap, 1, "inventory_cap")
    sf = float(skew_fraction)
    if not math.isfinite(sf) or not 0.0 < sf < 1.0:
        raise ValueError(f"skew_fraction must lie in (0, 1), got {skew_fraction!r}")
    thr = sf * cap

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.best_bid is None or state.best_ask is None:
            return (None, None)
        q = state.inventory
        if q >= thr:
            return _offset_quotes(state, (1, -1), tk)
        if q <= -thr:
            return _offset_quotes(state, (-1, 1), tk)
        return _offset_quotes(state, (0, 0), tk)

    return policy


# ---------------------------------------------------------------------------
# Scenario-bandit fine-tuning runner (paper Algorithm C loop)
# ---------------------------------------------------------------------------


def run_scenario_bandit_finetuning(
    *,
    agent: C51MarketMaker,
    config: ZILobConfig,
    pool: list[Scenario],
    n_episodes: int,
    horizon: float,
    bandit: ScenarioBandit | None = None,
    generator: ScenarioGenerator | None = None,
    expected_mos: int | None = None,
    seed_base: int = 0,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    learn_every: int = 1,
) -> dict[str, Any]:
    """Run the Algorithm-C selection-rollout-update loop.

    Per fine-tuning episode: the bandit draws a scenario index from its
    current weights, the agent trains one ordinary
    :func:`run_rl_mm_session` episode under the corresponding
    :class:`ScheduledRegimeFlow` (identical C51 TD update — the bandit
    changes *which* scenarios are replayed, never how they are learned
    from), the episode's penalized terminal score ``H_T`` becomes the arm
    loss ``l_i = -H_T``, and the bandit's EWMA difficulty absorbs it.
    Scheduled refreshes regenerate the easiest pool slice when ``generator``
    is given (skipped — and counted as such — otherwise).

    Deterministic given the seeds: episode sim seed ``seed_base + u``,
    bandit selection seeded by ``bandit``/``seed_base``, scenario draws by
    the pool's generator. Returns a SYNTHETIC diagnostic bundle; every
    P&L-like key is ``sim_internal_*`` and never a headline metric.
    """
    if not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    if not isinstance(pool, list) or len(pool) < 2:
        raise ValueError("pool must be a list of at least 2 Scenario arms")
    for s in pool:
        if not isinstance(s, Scenario):
            raise TypeError(f"pool entries must be Scenario, got {s!r}")
    n_ep = _int_at_least(n_episodes, 1, "n_episodes")
    h = _pos_finite(horizon, "horizon")
    _seed_int(seed_base)
    le = _int_at_least(learn_every, 1, "learn_every")
    _pos_finite(decision_interval, "decision_interval")
    _pos_finite(sample_interval, "sample_interval")
    phi = _nonneg_finite(reward_phi, "reward_phi")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    if bandit is not None and not isinstance(bandit, ScenarioBandit):
        raise TypeError("bandit must be a ScenarioBandit or None")
    if bandit is not None and bandit.pool is not pool:
        raise ValueError("bandit must be constructed over the same pool object")
    if generator is not None and not isinstance(generator, ScenarioGenerator):
        raise TypeError("generator must be a ScenarioGenerator or None")
    emos = (
        _int_at_least(int(expected_mos), 1, "expected_mos")
        if expected_mos is not None
        else default_expected_mos(config, h)
    )
    bt = bandit if bandit is not None else ScenarioBandit(pool, n_updates_planned=n_ep)

    episodes: list[dict[str, Any]] = []
    all_losses: list[float] = []
    score_path: list[float] = []
    for u in range(n_ep):
        weights = bt.weights()
        arm = bt.select()
        sc = bt.pool[arm]
        flow = ScheduledRegimeFlow(sc)
        agent.begin_episode()
        bundle = run_rl_mm_session(
            config=replace(config, seed=seed_base + u),
            horizon=h,
            agent=agent,
            training=True,
            learn_every=le,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            inventory_cap=agent.spec.inventory_cap,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            flow=flow,
        )
        score = penalized_terminal_score(
            bundle,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            inventory_cap=agent.spec.inventory_cap,
        )
        h_t = score["sim_internal_terminal_score"]
        bt.record(arm, score["sim_internal_arm_loss"])
        refreshed: list[int] = []
        if bt.needs_refresh() and generator is not None:
            refreshed = bt.refresh(generator, emos)
        all_losses.extend(bundle["loss_curve"])
        score_path.append(h_t)
        episodes.append(
            {
                "episode": u,
                "arm": arm,
                "family": sc.family,
                "scenario_seed": sc.seed,
                "n_regimes": sc.n_regimes,
                "regimes_touched": len({idx for _, idx in flow.transitions}) + 1,
                "weight_at_selection": float(weights[arm]),
                "sim_internal_terminal_score": float(h_t),
                "sim_internal_arm_loss": float(-h_t),
                "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
                "sim_internal_reward_mean": bundle["sim_internal_reward_mean"],
                "difficulty_after": float(bt.difficulties[arm]),
                "seed": bundle["seed"],
                "session_completed": bool(bundle["session_completed"]),
                "n_decisions": int(bundle["n_decisions"]),
                "n_fills": int(bundle["n_fills"]),
                "max_abs_inventory": int(bundle["max_abs_inventory"]),
                "n_inventory_gated": int(bundle["n_inventory_gated"]),
                "n_flow_transitions": len(flow.transitions),
                "refreshed_arms": refreshed,
            }
        )
    sp = np.asarray(score_path, dtype=np.float64)
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "module_revision": RLMM_C51_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "scenario_bandit_finetuning",
        "n_episodes": n_ep,
        "pool_size": len(bt.pool),
        "expected_mos": emos,
        "episodes": episodes,
        "sim_internal_score_path": score_path,
        "sim_internal_score_mean": float(sp.mean()),
        "sim_internal_score_min": float(sp.min()),
        "sim_internal_score_max": float(sp.max()),
        "loss_curve": all_losses,
        "n_updates_agent": agent.n_updates,
        "bandit": {
            "n_updates": bt.n_updates,
            "n_refreshes": bt.n_refreshes,
            "difficulties": bt.difficulties.tolist(),
            "weights_final": bt.weights().tolist(),
            "arms_drawn": [e["arm"] for e in episodes],
            "families_drawn": [e["family"] for e in episodes],
            "pool_families": [s.family for s in bt.pool],
        },
        "training_budget": {
            "horizon": float(h),
            "n_episodes": n_ep,
            "decision_interval": float(decision_interval),
            "learn_every": le,
            "reward_phi": phi,
            "reward_wall_fraction": fw,
            "seed_base": int(seed_base),
        },
    }


# ---------------------------------------------------------------------------
# Scenario-mixture robustness evaluation
# ---------------------------------------------------------------------------


def evaluate_scenario_robustness(
    *,
    config: ZILobConfig,
    horizon: float,
    agent: C51MarketMaker | None = None,
    policies: dict[str, QuotePolicy] | None = None,
    generator: ScenarioGenerator | None = None,
    expected_mos: int | None = None,
    n_scenarios: int = 2,
    families: tuple[str, ...] = ("stationary", "random_persistence", "correlated_direction"),
    seed_base: int = 101,
    stationary_p_buy: float = 0.5,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    inventory_cap: int | None = None,
) -> dict[str, Any]:
    """Paired-seed evaluation across the scenario family mixture.

    Draws ``n_scenarios`` fresh evaluation scenarios per family —
    ``stationary`` (single constant-``p_buy`` regime), ``random_persistence``,
    ``correlated_direction`` — and runs every actor (the greedy ``agent``
    plus each named classic ``policies`` entry) through
    :func:`run_rl_mm_session` on identical per-scenario seeds. Evaluation
    draws are *fresh* (never the bandit's training pool) — this is the
    held-out mixture of the {stationary, regime-switching} evaluation
    protocol.

    The agent acts greedily and is not mutated. With ``agent=None`` the
    evaluation is fully torch-free. Returns SYNTHETIC simulator-internal
    diagnostics; ``sim_internal_*`` keys are never headline metrics.
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    h = _pos_finite(horizon, "horizon")
    if agent is not None and not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker or None")
    pols: dict[str, QuotePolicy] = {}
    if policies is not None:
        for name, p in policies.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"policy names must be non-empty strings, got {name!r}")
            if not callable(p):
                raise TypeError(f"policy {name!r} must be callable")
            pols[name] = p
    if agent is None and not pols:
        raise ValueError("nothing to evaluate: provide an agent and/or policies")
    gen = generator if generator is not None else ScenarioGenerator(seed=seed_base + 977)
    if not isinstance(gen, ScenarioGenerator):
        raise TypeError("generator must be a ScenarioGenerator or None")
    ns = _int_at_least(n_scenarios, 1, "n_scenarios")
    _seed_int(seed_base)
    p_stat = _prob(stationary_p_buy, "stationary_p_buy")
    fams = tuple(families)
    if not fams:
        raise ValueError("families must be a non-empty tuple")
    for f in fams:
        if f != "stationary" and f not in SCENARIO_FAMILIES:
            raise ValueError(f"unknown scenario family {f!r}")
    cap: int | None = None
    if agent is not None:
        cap = agent.spec.inventory_cap
        if inventory_cap is not None and int(inventory_cap) != cap:
            raise ValueError(f"inventory_cap ({inventory_cap}) must equal agent cap ({cap})")
    elif inventory_cap is not None:
        cap = _int_at_least(inventory_cap, 1, "inventory_cap")
    emos = (
        _int_at_least(int(expected_mos), 1, "expected_mos")
        if expected_mos is not None
        else default_expected_mos(config, h)
    )
    actors: list[tuple[str, C51MarketMaker | None, QuotePolicy | None]] = []
    if agent is not None:
        actors.append(("c51", agent, None))
    actors.extend((name, None, p) for name, p in pols.items())

    sessions: dict[str, list[dict[str, Any]]] = {}
    for f_idx, fam in enumerate(fams):
        for i in range(ns):
            sc = (
                stationary_scenario(emos, p_buy=p_stat, seed=seed_base + 5_000_000 + i)
                if fam == "stationary"
                else gen.draw(emos, family=fam)
            )
            ep_seed = seed_base + 10_000 * f_idx + i  # paired across actors
            for name, ag, pol in actors:
                bundle = run_rl_mm_session(
                    config=replace(config, seed=ep_seed),
                    horizon=h,
                    agent=ag,
                    policy=pol,
                    training=False,
                    decision_interval=decision_interval,
                    sample_interval=sample_interval,
                    inventory_cap=cap,
                    reward_phi=reward_phi,
                    reward_wall_fraction=reward_wall_fraction,
                    flow=ScheduledRegimeFlow(sc),
                )
                path = np.asarray(bundle["sim_internal_mtm_pnl_path"], dtype=np.float64)
                finite = path[np.isfinite(path)]
                score = penalized_terminal_score(
                    bundle,
                    reward_phi=reward_phi,
                    reward_wall_fraction=reward_wall_fraction,
                    inventory_cap=cap,
                )
                sessions.setdefault(f"{name}_{fam}", []).append(
                    {
                        "seed": ep_seed,
                        "scenario_seed": sc.seed,
                        "n_regimes": sc.n_regimes,
                        "session_completed": bool(bundle["session_completed"]),
                        "max_abs_inventory": int(bundle["max_abs_inventory"]),
                        "mean_abs_inventory": float(bundle["mean_abs_inventory"]),
                        "inventory_final": int(bundle["inventory_final"]),
                        "n_fills": int(bundle["n_fills"]),
                        "n_decisions": int(bundle["n_decisions"]),
                        "n_inventory_gated": int(bundle["n_inventory_gated"]),
                        "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
                        "sim_internal_mtm_pnl_path_mean": float(finite.mean())
                        if finite.size
                        else float("nan"),
                        "sim_internal_mtm_pnl_path_std": float(finite.std())
                        if finite.size
                        else float("nan"),
                        "sim_internal_mtm_pnl_path_min": float(finite.min())
                        if finite.size
                        else float("nan"),
                        "sim_internal_terminal_score": float(score["sim_internal_terminal_score"]),
                    }
                )
    metrics: dict[str, float] = {}
    for combo, rows in sessions.items():
        n = float(len(rows))
        completed = sum(1.0 for r in rows if r["session_completed"])
        finals = np.asarray([r["sim_internal_mtm_pnl_final"] for r in rows])
        scores = np.asarray([r["sim_internal_terminal_score"] for r in rows])
        metrics[f"{combo}_session_completion_rate"] = completed / n
        metrics[f"{combo}_inventory_saturation_rate"] = (
            sum(1.0 for r in rows if cap is not None and r["max_abs_inventory"] >= cap) / n
        )
        metrics[f"{combo}_max_abs_inventory_mean"] = float(
            np.mean([r["max_abs_inventory"] for r in rows])
        )
        metrics[f"{combo}_n_fills_mean"] = float(np.mean([r["n_fills"] for r in rows]))
        metrics[f"sim_internal_mtm_pnl_final_mean_{combo}"] = float(finals.mean())
        metrics[f"sim_internal_mtm_pnl_final_std_{combo}"] = float(finals.std())
        metrics[f"sim_internal_mtm_pnl_final_min_{combo}"] = float(finals.min())
        metrics[f"sim_internal_terminal_score_mean_{combo}"] = float(scores.mean())
        metrics[f"sim_internal_terminal_score_std_{combo}"] = float(scores.std())
        metrics[f"sim_internal_terminal_score_min_{combo}"] = float(scores.min())
    if agent is not None:
        for fam in fams:
            for other in pols:
                gap = (
                    metrics[f"sim_internal_terminal_score_mean_c51_{fam}"]
                    - metrics[f"sim_internal_terminal_score_mean_{other}_{fam}"]
                )
                metrics[f"sim_internal_score_gap_c51_minus_{other}_{fam}_mean"] = gap
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "module_revision": RLMM_C51_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "scenario_robustness_evaluation",
        "note": (
            "simulator-internal synthetic policy comparison under a scenario "
            "mixture; never headline metrics, never market evidence"
        ),
        "families": fams,
        "actors": [a[0] for a in actors],
        "n_scenarios": ns,
        "seed_base": int(seed_base),
        "horizon": h,
        "expected_mos": emos,
        "inventory_cap": cap,
        "sessions": sessions,
        "metrics": metrics,
    }
