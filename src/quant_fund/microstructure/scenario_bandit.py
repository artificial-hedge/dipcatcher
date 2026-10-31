"""Scenario-bandit robust fine-tuning (Moret & Lillo 2026, Algorithm C).

**Labeled SYNTHETIC** research infrastructure — the third stage of the paper's
training pipeline on top of the wave-16 C51 market maker
(``microstructure.rl_market_maker``) and the wave-14 ZI-LOB
(``microstructure.zi_lob_simulator``). Citations as in
``rl_market_maker``: arXiv:2609.11614 Sec. 9 / Appendix on the scenario pool,
difficulty EWMA, softmax reweighting, and adaptive pool refresh.

Paper formulation implemented here (arXiv:2609.11614 Sec. 9):

- **Scenario** ``xi_i = ((tau_{i,k}, L_{i,k}, p_{i,k})_{k=1}^{K_i}, zeta_i)``:
  an exogenous regime *plan* — a stored sequence of regime durations (in
  market-order events) and buy-MO probabilities, plus the seed used to
  generate it. ``tau`` is retained for reproducibility and is not needed to
  evaluate the schedule; the plan is replayed deterministically on the MO
  clock by :class:`~quant_fund.microstructure.zi_lob_simulator.ScenarioRegimeFlow`.
- **Reference generator ``P_0``**: a scenario-level equal-weight mixture of
  two families — ``random_persistence`` (tau_k ~ Uniform{15,30,60,120,240},
  L_k ~ Exp(1/tau_k), p_k ~ Uniform[0.20, 0.80]) and ``correlated_direction``
  (same (tau, L) mechanism but p_k = 0.5 + s_k * I_k, sign s_k retained across
  regime boundaries with probability ``RHO_SIDE = 0.85``,
  I_k ~ Uniform[0.00, 0.30]). The pool is initialized with a deterministic
  half/half family split; refresh replacements are i.i.d. draws from P_0.
- **Arm score**: ``ell_i(theta) = -H_T(theta; xi_i)`` where ``H_T`` is the
  *undiscounted* terminal penalized PnL — terminal cash + mark-to-market minus
  the sum over decisions of the same quadratic + inventory-wall penalties as
  the per-decision reward (Eq. 16). Difficulty is an EWMA of realized losses
  (``eta_ewma = 0.05``); non-selected arms are unchanged.
- **Sampling**: standardize difficulties ``nu_i = (d_i - d_bar)/(s_d +
  eps_nu)`` (zero vector when s_d is numerically zero), softmax
  ``w_tilde_i ∝ exp(beta_sb * nu_i)``, mix with the uniform:
  ``w_i = (1 - eps) * w_tilde_i + eps / M``, cap each arm at ``w_max = 0.05``
  redistributing the excess, so no arm is ever starved nor dominant.
- **Schedules**: ``beta_sb`` rises linearly 0.00 -> 0.50 and ``eps`` falls
  linearly 1.00 -> 0.50 over the first half of fine-tuning; every
  ``refresh_every`` bandit updates the easiest ``refresh_frac`` (10%) of the
  pool is regenerated from P_0 with new draws seated at the current
  pool-average difficulty.
- The TD update itself is unchanged — the bandit only re-weights *which*
  scenario the next episode replays; policy parameters persist across
  episodes (a warm start from a regime-trained checkpoint is supported by
  simply passing that agent).

Honesty (AGENTS.md contract): seeded SYNTHETIC flows only — no real order
flow, no headline performance ratios; ``sim_internal_*`` diagnostics only,
never market evidence; no live-trading claim.
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
    RegimeState,
    ScenarioRegimeFlow,
    ZILobConfig,
)

SCENARIO_BANDIT_REVISION = "scenario_bandit.v1"

#: Paper Sec. 9 constants (arXiv:2609.11614).
TAU_CHOICES: tuple[float, ...] = (15.0, 30.0, 60.0, 120.0, 240.0)
P_BUY_LO = 0.20
P_BUY_HI = 0.80
CORR_SIDE_PERSIST = 0.85
CORR_IMBALANCE_MAX = 0.30
P_BUY_CENTER = 0.50
POOL_SIZE = 256
DIFFICULTY_EWMA = 0.05
BETA_SB_FINAL = 0.50
EPSILON_FINAL = 0.50
W_MAX = 0.05
REFRESH_EVERY = 500
REFRESH_FRAC = 0.10
EPS_NU = 1e-12

FAMILY_RANDOM_PERSISTENCE = "random_persistence"
FAMILY_CORRELATED_DIRECTION = "correlated_direction"
SCENARIO_FAMILIES = (FAMILY_RANDOM_PERSISTENCE, FAMILY_CORRELATED_DIRECTION)


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
        raise ValueError(f"{name} must lie in [0, 1], got {x!r}")
    return v


def _int_at_least(x: int, floor: int, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < floor:
        raise ValueError(f"{name} must be an int >= {floor}, got {x!r}")
    return x


def _seed_int(x: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"seed must be an int, got {x!r}")
    return x


# ---------------------------------------------------------------------------
# Regime plans (paper's xi = ((tau_k, L_k, p_k), zeta))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeLeg:
    """One leg of a stored scenario: persistence scale tau (reproducibility
    only — the paper keeps it in the tuple but does not evaluate it), the
    leg's duration in market-order events, and its buy-MO probability."""

    tau: float
    length_mo: int
    p_buy: float
    intensity_mult: float = 1.0

    def __post_init__(self) -> None:
        _pos_finite(self.tau, "tau")
        _int_at_least(self.length_mo, 1, "length_mo")
        _prob(self.p_buy, "p_buy")
        _pos_finite(self.intensity_mult, "intensity_mult")


@dataclass(frozen=True)
class ScenarioPlan:
    """A stored exogenous regime scenario (one bandit arm)."""

    legs: tuple[RegimeLeg, ...]
    seed: int
    family: str

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("ScenarioPlan requires at least one leg")
        for i, leg in enumerate(self.legs):
            if not isinstance(leg, RegimeLeg):
                raise TypeError(f"legs[{i}] must be a RegimeLeg")
        _seed_int(self.seed)
        if self.family not in SCENARIO_FAMILIES:
            raise ValueError(f"family must be one of {SCENARIO_FAMILIES}, got {self.family!r}")

    @property
    def total_mo(self) -> int:
        return sum(leg.length_mo for leg in self.legs)

    def to_flow(self) -> ScenarioRegimeFlow:
        """Materialize the plan as a deterministic MO-clock flow."""
        return ScenarioRegimeFlow(
            tuple(
                (
                    RegimeState(f"{self.family}:leg{k}", leg.intensity_mult, leg.p_buy),
                    leg.length_mo,
                )
                for k, leg in enumerate(self.legs)
            )
        )


def expected_mo_count(config: ZILobConfig, horizon: float) -> int:
    """Expected market-order count of an episode, M_hat = 2*mu*horizon.

    Both Poisson MO sides have total intensity ``2 * config.mu`` (buy prob
    ``p_buy`` splits it); regime intensity multipliers are >= 1-free at
    generation time so this is the paper's sizing quantity. Fail-closed on a
    degenerate horizon (no MOs expected -> cannot build a plan).
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    h = _pos_finite(horizon, "horizon")
    m = 2.0 * float(config.mu) * h
    if m < 1.0:
        raise ValueError(
            f"expected MO count {m:.3f} < 1 for horizon {horizon!r} — "
            "scenario plans cannot be sized"
        )
    return int(math.ceil(m))


def _sample_random_persistence_legs(rng: np.random.Generator, mo_horizon: int) -> list[RegimeLeg]:
    legs: list[RegimeLeg] = []
    total = 0
    while total < mo_horizon:
        tau = float(rng.choice(TAU_CHOICES))
        length = max(1, int(math.ceil(float(rng.exponential(tau)))))
        p_buy = float(rng.uniform(P_BUY_LO, P_BUY_HI))
        legs.append(RegimeLeg(tau=tau, length_mo=length, p_buy=p_buy))
        total += length
    return legs


def _sample_correlated_direction_legs(rng: np.random.Generator, mo_horizon: int) -> list[RegimeLeg]:
    legs: list[RegimeLeg] = []
    total = 0
    side = -1.0 if float(rng.random()) < 0.5 else 1.0
    while total < mo_horizon:
        if legs and float(rng.random()) >= CORR_SIDE_PERSIST:
            side = -side
        tau = float(rng.choice(TAU_CHOICES))
        length = max(1, int(math.ceil(float(rng.exponential(tau)))))
        imbalance = float(rng.uniform(0.0, CORR_IMBALANCE_MAX))
        p_buy = P_BUY_CENTER + side * imbalance
        legs.append(RegimeLeg(tau=tau, length_mo=length, p_buy=p_buy))
        total += length
    return legs


def sample_scenario(
    family: str,
    *,
    seed: int,
    mo_horizon: int,
) -> ScenarioPlan:
    """Draw one scenario from a family at the episode MO horizon."""
    _seed_int(seed)
    _int_at_least(mo_horizon, 1, "mo_horizon")
    rng = np.random.default_rng(seed)
    if family == FAMILY_RANDOM_PERSISTENCE:
        legs = _sample_random_persistence_legs(rng, mo_horizon)
    elif family == FAMILY_CORRELATED_DIRECTION:
        legs = _sample_correlated_direction_legs(rng, mo_horizon)
    else:
        raise ValueError(f"family must be one of {SCENARIO_FAMILIES}, got {family!r}")
    return ScenarioPlan(legs=tuple(legs), seed=seed, family=family)


def sample_p0(*, seed: int, mo_horizon: int) -> ScenarioPlan:
    """Draw a scenario from the reference generator P_0 (equal-weight
    scenario-level mixture of the two families)."""
    _seed_int(seed)
    rng = np.random.default_rng(seed)
    family = FAMILY_RANDOM_PERSISTENCE if float(rng.random()) < 0.5 else FAMILY_CORRELATED_DIRECTION
    plan_seed = int(rng.integers(0, 2**63 - 1))
    return sample_scenario(family, seed=plan_seed, mo_horizon=mo_horizon)


# ---------------------------------------------------------------------------
# Scenario pool + difficulty-tracking bandit
# ---------------------------------------------------------------------------


class ScenarioPool:
    """Finite pool of exogenous regime scenarios (bandit arms).

    Initialized with a deterministic half/half family split (128/128 at the
    paper's M=256); ``refresh_easiest`` replaces the lowest-difficulty
    fraction with i.i.d. draws from P_0, each seated at the current
    pool-average difficulty so new arms are neutral until their first rollout.
    """

    def __init__(
        self,
        plans: list[ScenarioPlan],
    ) -> None:
        if not plans:
            raise ValueError("ScenarioPool requires at least one plan")
        for i, p in enumerate(plans):
            if not isinstance(p, ScenarioPlan):
                raise TypeError(f"plans[{i}] must be a ScenarioPlan")
        self.plans = list(plans)

    @classmethod
    def initial(
        cls,
        *,
        size: int = POOL_SIZE,
        seed: int = 0,
        mo_horizon: int,
    ) -> ScenarioPool:
        """Paper initialization: deterministic half/half family split."""
        _int_at_least(size, 2, "size")
        _seed_int(seed)
        rng = np.random.default_rng(seed)
        plans: list[ScenarioPlan] = []
        per_family = size // len(SCENARIO_FAMILIES)
        for family in SCENARIO_FAMILIES:
            n_i = per_family + (1 if family == SCENARIO_FAMILIES[0] and size % 2 else 0)
            for _ in range(n_i):
                plan_seed = int(rng.integers(0, 2**63 - 1))
                plans.append(sample_scenario(family, seed=plan_seed, mo_horizon=mo_horizon))
        return cls(plans)

    @property
    def size(self) -> int:
        return len(self.plans)

    def refresh_easiest(
        self,
        difficulty: NDArray[np.float64],
        *,
        frac: float = REFRESH_FRAC,
        rng: np.random.Generator,
        mo_horizon: int,
    ) -> list[int]:
        """Replace the easiest ``frac`` of arms with fresh P_0 draws.

        Returns the replaced arm indices; the caller seats each new arm's
        difficulty at the current pool average (paper: new scenarios are
        neutral until first rollout)."""
        if len(difficulty) != self.size:
            raise ValueError(f"difficulty vector length {len(difficulty)} != pool size {self.size}")
        f = float(frac)
        if not math.isfinite(f) or f <= 0.0 or f > 1.0:
            raise ValueError(f"frac must lie in (0, 1], got {frac!r}")
        n_rep = max(1, int(math.floor(f * self.size)))
        easiest = np.argsort(difficulty, kind="stable")[:n_rep]
        for idx in easiest:
            plan_seed = int(rng.integers(0, 2**63 - 1))
            self.plans[int(idx)] = sample_p0(seed=plan_seed, mo_horizon=mo_horizon)
        return [int(i) for i in easiest]


class ScenarioBandit:
    """Difficulty-tracking softmax bandit over a :class:`ScenarioPool`.

    Paper Sec. 9: arm difficulty ``d_i`` is an EWMA (``eta = 0.05``) of the
    realized scenario loss ``ell_i = -H_T``; sampling probabilities are the
    standardized-difficulty softmax mixed with the uniform distribution and
    capped at ``w_max`` with redistribution. ``beta_sb`` warms up 0 -> 0.50 and
    ``eps`` cools 1.00 -> 0.50 linearly over the first half of the configured
    update budget; every ``refresh_every`` updates the pool's easiest
    ``refresh_frac`` is regenerated. Fully seeded/deterministic.
    """

    def __init__(
        self,
        pool: ScenarioPool,
        *,
        total_updates: int,
        seed: int = 0,
        eta: float = DIFFICULTY_EWMA,
        beta_final: float = BETA_SB_FINAL,
        epsilon_final: float = EPSILON_FINAL,
        w_max: float = W_MAX,
        refresh_every: int = REFRESH_EVERY,
        refresh_frac: float = REFRESH_FRAC,
    ) -> None:
        if not isinstance(pool, ScenarioPool):
            raise TypeError("pool must be a ScenarioPool")
        self._pool = pool
        self._total = _int_at_least(total_updates, 1, "total_updates")
        _seed_int(seed)
        self._eta = _pos_finite(eta, "eta")
        self._beta_final = _nonneg_finite(beta_final, "beta_final")
        self._eps_final = _prob(epsilon_final, "epsilon_final")
        self._w_max = _prob(w_max, "w_max")
        if self._w_max * pool.size < 1.0:
            raise ValueError(
                f"w_max {w_max!r} too small for pool size {pool.size}: "
                "cannot renormalize after capping"
            )
        self._refresh_every = _int_at_least(refresh_every, 1, "refresh_every")
        self._refresh_frac = float(refresh_frac)
        if not math.isfinite(self._refresh_frac) or not 0.0 < self._refresh_frac <= 1.0:
            raise ValueError(f"refresh_frac must lie in (0, 1], got {refresh_frac!r}")
        self.difficulty = np.zeros(pool.size, dtype=np.float64)
        self.n_updates = 0
        self.n_refreshes = 0
        self._last_refresh_at = 0
        self.refresh_history: list[tuple[int, list[int]]] = []
        self.selection_history: list[int] = []
        self.loss_history: list[tuple[int, float]] = []

    @property
    def pool(self) -> ScenarioPool:
        return self._pool

    @property
    def n_arms(self) -> int:
        return self._pool.size

    def schedule(self, update_index: int) -> tuple[float, float]:
        """(beta_sb, eps) at bandit update ``update_index`` (0-based)."""
        u = _int_at_least(update_index, 0, "update_index")
        half = max(self._total / 2.0, 1.0)
        frac = min(u / half, 1.0)
        beta = self._beta_final * frac
        eps = 1.0 + (self._eps_final - 1.0) * frac
        return beta, eps

    def probabilities(self, update_index: int | None = None) -> NDArray[np.float64]:
        """Sampling distribution over arms at the current difficulty vector."""
        u = self.n_updates if update_index is None else update_index
        beta, eps = self.schedule(u)
        d = self.difficulty
        d_bar = float(d.mean())
        s_d = float(d.std())
        if s_d <= EPS_NU:
            nu = np.zeros_like(d)
        else:
            nu = (d - d_bar) / (s_d + EPS_NU)
        logits = beta * nu
        logits -= float(logits.max())
        w_tilde = np.exp(logits)
        w_tilde /= float(w_tilde.sum())
        w = (1.0 - eps) * w_tilde + eps / self.n_arms
        return _cap_and_renormalize(w, self._w_max)

    def select(self, rng: np.random.Generator | None = None) -> int:
        """Draw an arm index under the current probabilities."""
        r = rng if rng is not None else np.random.default_rng(self.n_updates + 1)
        w = self.probabilities()
        arm = int(r.choice(self.n_arms, p=w))
        self.selection_history.append(arm)
        return arm

    def observe(self, arm: int, loss: float) -> float:
        """EWMA difficulty update for the selected arm (paper Eq. 9.2)."""
        i = int(arm)
        if not 0 <= i < self.n_arms:
            raise ValueError(f"arm must lie in [0, {self.n_arms}), got {arm!r}")
        ell = float(loss)
        if not math.isfinite(ell):
            raise ValueError(f"loss must be finite, got {loss!r}")
        self.difficulty[i] = (1.0 - self._eta) * self.difficulty[i] + self._eta * ell
        self.n_updates += 1
        self.loss_history.append((i, ell))
        return float(self.difficulty[i])

    def maybe_refresh(
        self,
        *,
        rng: np.random.Generator,
        mo_horizon: int,
    ) -> list[int]:
        """Every ``refresh_every`` updates, regenerate the easiest arms."""
        if (
            self.n_updates == 0
            or self.n_updates % self._refresh_every != 0
            or self.n_updates == self._last_refresh_at
        ):
            return []
        self._last_refresh_at = self.n_updates
        d_bar = float(self.difficulty.mean())
        idxs = self._pool.refresh_easiest(
            self.difficulty, frac=self._refresh_frac, rng=rng, mo_horizon=mo_horizon
        )
        for i in idxs:
            self.difficulty[i] = d_bar
        self.n_refreshes += 1
        self.refresh_history.append((self.n_updates, idxs))
        return idxs


def _cap_and_renormalize(w: NDArray[np.float64], w_max: float) -> NDArray[np.float64]:
    """Cap each arm at ``w_max``; redistribute the excess over uncapped arms
    in proportion to their remaining headroom (iterative waterfilling —
    converges since the capped set is monotone)."""
    out = np.asarray(w, dtype=np.float64).copy()
    for _ in range(len(out) + 1):
        over = out > w_max
        if not bool(over.any()):
            break
        excess = float((out[over] - w_max).sum())
        out[over] = w_max
        under = ~over
        headroom = np.maximum(w_max - out[under], 0.0)
        room = float(headroom.sum())
        if room <= 0.0:
            break
        out[under] += excess * headroom / room
    total = float(out.sum())
    if total <= 0.0:
        return np.full_like(out, 1.0 / len(out))
    return out / total


# ---------------------------------------------------------------------------
# Fine-tuning loop (Algorithm C driver)
# ---------------------------------------------------------------------------


def run_scenario_finetune(
    agent: C51MarketMaker,
    *,
    config: ZILobConfig,
    horizon: float,
    n_episodes: int,
    pool_size: int = POOL_SIZE,
    seed: int = 0,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    learn_every: int = 1,
    refresh_every: int = REFRESH_EVERY,
    refresh_frac: float = REFRESH_FRAC,
    w_max: float = W_MAX,
) -> dict[str, Any]:
    """Algorithm C: scenario-bandit robust fine-tuning of a trained agent.

    Each episode: the bandit draws an arm under its difficulty-weighted
    probabilities, the stored scenario is replayed as a deterministic
    MO-clock flow, the agent trains through the episode (ordinary C51 TD
    updates — unchanged), and the arm's difficulty absorbs the undiscounted
    terminal penalized PnL ``H_T = PnL_T - sum_t [phi q^2 + phi (|q|-q_w)_+^2]``
    via EWMA. The bandit thus redirects training effort toward the scenario
    lower tail discovered by the *current* policy.

    Deterministic given (agent, config, seed): episode simulator seeds are
    ``seed + episode`` and scenario draws are seeded off the pool seed.

    Returns a SYNTHETIC diagnostic bundle — ``sim_internal_*`` keys only,
    never headline metrics, never market evidence.
    """
    if not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    n_ep = _int_at_least(n_episodes, 1, "n_episodes")
    _seed_int(seed)
    _int_at_least(pool_size, 2, "pool_size")
    h = _pos_finite(horizon, "horizon")
    _pos_finite(decision_interval, "decision_interval")
    _pos_finite(sample_interval, "sample_interval")
    _nonneg_finite(reward_phi, "reward_phi")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")

    mo_horizon = expected_mo_count(config, h)
    pool = ScenarioPool.initial(size=pool_size, seed=seed, mo_horizon=mo_horizon)
    # Feasibility floor: the paper's 0.05 cap only admits pools >= 20 arms;
    # smaller test pools lift the cap to the uniform weight (a no-op cap).
    w_max_eff = max(float(w_max), 1.0 / pool.size)
    bandit = ScenarioBandit(
        pool,
        total_updates=n_ep,
        seed=seed,
        refresh_every=refresh_every,
        refresh_frac=refresh_frac,
        w_max=w_max_eff,
    )
    rng = np.random.default_rng(seed + 1)

    episodes: list[dict[str, Any]] = []
    all_losses: list[float] = []
    family_counts = {f: 0 for f in SCENARIO_FAMILIES}
    arm_pulls = np.zeros(pool.size, dtype=np.int64)
    for u in range(n_ep):
        beta, eps = bandit.schedule(u)
        arm = bandit.select(rng)
        plan = pool.plans[arm]
        family_counts[plan.family] += 1
        arm_pulls[arm] += 1
        cfg_i = replace(config, seed=seed + 2 + u)
        agent.begin_episode()
        bundle = run_rl_mm_session(
            config=cfg_i,
            horizon=h,
            agent=agent,
            training=True,
            learn_every=learn_every,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            inventory_cap=agent.spec.inventory_cap,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            flow=plan.to_flow(),
        )
        pnl_t = float(bundle["sim_internal_mtm_pnl_final"])
        penalty_sum = float(bundle["sim_internal_penalty_sum"])
        h_t = pnl_t - penalty_sum
        d_new = bandit.observe(arm, -h_t)
        refreshed = bandit.maybe_refresh(rng=rng, mo_horizon=mo_horizon)
        all_losses.extend(bundle["loss_curve"])
        episodes.append(
            {
                "episode": u,
                "arm": arm,
                "family": plan.family,
                "scenario_seed": plan.seed,
                "n_legs": len(plan.legs),
                "beta_sb": float(beta),
                "eps": float(eps),
                "sim_internal_terminal_score": float(h_t),
                "sim_internal_mtm_pnl_final": pnl_t,
                "sim_internal_penalty_sum": penalty_sum,
                "difficulty_after": float(d_new),
                "n_refreshed": len(refreshed),
                "n_fills": bundle["n_fills"],
                "max_abs_inventory": bundle["max_abs_inventory"],
                "session_completed": bundle["session_completed"],
            }
        )

    d = bandit.difficulty
    w_final = bandit.probabilities()
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "bandit_revision": SCENARIO_BANDIT_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "scenario_bandit_finetune",
        "n_episodes": n_ep,
        "pool_size": pool.size,
        "mo_horizon": mo_horizon,
        "episodes": episodes,
        "family_pulls": {k: int(v) for k, v in family_counts.items()},
        "n_arms_pulled": int((arm_pulls > 0).sum()),
        "max_pull_share": float(arm_pulls.max() / max(n_ep, 1)),
        "difficulty_mean": float(d.mean()),
        "difficulty_std": float(d.std()),
        "difficulty_max": float(d.max()),
        "difficulty_min": float(d.min()),
        "weight_max_final": float(w_final.max()),
        "weight_entropy_final": float(-(w_final * np.log(np.clip(w_final, 1e-300, None))).sum()),
        "n_refreshes": bandit.n_refreshes,
        "loss_curve": all_losses,
        "n_updates": agent.n_updates,
        "epsilon_final": agent.epsilon,
        "training_budget": {
            "horizon": float(h),
            "n_episodes": n_ep,
            "pool_size": int(pool_size),
            "seed": int(seed),
            "refresh_every": int(refresh_every),
            "refresh_frac": float(refresh_frac),
        },
    }
