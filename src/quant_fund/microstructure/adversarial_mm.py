"""Learned adversary for robust market making (Glielmo-style, Moret & Lillo 2026).

The scenario-bandit lane (:mod:`scenario_bandit`, Algorithm C) reweights a
*static pool* of stored scenarios toward the current policy's lower tail.
This module implements the complementary piece from the same paper: an
adversary that **learns** to hurt the defender — a semi-MDP player acting
at regime boundaries through :class:`AdversarialFlow`.

Game structure (paper Sec. 8-9):

- The adversary acts only at *regime boundaries*: when a leg's MO budget is
  exhausted it observes ``(inventory_norm, prev_p_buy)`` and picks the next
  leg's ``p_buy`` from the 13-point grid :data:`P_BUY_GRID`. Leg *durations*
  stay environmental — drawn by ``duration_sampler`` (Pareto by default,
  matching the heavy-tailed metaorder lengths of Lillo-Mike-Farmer 2005).
- Its reward is the negation of the defender's shaped reward summed over
  the leg's MO span (the zero-sum objective). For a classic ``QuotePolicy``
  defender there is no per-decision reward stream — the adversary then
  receives the sparse terminal reward ``-H_T`` where
  ``H_T = PnL_T - integral of the inventory penalty over the episode``.
- The defender acts greedily (evaluation mode): the adversary is the only
  learner inside :func:`run_adversarial_eval`, matching the paper's
  alternating-training protocol's adversary phase.

Everything here is a SYNTHETIC diagnostic over the ZI-LOB simulator —
``sim_internal_*`` keys only, never a headline metric, never market
evidence, no live-trading claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.microstructure.rl_market_maker import (
    RL_MM_REVISION,
    C51MarketMaker,
    _torch,
    run_rl_mm_session,
)
from quant_fund.microstructure.scenario_bandit import _int_at_least, _pos_finite, _prob, _seed_int
from quant_fund.microstructure.zi_lob_simulator import (
    ZI_LOB_REVISION,
    AdversarialFlow,
    QuotePolicy,
    RegimeState,
    ScenarioRegimeFlow,
    ZILobConfig,
    run_mm_session,
)

ADVERSARIAL_MM_REVISION = "adversarial_mm.v1"

#: 13-point adversary action grid over p_buy in [0.20, 0.80] — symmetric
#: around fair value, bounded away from degenerate extremes (paper Sec. 8).
P_BUY_GRID: tuple[float, ...] = tuple(round(0.20 + 0.05 * i, 2) for i in range(13))
P_BUY_LO, P_BUY_HI = P_BUY_GRID[0], P_BUY_GRID[-1]

#: Clip bound applied to the adversary's normalised-inventory observation —
#: a hard-capped defender can legitimately sit at |q| = cap, and transient
#: accounting overshoot must not leave the feature unbounded.
_INV_OBS_CLIP = 1.5


def make_pareto_duration(
    *,
    alpha: float = 1.5,
    l_min: int = 15,
    l_cap: int = 500,
) -> Callable[[np.random.Generator], int]:
    """Pareto leg-duration sampler over the MO clock.

    ``numpy.random.pareto(alpha)`` returns a Lomax ``x >= 0``; the leg
    duration is ``l_min * (1 + x)`` capped at ``l_cap``, i.e. a discrete
    power-law tail with scale ``l_min`` — the heavy-tailed metaorder-length
    convention of the paper's adversary.
    """
    a = float(alpha)
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError(f"alpha must be positive and finite, got {alpha!r}")
    lo = _int_at_least(l_min, 1, "l_min")
    cap = _int_at_least(l_cap, lo, "l_cap")

    def _draw(rng: np.random.Generator) -> int:
        x = float(rng.pareto(a))
        return min(cap, max(1, int(math.ceil(lo * (1.0 + x)))))

    return _draw


def make_uniform_duration(length: int) -> Callable[[np.random.Generator], int]:
    """Constant leg length — the degenerate adversary-duration choice used
    in tests and grid sweeps."""
    n = _int_at_least(length, 1, "length")

    def _draw(rng: np.random.Generator) -> int:
        del rng
        return n

    return _draw


# ---------------------------------------------------------------------------
# Adversary agents
# ---------------------------------------------------------------------------


class AdversaryAgent:
    """Epsilon-greedy DQN over the 13-point ``p_buy`` grid.

    Observation ``(inventory / inv_limit, prev_p_buy)``; reward is the
    negation of the defender's shaped reward over the leg — a zero-sum
    semi-MDP (actions at boundaries only). Torch is imported lazily through
    the shared ``nn``-extra guard; numpy and torch are seeded so identical
    call sequences are deterministic.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        batch_size: int = 32,
        replay_capacity: int = 10_000,
        target_update_every: int = 20,
        inv_limit: float = 8.0,
        seed: int = 0,
    ) -> None:
        torch = _torch()
        _seed_int(seed)
        h = _int_at_least(hidden, 1, "hidden")
        g = float(gamma)
        if not math.isfinite(g) or not 0.0 <= g <= 1.0:
            raise ValueError(f"gamma must lie in [0, 1], got {gamma!r}")
        lr_f = float(lr)
        if not math.isfinite(lr_f) or lr_f <= 0.0:
            raise ValueError(f"lr must be positive and finite, got {lr!r}")
        for name, v in (("eps_start", eps_start), ("eps_end", eps_end)):
            if not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {v!r}")
        if not 0.0 < float(eps_decay) <= 1.0:
            raise ValueError(f"eps_decay must lie in (0, 1], got {eps_decay!r}")
        if float(eps_end) > float(eps_start):
            raise ValueError("eps_end must be <= eps_start")
        self._batch = _int_at_least(batch_size, 1, "batch_size")
        cap_r = _int_at_least(replay_capacity, self._batch, "replay_capacity")
        self._target_every = _int_at_least(target_update_every, 1, "target_update_every")
        self._inv_limit = float(inv_limit)
        if not math.isfinite(self._inv_limit) or self._inv_limit <= 0.0:
            raise ValueError(f"inv_limit must be positive and finite, got {inv_limit!r}")
        self._gamma = g
        self._torch = torch
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        self._rng = np.random.default_rng(seed + 31_415)
        n_actions = len(P_BUY_GRID)
        self._q_net = torch.nn.Sequential(
            torch.nn.Linear(2, h), torch.nn.ReLU(), torch.nn.Linear(h, n_actions)
        )
        self._q_target = torch.nn.Sequential(
            torch.nn.Linear(2, h), torch.nn.ReLU(), torch.nn.Linear(h, n_actions)
        )
        self.update_target()
        self._opt = torch.optim.AdamW(self._q_net.parameters(), lr=lr_f)
        self._buffer_states: list[tuple[float, float]] = []
        self._buffer_actions: list[int] = []
        self._buffer_rewards: list[float] = []
        self._buffer_next: list[tuple[float, float]] = []
        self._buffer_done: list[bool] = []
        self._capacity = cap_r
        self.epsilon = float(eps_start)
        self._eps_end = float(eps_end)
        self._eps_decay = float(eps_decay)
        self._n_updates = 0
        self._n_episodes = 0
        self.losses: list[float] = []

    # -- public API ----------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return len(P_BUY_GRID)

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def inv_limit(self) -> float:
        return self._inv_limit

    @property
    def buffer_size(self) -> int:
        return len(self._buffer_rewards)

    def obs(self, inventory: float, prev_p_buy: float) -> tuple[float, float]:
        """Normalised adversary observation (clipped inventory feature)."""
        inv = float(inventory) / self._inv_limit
        return (min(max(inv, -_INV_OBS_CLIP), _INV_OBS_CLIP), float(prev_p_buy))

    def select(self, inventory: float, prev_p_buy: float, *, greedy: bool = False) -> int:
        """Action index into :data:`P_BUY_GRID` for the given raw observation."""
        _prob(prev_p_buy, "prev_p_buy")
        obs = self.obs(inventory, prev_p_buy)
        if not greedy and float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        torch = self._torch
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self._q_net(t)
        return int(q.argmax(dim=1).item())

    def picker(self, *, greedy: bool = False) -> Callable[[float, float], float]:
        """Flow-pluggable callback: raw inventory + prev p_buy -> p_buy."""

        def _pick(inventory: float, prev_p_buy: float) -> float:
            return P_BUY_GRID[self.select(inventory, prev_p_buy, greedy=greedy)]

        return _pick

    def store_transition(
        self,
        obs: tuple[float, float],
        action: int,
        reward: float,
        next_obs: tuple[float, float],
        done: bool,
    ) -> None:
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action {action} out of range {self.n_actions}")
        r = float(reward)
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        if len(self._buffer_rewards) >= self._capacity:
            for buf in (
                self._buffer_states,
                self._buffer_actions,
                self._buffer_rewards,
                self._buffer_next,
                self._buffer_done,
            ):
                buf.pop(0)
        self._buffer_states.append((float(obs[0]), float(obs[1])))
        self._buffer_actions.append(int(action))
        self._buffer_rewards.append(r)
        self._buffer_next.append((float(next_obs[0]), float(next_obs[1])))
        self._buffer_done.append(bool(done))

    def learn(self) -> float | None:
        """One AdamW step on a uniform replay mini-batch; None if underfilled."""
        n = len(self._buffer_rewards)
        if n < self._batch:
            return None
        torch = self._torch
        idx = self._rng.choice(n, size=self._batch, replace=False)
        s = torch.as_tensor([self._buffer_states[i] for i in idx], dtype=torch.float32)
        a = torch.as_tensor([self._buffer_actions[i] for i in idx], dtype=torch.int64)
        r = torch.as_tensor([self._buffer_rewards[i] for i in idx], dtype=torch.float32)
        sp = torch.as_tensor([self._buffer_next[i] for i in idx], dtype=torch.float32)
        d = torch.as_tensor([self._buffer_done[i] for i in idx], dtype=torch.float32)
        q = self._q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self._q_target(sp).max(dim=1).values
            target = r + (1.0 - d) * self._gamma * q_next
        loss = torch.nn.functional.mse_loss(q, target)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._n_updates += 1
        if self._n_updates % self._target_every == 0:
            self.update_target()
        val = float(loss.detach().item())
        self.losses.append(val)
        return val

    def update_target(self) -> None:
        self._q_target.load_state_dict(self._q_net.state_dict())

    def end_episode(self) -> None:
        self._n_episodes += 1
        self.epsilon = max(self._eps_end, self.epsilon * self._eps_decay)


class RandomAdversary:
    """Uniform-random grid picker — the torch-free adversary baseline."""

    def __init__(self, *, seed: int = 0) -> None:
        _seed_int(seed)
        self._rng = np.random.default_rng(seed + 27_182)

    @property
    def n_actions(self) -> int:
        return len(P_BUY_GRID)

    def select(self, inventory: float, prev_p_buy: float, *, greedy: bool = False) -> int:
        del inventory, greedy
        _prob(prev_p_buy, "prev_p_buy")
        return int(self._rng.integers(self.n_actions))

    def picker(self, *, greedy: bool = False) -> Callable[[float, float], float]:
        del greedy

        def _pick(inventory: float, prev_p_buy: float) -> float:
            return P_BUY_GRID[self.select(inventory, prev_p_buy)]

        return _pick


# ---------------------------------------------------------------------------
# Episodes + evaluation
# ---------------------------------------------------------------------------


def _adversary_transitions(
    flow: AdversarialFlow,
    *,
    inv_limit: float,
    final_inventory: float,
) -> list[tuple[tuple[float, float], float, float, tuple[float, float], bool]]:
    """Boundary transitions implied by the flow's log, without rewards.

    Returns ``(obs, p_buy, mo_at_pick, next_obs, done)`` per resolved leg
    after the first (the seed leg has no preceding action). ``mo_at_pick``
    is the MO index where the boundary fired — the reward attribution key.
    """
    out: list[tuple[tuple[float, float], float, float, tuple[float, float], bool]] = []
    log = flow.boundary_log
    for k, (n_mo, inv, prev_p, chosen) in enumerate(log):
        obs = (min(max(inv / inv_limit, -_INV_OBS_CLIP), _INV_OBS_CLIP), prev_p)
        if k + 1 < len(log):
            _n2, inv2, _pp, _c2 = log[k + 1]
            next_obs = (
                min(max(inv2 / inv_limit, -_INV_OBS_CLIP), _INV_OBS_CLIP),
                chosen,
            )
            out.append((obs, chosen, float(n_mo), next_obs, False))
        else:
            term = (
                min(max(final_inventory / inv_limit, -_INV_OBS_CLIP), _INV_OBS_CLIP),
                chosen,
            )
            out.append((obs, chosen, float(n_mo), term, True))
    return out


def adversarial_episode(
    *,
    config: ZILobConfig,
    horizon: float,
    picker: Callable[[float, float], float],
    duration_sampler: Callable[[np.random.Generator], int],
    seed: int,
    defender_agent: C51MarketMaker | None = None,
    defender_policy: QuotePolicy | None = None,
    inventory_cap: int | None = None,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    first_p_buy: float = 0.5,
    max_legs: int = 10_000,
) -> tuple[dict[str, Any], AdversarialFlow]:
    """Run one session under an adversarially-steered flow.

    Exactly one of ``defender_agent`` (C51, evaluated greedily — the
    adversary alone learns) / ``defender_policy`` (classic quotes) must be
    given. Returns the session bundle plus the resolved flow (whose
    ``boundary_log`` carries the adversary's transition context).
    """
    if (defender_agent is None) == (defender_policy is None):
        raise ValueError("exactly one of defender_agent / defender_policy must be provided")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    h = _pos_finite(horizon, "horizon")
    _seed_int(seed)
    flow = AdversarialFlow(
        picker=picker,
        duration_sampler=duration_sampler,
        seed=seed,
        first_p_buy=first_p_buy,
        max_legs=max_legs,
    )
    if defender_agent is not None:
        if not isinstance(defender_agent, C51MarketMaker):
            raise TypeError("defender_agent must be a C51MarketMaker")
        bundle = run_rl_mm_session(
            config=config,
            horizon=h,
            agent=defender_agent,
            training=False,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            inventory_cap=inventory_cap,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            flow=flow,
        )
    else:
        if not callable(defender_policy):
            raise TypeError("defender_policy must be callable")
        pol = defender_policy
        if not (pol is not None):
            raise ValueError("pol is not None")
        bundle = run_mm_session(
            config=config,
            policy=pol,
            horizon=h,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            inventory_cap=inventory_cap,
            flow=flow,
        )
    return bundle, flow


def _episode_penalty(inv_path: Sequence[float], *, phi: float, wall: float) -> float:
    """``sum_t phi * q^2 + phi * (|q| - wall)_+^2`` over the recorded path —
    the same penalty shape as the agent-mode reward, integrated on the
    decision grid for both defender kinds."""
    q = np.asarray(inv_path, dtype=np.float64)
    return float(
        phi * np.square(q).sum() + phi * np.square(np.maximum(np.abs(q) - wall, 0.0)).sum()
    )


def run_adversarial_eval(
    *,
    config: ZILobConfig,
    horizon: float,
    n_episodes: int,
    adversary: AdversaryAgent,
    defender_agent: C51MarketMaker | None = None,
    defender_policy: QuotePolicy | None = None,
    inventory_cap: int | None = None,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    duration_sampler: Callable[[np.random.Generator], int] | None = None,
    first_p_buy: float = 0.5,
    train_adversary: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Adversary-phase evaluation: the defender acts greedily while the
    adversary steers flow legs and (optionally) learns.

    Reward attribution: for a C51 defender the adversary's leg reward is the
    negated sum of the defender's per-decision shaped rewards whose MO
    timestamp falls inside the leg (``sim_internal_reward_mo_index``);
    ``adversary_return`` is the negated sum over adversary-chosen legs —
    defender rewards under the *seed* leg (before the first boundary, where
    no adversary action exists) are unattributed and excluded. For a
    classic-policy defender there is no reward stream — the
    adversary receives the sparse terminal reward ``-H_T`` where
    ``H_T = PnL_T - penalty`` is computed post-hoc from the inventory path.

    Returns a SYNTHETIC diagnostic bundle (``kind="adversarial_mm_eval"``).
    """
    if not isinstance(adversary, AdversaryAgent):
        raise TypeError("adversary must be an AdversaryAgent")
    if (defender_agent is None) == (defender_policy is None):
        raise ValueError("exactly one of defender_agent / defender_policy must be provided")
    if defender_agent is not None and not isinstance(defender_agent, C51MarketMaker):
        raise TypeError("defender_agent must be a C51MarketMaker")
    if defender_policy is not None and not callable(defender_policy):
        raise TypeError("defender_policy must be callable")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    n_ep = _int_at_least(n_episodes, 1, "n_episodes")
    _seed_int(seed)
    h = _pos_finite(horizon, "horizon")
    phi = float(reward_phi)
    if not math.isfinite(phi) or phi < 0.0:
        raise ValueError(f"reward_phi must be non-negative and finite, got {reward_phi!r}")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    sampler = duration_sampler or make_pareto_duration()
    cap = defender_agent.spec.inventory_cap if defender_agent is not None else inventory_cap
    inv_limit = float(cap) if cap is not None else adversary.inv_limit
    wall = inv_limit * fw

    episodes: list[dict[str, Any]] = []
    for u in range(n_ep):
        cfg_i = replace(config, seed=seed + 7 + u)
        bundle, flow = adversarial_episode(
            config=cfg_i,
            horizon=h,
            picker=adversary.picker(),
            duration_sampler=sampler,
            seed=seed + 10_000 + u,
            defender_agent=defender_agent,
            defender_policy=defender_policy,
            inventory_cap=inventory_cap,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            first_p_buy=first_p_buy,
        )
        penalty = (
            _episode_penalty(bundle["inventory_path"], phi=phi, wall=wall)
            if defender_agent is None
            else float(bundle["sim_internal_penalty_sum"])
        )
        h_t = float(bundle["sim_internal_mtm_pnl_final"]) - penalty

        transitions = _adversary_transitions(
            flow,
            inv_limit=adversary.inv_limit,
            final_inventory=float(bundle["inventory_final"]),
        )
        rewards = bundle.get("sim_internal_reward_path") or []
        rewards_mo = bundle.get("sim_internal_reward_mo_index") or []
        leg_reward = -h_t
        if rewards and rewards_mo:
            # Dense attribution: leg k covers MOs (bound_k, bound_{k+1}];
            # decision rewards are mapped by the MO count at their close.
            bounds = [t[2] for t in transitions] + [float("inf")]
            neg_rewards = [-float(r) for r in rewards]
            leg_reward_parts: list[float] = []
            for k, _t in enumerate(transitions):
                lo_mo, hi_mo = bounds[k], bounds[k + 1]
                leg_reward_parts.append(
                    sum(
                        r
                        for r, m in zip(neg_rewards, rewards_mo, strict=True)
                        if lo_mo < m <= hi_mo
                    )
                )
        else:
            leg_reward_parts = []

        adv_return = 0.0
        n_boundary_trans = len(transitions)
        for k, (obs, chosen, _mo_at, next_obs, done) in enumerate(transitions):
            if leg_reward_parts:
                r_k = leg_reward_parts[k]
            else:
                r_k = leg_reward if done else 0.0
            adv_return += r_k
            if train_adversary:
                try:
                    action_idx = P_BUY_GRID.index(round(chosen, 2))
                except ValueError:
                    action_idx = min(
                        range(len(P_BUY_GRID)),
                        key=lambda i: abs(P_BUY_GRID[i] - chosen),
                    )
                adversary.store_transition(obs, action_idx, r_k, next_obs, done)
        if train_adversary:
            adversary.learn()
            adversary.end_episode()

        episodes.append(
            {
                "episode": u,
                "n_boundaries": n_boundary_trans,
                "boundary_mo": [t[2] for t in transitions],
                "sim_internal_terminal_score": h_t,
                "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
                "sim_internal_penalty_sum": penalty,
                "adversary_return": adv_return,
                "expected_p_buy": float(flow.expected_p_buy()) if flow.n_mo else float("nan"),
                "chosen_p_buy_mean": float(np.mean([c for _n, _i, _p, c in flow.boundary_log]))
                if flow.boundary_log
                else float("nan"),
                "n_fills": bundle["n_fills"],
                "max_abs_inventory": bundle["max_abs_inventory"],
                "session_completed": bool(bundle.get("session_completed", True)),
            }
        )

    scores = np.asarray([e["sim_internal_terminal_score"] for e in episodes], dtype=np.float64)
    adv = np.asarray([e["adversary_return"] for e in episodes], dtype=np.float64)
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "adversary_revision": ADVERSARIAL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "adversarial_mm_eval",
        "n_episodes": n_ep,
        "defender_kind": "c51_rl" if defender_agent is not None else "classic_policy",
        "train_adversary": bool(train_adversary),
        "episodes": episodes,
        "defender_score_mean": float(scores.mean()),
        "defender_score_min": float(scores.min()),
        "defender_score_std": float(scores.std()),
        "adversary_return_mean": float(adv.mean()),
        "adversary_epsilon_final": adversary.epsilon,
        "adversary_n_updates": adversary.n_updates,
        "adversary_buffer": adversary.buffer_size,
        "budget": {
            "horizon": float(h),
            "n_episodes": n_ep,
            "seed": int(seed),
            "inv_limit": inv_limit,
            "reward_phi": phi,
            "reward_wall_fraction": fw,
        },
    }


def constant_flow_sweep(
    *,
    config: ZILobConfig,
    horizon: float,
    defender_agent: C51MarketMaker | None = None,
    defender_policy: QuotePolicy | None = None,
    inventory_cap: int | None = None,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    p_buy_grid: Sequence[float] = P_BUY_GRID,
    seed: int = 0,
) -> dict[str, Any]:
    """Worst-constant baseline: defender score under each constant ``p_buy``
    from :data:`P_BUY_GRID` — the static-flow reference the learned
    adversary should underperform (lower defender score = stronger attack).

    Deterministic given ``seed`` (each grid point gets ``seed + i``).
    SYNTHETIC bundle (``kind="constant_flow_sweep"``).
    """
    if (defender_agent is None) == (defender_policy is None):
        raise ValueError("exactly one of defender_agent / defender_policy must be provided")
    if defender_agent is not None and not isinstance(defender_agent, C51MarketMaker):
        raise TypeError("defender_agent must be a C51MarketMaker")
    if defender_policy is not None and not callable(defender_policy):
        raise TypeError("defender_policy must be callable")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    if not p_buy_grid:
        raise ValueError("p_buy_grid is empty")
    for p in p_buy_grid:
        _prob(float(p), "p_buy_grid entry")
    _seed_int(seed)
    h = _pos_finite(horizon, "horizon")
    phi = float(reward_phi)
    if not math.isfinite(phi) or phi < 0.0:
        raise ValueError(f"reward_phi must be non-negative and finite, got {reward_phi!r}")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    cap = defender_agent.spec.inventory_cap if defender_agent is not None else inventory_cap
    inv_limit = float(cap) if cap is not None else 8.0
    wall = inv_limit * fw
    mo_horizon = max(1, int(math.ceil(2.0 * float(config.mu) * h)))

    rows: list[dict[str, Any]] = []
    for i, p_buy in enumerate(p_buy_grid):
        cfg_i = replace(config, seed=seed + 101 + i, p_buy=float(p_buy))
        flow = ScenarioRegimeFlow(
            [(RegimeState(f"constant:{p_buy:.2f}", 1.0, float(p_buy)), mo_horizon)]
        )
        if defender_agent is not None:
            bundle = run_rl_mm_session(
                config=cfg_i,
                horizon=h,
                agent=defender_agent,
                training=False,
                decision_interval=decision_interval,
                sample_interval=sample_interval,
                inventory_cap=inventory_cap,
                reward_phi=reward_phi,
                reward_wall_fraction=reward_wall_fraction,
                flow=flow,
            )
            penalty = float(bundle["sim_internal_penalty_sum"])
        else:
            pol = defender_policy
            if not (pol is not None):
                raise ValueError("pol is not None")  # narrowed by the caller validation above
            bundle = run_mm_session(
                config=cfg_i,
                policy=pol,
                horizon=h,
                decision_interval=decision_interval,
                sample_interval=sample_interval,
                inventory_cap=inventory_cap,
                flow=flow,
            )
            penalty = _episode_penalty(bundle["inventory_path"], phi=phi, wall=wall)
        h_t = float(bundle["sim_internal_mtm_pnl_final"]) - penalty
        rows.append(
            {
                "p_buy": float(p_buy),
                "sim_internal_terminal_score": h_t,
                "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
                "sim_internal_penalty_sum": penalty,
                "n_fills": bundle["n_fills"],
                "max_abs_inventory": bundle["max_abs_inventory"],
            }
        )
    worst = min(rows, key=lambda r: float(r["sim_internal_terminal_score"]))
    scores = np.asarray([r["sim_internal_terminal_score"] for r in rows], dtype=np.float64)
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "constant_flow_sweep",
        "defender_kind": "c51_rl" if defender_agent is not None else "classic_policy",
        "rows": rows,
        "worst_p_buy": float(worst["p_buy"]),
        "worst_score": float(worst["sim_internal_terminal_score"]),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "mo_horizon": mo_horizon,
    }


__all__ = [
    "ADVERSARIAL_MM_REVISION",
    "P_BUY_GRID",
    "P_BUY_HI",
    "P_BUY_LO",
    "AdversaryAgent",
    "RandomAdversary",
    "adversarial_episode",
    "constant_flow_sweep",
    "make_pareto_duration",
    "make_uniform_duration",
    "run_adversarial_eval",
]
