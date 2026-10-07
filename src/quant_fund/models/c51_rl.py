"""C51 categorical distributional RL: reusable core + ZI-LOB step API.

**Labeled SYNTHETIC** research infrastructure (backlog B4-ii,
``docs/SOTA_WAVE13_BACKLOG.md``): the model-layer half of the C51 lane — a
torch-free numpy reference for the categorical Bellman projection, a generic
categorical DQN agent, and a small ``reset()``/``step()`` environment adapter
over the lane B4-i zero-intelligence limit-order book.

References:

- Bellemare, M.G., Dabney, W., Munos, R. (2017). A distributional perspective
  on reinforcement learning. *ICML 2017*, arXiv:1707.06887 — the fixed atom
  support ``{z_i}`` (Sec. 4, ``V_min=-10``, ``V_max=10``, ``N=51`` defaults
  reproduced here), the categorical projection ``Phi`` of Algorithm 1
  (Eqs. 7-9: shift target atoms by ``r + gamma z``, clip to the support, split
  each atom's mass over its two neighbours), the categorical cross-entropy
  loss ``-sum_i m_i log p_i`` (Eq. 11), and the ``Q = E[Z]`` read-out used for
  action selection (Eq. 5).
- Bellemare, M.G., Dabney, W., Ostrovski, G., ... (2017) / same paper, Sec. 3:
  the distributional policy evaluation operator ``Z^pi`` whose fixed point the
  agent regresses toward; the projection is what makes the Bellman operator
  representable on a fixed support.
- van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning
  with double Q-learning. *AAAI 2016*, arXiv:1509.06461 — decoupled target
  action selection: the *online* net picks ``a*``, the *target* net supplies
  the return distribution for ``a*``.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. *Nature* 518, 529-533 — periodic hard target updates and uniform
  experience replay.
- Moret, Lillo (2026). Deep learning of robust market making under
  regime-switching order flow. arXiv:2609.11614 — the ZI-LOB market-making
  setting this adapter exposes; the paper-faithful session lane lives in
  ``microstructure.rl_market_maker`` (see differentiation below).
- Avellaneda, Stoikov (2008). High-frequency trading in a limit order book.
  *Quantitative Finance* 8(3):217-224; Gueant, Lehalle, Fernandez-Tapia (2012).
  *Operations Research* 60(1):1167-1187 — the closed-form reference policies
  mapped onto this environment's discrete action space by
  :func:`classic_action_policy`.

Composition / differentiation (nothing here modifies the lane B4-i simulator):

- ``microstructure.zi_lob_simulator`` owns the event engine, the FIFO matching,
  the Markov regime flow, the AS/GLFT closed forms and their ``QuotePolicy``
  adapters. :class:`ZILobQuoteEnv` **wraps** a :class:`ZILobSimulator` instance
  by composition (never subclassing internals, never editing the module) and
  exposes the gym-style step API the lane module deliberately does not have.
- ``microstructure.rl_market_maker`` is the *paper-faithful* C51 market maker:
  the Moret-Lillo state vector (their Eq. 10), the SMDP event-clock discount
  ``gamma_event**N_t`` with n-step folding (Eqs. 9, 23-24), dueling heads, the
  Beta-Bernoulli flow-bias filter (Appendix A), session accounting and the
  AS/GLFT evaluation harness. This module is the *reusable algebra* beside it:
  a numpy :func:`categorical_projection` reference (the lane module's
  projection exists only inside a torch no-grad block, so it cannot be checked
  against hand-computed atoms), a plain per-action softmax-head
  :class:`C51Agent` with optional risk-averse quantile action selection, and
  a minimal observation/reward contract for experiments. Its uniform ring
  buffer is **reused** from the lane module (:class:`ReplayBuffer`, single
  source of truth) rather than reimplemented.
- ``models.policy_gradient_maker`` is the on-policy sibling (clipped PPO) on
  the same synthetic book; this module is the off-policy distributional one.
  The two share no code path.

Honesty: everything here runs inside a SYNTHETIC zero-intelligence simulator
with exogenous parametrized flow. All money-like outputs are namespaced
``sim_internal_*`` — simulator-internal learning signals and accounting
diagnostics, never headline metrics, never proper forecast scores, never
market evidence. No Sharpe/Sortino/Calmar/P&L/NAV headline anywhere
(``FORBIDDEN_RESEARCH_METRIC_KEYS`` in ``quant_fund.research.catalog``); no
broker connectivity and no live-trading claim. ZI flow's mechanical response
to our quotes does not establish endogenous strategic competitors, latency,
hidden liquidity, financing or queue calibration.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models.deep_hedging``), so this module imports
cleanly without torch and every torch entry point raises ``ImportError`` with
install guidance. The numpy core (atom support, projection, quantile
read-outs, environment, policies) is usable torch-free. Fail-closed edges:
``ValueError`` on a degenerate atom grid (``n_atoms < 2``, ``v_min >= v_max``,
non-uniform spacing, malformed probabilities), on invalid configs, on
out-of-range actions, on marketable quote offsets, and on shape mismatches;
``RuntimeError`` on an event budget overrun or a non-finite loss. Training is
CPU single-thread and deterministic given ``seed`` (GPU determinism is not
claimed).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.rl_market_maker import ReplayBuffer
from quant_fund.microstructure.zi_lob_simulator import (
    MMState,
    MOFlow,
    ZILobConfig,
    ZILobSimulator,
    as_policy,
    glft_policy,
    santa_fe_config,
)

Array = NDArray[np.float64]

#: One discrete quoting cell: ``(delta_bid, delta_ask)`` tick offsets relative
#: to the current best quotes, ``bid = best_bid - delta_bid * tick`` and
#: ``ask = best_ask + delta_ask * tick``. ``delta = -1`` posts one tick *inside*
#: the touch (a new best quote, front of a fresh FIFO queue — the only way to
#: be filled promptly in a book whose zero-intelligence limit flow keeps
#: re-posting inside you), ``0`` joins the touch (back of its queue), and
#: ``+k`` steps ``k`` ticks deeper. Levels are always clamped to stay strictly
#: inside the opposite quote and uncrossed, so the engine's fail-closed
#: marketable-order guard can never trip (see :meth:`ZILobQuoteEnv.step`).
QuoteAction = tuple[int, int]
ClassicKind = Literal["as", "glft"]
EnvPolicy = Callable[[Array, "ZILobQuoteEnv"], int]

__all__ = [
    "C51_RL_REVISION",
    "DEFAULT_QUOTE_ACTIONS",
    "ENV_TAG",
    "N_OBS_FEATURES",
    "C51Agent",
    "C51AgentConfig",
    "C51TrainResult",
    "EnvStep",
    "ZILobQuoteEnv",
    "ZILobQuoteEnvConfig",
    "agent_policy",
    "atom_support",
    "categorical_projection",
    "classic_action_policy",
    "compare_c51_baselines",
    "distribution_mean",
    "distribution_quantile",
    "nearest_quote_action",
    "random_action_policy",
    "run_env_episode",
    "train_c51_on_zi_lob",
    "validate_quote_actions",
]

C51_RL_REVISION = "SYNTHETIC_C51_CORE_v1"

#: Tag on this environment's resting orders, so fills can be attributed the
#: same way ``run_mm_session`` attributes them (lane B4-i convention).
ENV_TAG = "c51_env_session"

#: Observation width of :meth:`ZILobQuoteEnv.observation` (see its docstring).
N_OBS_FEATURES = 9

#: Default action space: six spread/skew cells over ``{-1, 0, +1, +2}`` tick
#: offsets. ``(-1, -1)`` takes the front of both queues (the most aggressive
#: spread-capture cell), ``(0, 0)``/``(1, 1)``/``(2, 2)`` step symmetrically
#: back, and the two skew cells ``(-1, 0)``/``(0, -1)`` put one leg inside the
#: touch while the other joins it — the pair that lets a policy lean into
#: one-sided flow. Same cell count as the Moret-Lillo grid (their Eq. 15),
#: with ``+2`` in place of their second inside-posting cell.
DEFAULT_QUOTE_ACTIONS: tuple[QuoteAction, ...] = (
    (-1, -1),
    (0, 0),
    (1, 1),
    (2, 2),
    (-1, 0),
    (0, -1),
)

# Deterministic normalization clips (a transiently degenerate book must not be
# able to inject an outlier into the network input).
_SPREAD_CLIP = 8.0
_DEPTH_CLIP = 8.0
_RETURN_CLIP = 5.0
_PROB_SUM_TOL = 1e-6
_GRID_SPACING_TOL = 1e-9


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the C51 core needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (local; the simulator's are private)
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
    return int(x)


def _seed_int(x: int) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise ValueError(f"seed must be an int, got {x!r}")
    return int(x)


def _synthetic_envelope(seed: int) -> dict[str, Any]:
    """The lane honesty header every returned mapping carries."""
    return {
        "label": "SYNTHETIC",
        "data_source": C51_RL_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# Categorical support + Algorithm 1 projection (numpy core, torch-free)
# ---------------------------------------------------------------------------


def atom_support(v_min: float, v_max: float, n_atoms: int) -> tuple[Array, float]:
    """Fixed C51 return support ``z_i = v_min + i * dz`` (Bellemare et al. 2017).

    Returns ``(atoms, dz)`` with ``dz = (v_max - v_min) / (n_atoms - 1)``.
    Fail-closed on ``n_atoms < 2``, non-finite bounds, and ``v_min >= v_max``.
    """
    n = _int_at_least(n_atoms, 2, "n_atoms")
    lo = float(v_min)
    hi = float(v_max)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"v_min/v_max must be finite, got ({v_min!r}, {v_max!r})")
    if hi <= lo:
        raise ValueError(f"require v_min < v_max, got ({v_min!r}, {v_max!r})")
    atoms = np.linspace(lo, hi, n, dtype=np.float64)
    return atoms, float((hi - lo) / (n - 1))


def _as_atom_grid(atoms: Array | Sequence[float], *, name: str = "atoms") -> Array:
    z = np.asarray(atoms, dtype=np.float64).ravel()
    if z.size < 2:
        raise ValueError(f"{name} must hold at least two atoms, got {z.size}")
    if not np.all(np.isfinite(z)):
        raise ValueError(f"{name} must be finite (NaN/inf rejected)")
    gaps = np.diff(z)
    if not np.all(gaps > 0.0):
        raise ValueError(f"{name} must be strictly increasing, got {z.tolist()!r}")
    if float(gaps.max() - gaps.min()) > _GRID_SPACING_TOL:
        raise ValueError(
            f"{name} must be uniformly spaced (Algorithm 1 assumes a constant dz); "
            f"got spacings in [{gaps.min():.6g}, {gaps.max():.6g}]"
        )
    return z


def _as_prob_rows(probs: Array | Sequence[Sequence[float]], n_atoms: int) -> Array:
    p = np.atleast_2d(np.asarray(probs, dtype=np.float64))
    if p.ndim != 2 or p.shape[1] != n_atoms:
        raise ValueError(
            f"probs must be a (batch, {n_atoms}) array of atom masses, got {np.shape(probs)}"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError("probs must be finite (NaN/inf rejected)")
    if np.any(p < 0.0):
        raise ValueError("probs must be non-negative")
    if np.any(np.abs(p.sum(axis=1) - 1.0) > _PROB_SUM_TOL):
        raise ValueError("each row of probs must sum to 1 (a categorical distribution)")
    return p


def distribution_mean(probs: Array | Sequence[Sequence[float]], atoms: Array) -> Array:
    """``Q = E[Z] = sum_i p_i z_i`` per row (Bellemare et al. 2017, Eq. 5)."""
    z = _as_atom_grid(atoms)
    p = _as_prob_rows(probs, z.size)
    return np.asarray(p @ z, dtype=np.float64)


def distribution_quantile(
    probs: Array | Sequence[Sequence[float]], atoms: Array, tau: float
) -> Array:
    """Lower ``tau``-quantile of each row's categorical distribution.

    ``q_tau = min{z_i : F(z_i) >= tau}`` with ``F`` the atom CDF — the
    inverse-CDF read-out that makes a C51 head usable for risk-averse action
    selection (a distributional agent can rank actions by a tail quantile
    instead of the mean). Fail-closed on ``tau`` outside ``(0, 1)``.
    """
    t = float(tau)
    if not math.isfinite(t) or t <= 0.0 or t > 1.0:
        raise ValueError(f"tau must lie in (0, 1], got {tau!r}")
    z = _as_atom_grid(atoms)
    p = _as_prob_rows(probs, z.size)
    cdf = np.cumsum(p, axis=1)
    idx = np.sum(cdf < t, axis=1).clip(max=z.size - 1)
    return np.asarray(z[idx], dtype=np.float64)


def categorical_projection(
    probs: Array | Sequence[Sequence[float]],
    rewards: float | Sequence[float] | Array,
    gammas: float | Sequence[float] | Array,
    *,
    atoms: Array | Sequence[float],
) -> Array:
    """Project a Bellman-updated categorical distribution onto the fixed support.

    Algorithm 1 / Eqs. 7-9 of Bellemare et al. (2017), vectorized over a batch:

    1. shift every target atom, ``Tz_j = clip(r + gamma * z_j, v_min, v_max)``
       — the clip is the projection's only approximation and is what keeps the
       target representable on the fixed support;
    2. locate it in atom units, ``b_j = (Tz_j - v_min) / dz``;
    3. split its mass over the bracketing atoms, ``m_{l_j} += p_j (u_j - b_j)``
       and ``m_{u_j} += p_j (b_j - l_j)``.

    ``l_j`` is clamped to ``[0, n_atoms - 2]`` and ``u_j = l_j + 1``, so an
    exact-integer ``b_j`` (including ``b_j = n_atoms - 1``) deposits its whole
    mass on one atom instead of vanishing. Total mass is preserved: the two
    weights sum to 1 for every ``j``, so a normalized input returns a
    normalized output (no renormalization step, matching the paper).

    ``rewards``/``gammas`` are scalars or per-row vectors (broadcast against
    ``probs``); ``gammas`` must lie in ``[0, 1]`` (``0`` is the terminal
    bootstrap). Fail-closed on shape, finiteness, and grid mismatches.
    """
    z = _as_atom_grid(atoms)
    n = int(z.size)
    p = _as_prob_rows(probs, n)
    batch = p.shape[0]
    dz = float(z[1] - z[0])
    r = np.broadcast_to(np.asarray(rewards, dtype=np.float64).reshape(-1), (batch,)).astype(float)
    g = np.broadcast_to(np.asarray(gammas, dtype=np.float64).reshape(-1), (batch,)).astype(float)
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(g))):
        raise ValueError("rewards and gammas must be finite (NaN/inf rejected)")
    if np.any(g < 0.0) or np.any(g > 1.0):
        raise ValueError(f"gammas must lie in [0, 1], got min={g.min():.6g} max={g.max():.6g}")
    tz = np.clip(r[:, None] + g[:, None] * z[None, :], float(z[0]), float(z[-1]))
    b = (tz - float(z[0])) / dz
    lo = np.clip(np.floor(b), 0.0, float(n - 2)).astype(np.int64)
    hi = lo + 1
    w_lo = hi.astype(np.float64) - b
    w_hi = b - lo.astype(np.float64)
    out = np.zeros_like(p)
    rows = np.broadcast_to(np.arange(batch)[:, None], lo.shape)
    np.add.at(out, (rows, lo), p * w_lo)
    np.add.at(out, (rows, hi), p * w_hi)
    return out


def validate_quote_actions(actions: Sequence[QuoteAction]) -> tuple[QuoteAction, ...]:
    """Validate a discrete quoting action space; fail closed on bad cells.

    Cells are ``(delta_bid, delta_ask)`` integer tick offsets relative to the
    touch: ``-1`` posts inside it (a new best quote), ``0`` joins it, ``+k``
    steps ``k`` ticks deeper. Offsets below ``-1`` are refused — the engine
    clamps them to the same level as ``-1``, so they are degenerate aliases
    rather than extra actions. Duplicates and a space with fewer than two
    cells are refused too (a quoting decision needs at least two cells).
    """
    cells = tuple(actions)
    if len(cells) < 2:
        raise ValueError(f"a discrete quoting action space needs >= 2 cells, got {len(cells)}")
    seen: set[QuoteAction] = set()
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise ValueError(f"action cells must be (delta_bid, delta_ask) pairs, got {cell!r}")
        for v in cell:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"action offsets must be ints, got {cell!r}")
            if v < -1 or v > 64:
                raise ValueError(
                    f"action offsets must be ints in [-1, 64] (-1 = inside the touch, "
                    f"0 = join it, k > 0 = k ticks deeper), got {cell!r}"
                )
        pair: QuoteAction = (int(cell[0]), int(cell[1]))
        if pair in seen:
            raise ValueError(f"duplicate action cell {pair!r}")
        seen.add(pair)
    return cells


# ---------------------------------------------------------------------------
# ZI-LOB step API (composition over the lane B4-i simulator; no modification)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZILobQuoteEnvConfig:
    """Environment contract for :class:`ZILobQuoteEnv`.

    ``lob`` is the wrapped :class:`ZILobConfig` (Moret-Lillo Santa Fe
    calibration by default); ``reset(seed=...)`` rebuilds the simulator from
    ``replace(lob, seed=seed)`` so one config drives many paired episodes.
    ``decision_interval`` is the throttled decision clock in simulated seconds
    (paper ``tau_c = 1 s``); ``horizon`` ends the episode. ``inventory_cap``
    gates the breaching leg and normalizes the inventory feature, so
    ``|inventory| <= cap`` holds for *any* policy. ``inventory_penalty`` is the
    quadratic holding cost in tick units; ``reward_scale`` rescales the whole
    reward so it sits inside the C51 support. ``return_scale`` normalizes the
    mid-return feature (``None`` -> ``s0 / tick``, i.e. one tick = 1.0).
    ``max_events_per_interval`` / ``max_recovery_events`` are fail-closed
    event budgets.
    """

    lob: ZILobConfig = field(default_factory=santa_fe_config)
    horizon: float = 60.0
    decision_interval: float = 1.0
    inventory_cap: int = 6
    inventory_penalty: float = 0.10
    reward_scale: float = 1.0
    quote_actions: tuple[QuoteAction, ...] = DEFAULT_QUOTE_ACTIONS
    spread_scale: float = 4.0
    depth_scale: float = 10.0
    return_scale: float | None = None
    max_events_per_interval: int = 20_000
    max_recovery_events: int = 5_000

    def __post_init__(self) -> None:
        if not isinstance(self.lob, ZILobConfig):
            raise TypeError("lob must be a ZILobConfig")
        h = _pos_finite(self.horizon, "horizon")
        di = _pos_finite(self.decision_interval, "decision_interval")
        if h < di:
            raise ValueError(f"horizon ({h}) must be >= decision_interval ({di})")
        _int_at_least(self.inventory_cap, 1, "inventory_cap")
        _nonneg_finite(self.inventory_penalty, "inventory_penalty")
        _pos_finite(self.reward_scale, "reward_scale")
        _pos_finite(self.spread_scale, "spread_scale")
        _pos_finite(self.depth_scale, "depth_scale")
        if self.return_scale is not None:
            _pos_finite(self.return_scale, "return_scale")
        _int_at_least(self.max_events_per_interval, 1, "max_events_per_interval")
        _int_at_least(self.max_recovery_events, 1, "max_recovery_events")
        validate_quote_actions(self.quote_actions)

    @property
    def effective_return_scale(self) -> float:
        """Mid-return feature scale: one tick of mid movement maps to 1.0."""
        if self.return_scale is not None:
            return float(self.return_scale)
        return float(self.lob.s0 / self.lob.tick)


@dataclass(frozen=True)
class EnvStep:
    """One decision of the step API: ``(obs, reward, terminated, info)``."""

    obs: Array
    reward: float
    terminated: bool
    info: dict[str, Any] = field(compare=False)


class ZILobQuoteEnv:
    """Gym-style quoting environment over the ZI limit-order book.

    **Wraps** a :class:`~quant_fund.microstructure.zi_lob_simulator.ZILobSimulator`
    by composition — the engine, its FIFO matching, its event clock and its
    regime flow are used exactly as lane B4-i defines them, and the simulator
    module is not modified. One :meth:`step` is one decision interval:

    1. the action cell ``(delta_bid, delta_ask)`` is turned into absolute quote
       levels around the current touch (``-1`` posts inside it, ``0`` joins it,
       ``+k`` steps deeper), clamped to stay strictly inside the opposite
       quote and uncrossed — so the engine's fail-closed marketable-order
       guard can never trip, for any action and any transient book;
    2. the leg that would breach ``inventory_cap`` is suppressed;
    3. a resting order is *kept* when its target level is unchanged and only
       canceled/reposted otherwise, so FIFO queue priority is treated as an
       economic resource ("smart quoting", Moret-Lillo Sec. 3);
    4. the event clock advances to the next decision time and fills tagged
       :data:`ENV_TAG` are accounted into ``(cash, inventory)``;
    5. the reward is the simulator-internal mark-to-market change in tick
       units minus the quadratic inventory penalty.

    Reward identity (a tested invariant, at any ``reward_scale``)::

        reward_t = reward_scale * ((MTM_t - MTM_{t-1}) / tick
                                   - inventory_penalty * (q_t / cap)^2)

    with ``MTM = cash + inventory * mid`` and ``MTM_0 = 0``, so the per-episode
    rewards telescope exactly onto the final mark-to-market:
    ``sum_t (reward_t / reward_scale + penalty_t) * tick == MTM_T``. Every term
    is a ``sim_internal_*`` learning signal for this synthetic engine — never a
    headline metric, never market evidence, no live-trading claim. The mid used
    for accounting is the decision-time mid (cached; reading it never steps the
    event clock), which is what makes the identity exact.

    Observation (9 floats, all normalized and clipped): touch spread in ticks,
    bid/ask depth at the touch, touch depth imbalance, ``inventory / cap``, the
    mid return since the previous decision in tick units, the own bid/ask
    queue front-ness ``1 / (1 + queue_ahead)`` (0 when that leg is not
    resting), and the elapsed-time fraction ``t / horizon``. The observation
    uses only information available at the decision time: the book, own
    queues, own inventory and the realized tape — never a future event and
    never the flow's hidden regime state.

    ``flow_factory`` optionally builds a fresh flow driver per episode
    (``seed -> MOFlow``), e.g. ``microstructure.rl_market_maker.paper_regime_flow``;
    ``None`` is the stationary flow of ``lob`` itself.
    """

    def __init__(
        self,
        config: ZILobQuoteEnvConfig,
        *,
        flow_factory: Callable[[int], MOFlow] | None = None,
    ) -> None:
        if not isinstance(config, ZILobQuoteEnvConfig):
            raise TypeError("config must be a ZILobQuoteEnvConfig")
        if flow_factory is not None and not callable(flow_factory):
            raise TypeError("flow_factory must be callable or None")
        self._cfg = config
        self._flow_factory = flow_factory
        self._actions = validate_quote_actions(config.quote_actions)
        self._sim: ZILobSimulator | None = None
        self._inventory = 0
        self._cash = 0.0
        self._bid_oid: int | None = None
        self._ask_oid: int | None = None
        self._bid_level: int | None = None
        self._ask_level: int | None = None
        self._trade_cursor = 0
        self._t_next = 0.0
        self._prev_mid = float(config.lob.s0)
        self._last_mid = float(config.lob.s0)
        self._terminated = True
        self._n_decisions = 0
        self._n_fills = self._n_fills_bid = self._n_fills_ask = 0
        self._n_suppressed = self._n_held = self._n_reposted = self._n_clips = 0
        self._max_abs_inv = 0
        self._abs_inv_sum = 0
        self._penalty_sum = 0.0
        self._reward_sum = 0.0
        self._fill_gain_sum = 0.0
        self._step_fills: list[tuple[float, int]] = []
        self.reset(seed=config.lob.seed)

    # -- introspection -------------------------------------------------------

    @property
    def config(self) -> ZILobQuoteEnvConfig:
        return self._cfg

    @property
    def sim(self) -> ZILobSimulator:
        """The wrapped simulator (read-only composition; never mutated here)."""
        if self._sim is None:  # pragma: no cover - __init__ always resets
            raise RuntimeError("environment is not initialized")
        return self._sim

    @property
    def quote_actions(self) -> tuple[QuoteAction, ...]:
        return self._actions

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    @property
    def obs_dim(self) -> int:
        return N_OBS_FEATURES

    @property
    def tick(self) -> float:
        return float(self._cfg.lob.tick)

    @property
    def t(self) -> float:
        return float(self.sim.t)

    @property
    def inventory(self) -> int:
        return self._inventory

    @property
    def mid(self) -> float:
        """Mid at the most recent decision point.

        Reading it never steps the event clock: book recovery (which must step
        when a side transiently empties) is confined to :meth:`reset` and
        :meth:`step`, so every accounting read-out is side-effect free.
        """
        return float(self._last_mid)

    @property
    def best_bid_level(self) -> int:
        lvl = self.sim.best_bid_level
        if lvl is None:
            raise RuntimeError("book has no bid side; call reset() or step()")
        return int(lvl)

    @property
    def best_ask_level(self) -> int:
        lvl = self.sim.best_ask_level
        if lvl is None:
            raise RuntimeError("book has no ask side; call reset() or step()")
        return int(lvl)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def mtm(self) -> float:
        """Simulator-internal mark-to-market ``cash + inventory * mid``."""
        return float(self._cash + self._inventory * self.mid)

    def mm_state(self) -> MMState:
        """Decision state for the simulator's ``QuotePolicy`` adapters."""
        sim = self.sim
        return MMState(
            t=sim.t,
            mid=sim.mid,
            best_bid=sim.best_bid,
            best_ask=sim.best_ask,
            inventory=self._inventory,
            tau=max(0.0, float(self._cfg.horizon) - float(sim.t)),
        )

    # -- internals -----------------------------------------------------------

    def _require_mid(self, why: str) -> float:
        """Return the mid, stepping (within budget) until both sides exist.

        A ZI book transiently thins — a run of market orders or cancellations
        can empty one side, leaving the mid undefined until limit flow
        re-seeds it. Mirrors the lane module's private recovery helper using
        only the public ``step()``/``mid`` surface.
        """
        sim = self.sim
        mid = sim.mid
        spent = 0
        budget = int(self._cfg.max_recovery_events)
        while mid is None:
            if spent >= budget:
                raise RuntimeError(
                    f"book did not regain two sides within {budget} events after {why}"
                )
            sim.step()
            spent += 1
            mid = sim.mid
        return float(mid)

    def _drain_trades(self) -> None:
        sim = self.sim
        while self._trade_cursor < len(sim.trades):
            tr = sim.trades[self._trade_cursor]
            self._trade_cursor += 1
            if tr.maker_tag != ENV_TAG:
                continue
            sign = 1 if tr.maker_side == "buy" else -1
            self._inventory += sign * int(tr.qty)
            self._cash -= sign * float(tr.price) * int(tr.qty)
            self._step_fills.append((float(tr.price), sign))
            self._n_fills += 1
            if sign > 0:
                self._n_fills_bid += 1
            else:
                self._n_fills_ask += 1
            if tr.maker_order_id == self._bid_oid:
                self._bid_oid = None
                self._bid_level = None
            elif tr.maker_order_id == self._ask_oid:
                self._ask_oid = None
                self._ask_level = None
            self._max_abs_inv = max(self._max_abs_inv, abs(self._inventory))

    def _cancel_outstanding(self) -> None:
        sim = self.sim
        for oid in (self._bid_oid, self._ask_oid):
            if oid is not None:
                sim.cancel_order(oid)
        self._bid_oid = self._ask_oid = None
        self._bid_level = self._ask_level = None

    def _post_leg(
        self,
        side: Literal["buy", "sell"],
        level: int | None,
        current_oid: int | None,
        current_level: int | None,
    ) -> tuple[int | None, int | None, bool]:
        """Repost one leg only when its target level changed (FIFO-preserving)."""
        sim = self.sim
        if level is None:
            if current_oid is not None:
                sim.cancel_order(current_oid)
            return None, None, False
        if current_oid is not None and current_level == level and sim.order_alive(current_oid):
            self._n_held += 1
            return current_oid, current_level, True
        if current_oid is not None:
            sim.cancel_order(current_oid)
        price = sim.level_to_price(level)
        if price <= 0.0:
            self._n_suppressed += 1
            return None, None, False
        oid = sim.submit_limit_order(side, price, ENV_TAG)
        self._n_reposted += 1
        return oid, level, False

    def _requote(self, action: QuoteAction) -> tuple[bool, bool, bool, bool]:
        d_bid, d_ask = action
        sim = self.sim
        cap = int(self._cfg.inventory_cap)
        bb, ba = sim.best_bid_level, sim.best_ask_level
        bid_level = None if bb is None else int(bb) - int(d_bid)
        ask_level = None if ba is None else int(ba) + int(d_ask)
        # Clamp to strictly inside the opposite quote, then uncross. The ZI
        # engine refuses marketable limit orders by design, and a book can
        # transiently narrow to one tick, so an inside-posting cell must never
        # be able to produce a crossed or marketable pair of quotes.
        if bid_level is not None and ba is not None and bid_level >= int(ba):
            bid_level = int(ba) - 1
            self._n_clips += 1
        if ask_level is not None and bb is not None and ask_level <= int(bb):
            ask_level = int(bb) + 1
            self._n_clips += 1
        if bid_level is not None and ask_level is not None and bid_level >= ask_level:
            bid_level = ask_level - 1
            self._n_clips += 1
        suppress_bid = bid_level is not None and self._inventory >= cap
        suppress_ask = ask_level is not None and self._inventory <= -cap
        if suppress_bid or suppress_ask:
            self._n_suppressed += int(suppress_bid) + int(suppress_ask)
        if suppress_bid:
            bid_level = None
        if suppress_ask:
            ask_level = None
        self._bid_oid, self._bid_level, held_bid = self._post_leg(
            "buy", bid_level, self._bid_oid, self._bid_level
        )
        self._ask_oid, self._ask_level, held_ask = self._post_leg(
            "sell", ask_level, self._ask_oid, self._ask_level
        )
        return suppress_bid, suppress_ask, held_bid, held_ask

    def _frontness(self, oid: int | None) -> float:
        if oid is None:
            return 0.0
        ahead = self.sim.queue_position(oid)
        return 0.0 if ahead is None else 1.0 / (1.0 + float(ahead))

    def observation(self) -> Array:
        """Current decision-time observation (pure; see the class docstring)."""
        sim = self.sim
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if bb is None or ba is None:
            raise RuntimeError("observation requires a two-sided book; call reset() first")
        cfg = self._cfg
        bid_d0 = float(sim.depth_at("buy", bb))
        ask_d0 = float(sim.depth_at("sell", ba))
        mid = float(self._last_mid)
        ret = (mid / self._prev_mid - 1.0) * cfg.effective_return_scale
        cap = float(cfg.inventory_cap)
        feats = (
            min(float(ba - bb) / cfg.spread_scale, _SPREAD_CLIP),
            min(bid_d0 / cfg.depth_scale, _DEPTH_CLIP),
            min(ask_d0 / cfg.depth_scale, _DEPTH_CLIP),
            (bid_d0 - ask_d0) / (bid_d0 + ask_d0 + 1.0),
            float(self._inventory) / cap,
            float(np.clip(ret, -_RETURN_CLIP, _RETURN_CLIP)),
            self._frontness(self._bid_oid),
            self._frontness(self._ask_oid),
            min(1.0, float(sim.t) / float(cfg.horizon)),
        )
        out = np.asarray(feats, dtype=np.float64)
        if out.shape != (N_OBS_FEATURES,) or not bool(np.all(np.isfinite(out))):
            raise RuntimeError("observation went non-finite or misshapen")
        return out

    # -- step API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> Array:
        """Rebuild the simulator (and flow) and return the first observation."""
        cfg = self._cfg
        s = int(cfg.lob.seed) if seed is None else _seed_int(seed)
        flow: MOFlow | None = None if self._flow_factory is None else self._flow_factory(s + 1)
        self._sim = ZILobSimulator(replace(cfg.lob, seed=s), flow=flow)
        self._inventory = 0
        self._cash = 0.0
        self._bid_oid = self._ask_oid = None
        self._bid_level = self._ask_level = None
        self._trade_cursor = 0
        self._t_next = float(cfg.decision_interval)
        self._terminated = False
        self._n_decisions = 0
        self._n_fills = self._n_fills_bid = self._n_fills_ask = 0
        self._n_suppressed = self._n_held = self._n_reposted = self._n_clips = 0
        self._max_abs_inv = 0
        self._abs_inv_sum = 0
        self._penalty_sum = 0.0
        self._reward_sum = 0.0
        self._fill_gain_sum = 0.0
        self._step_fills = []
        mid0 = self._require_mid("reset")
        self._prev_mid = mid0
        self._last_mid = mid0
        return self.observation()

    def step(self, action: int) -> EnvStep:
        """Apply one quoting action, advance one decision interval, reward."""
        if self._terminated:
            raise RuntimeError("episode is terminated; call reset() before step()")
        a = int(action)
        if isinstance(action, bool) or not 0 <= a < self.n_actions:
            raise ValueError(f"action must be an int in [0, {self.n_actions}), got {action!r}")
        sim = self.sim
        cfg = self._cfg
        suppress_bid, suppress_ask, held_bid, held_ask = self._requote(self._actions[a])
        mtm_prev = float(self._cash + self._inventory * self._last_mid)
        self._step_fills = []
        budget = int(cfg.max_events_per_interval)
        events = 0
        while sim.t < self._t_next:
            sim.step()
            events += 1
            if events > budget:
                raise RuntimeError(
                    f"decision interval exceeded {budget} events; raise "
                    f"max_events_per_interval or shorten decision_interval"
                )
        self._t_next += float(cfg.decision_interval)
        self._drain_trades()
        mid_next = self._require_mid("decision end")
        self._last_mid = mid_next
        mtm_next = float(self._cash + self._inventory * mid_next)
        fill_gain = (
            sum(sign * (mid_next - price) for price, sign in self._step_fills) / cfg.lob.tick
        )
        penalty = (
            float(cfg.inventory_penalty) * (float(self._inventory) / float(cfg.inventory_cap)) ** 2
        )
        reward = float(cfg.reward_scale) * ((mtm_next - mtm_prev) / cfg.lob.tick - penalty)
        self._n_decisions += 1
        self._abs_inv_sum += abs(self._inventory)
        self._penalty_sum += penalty
        self._reward_sum += reward
        self._fill_gain_sum += fill_gain
        obs = self.observation()
        self._prev_mid = mid_next
        terminated = bool(sim.t >= float(cfg.horizon))
        self._terminated = terminated
        info: dict[str, Any] = {
            "label": "SYNTHETIC",
            "t": float(sim.t),
            "mid": mid_next,
            "inventory": int(self._inventory),
            "action": a,
            "action_cell": self._actions[a],
            "suppressed_bid": suppress_bid,
            "suppressed_ask": suppress_ask,
            "held_bid": held_bid,
            "held_ask": held_ask,
            "n_fills_step": len(self._step_fills),
            "bid_level": self._bid_level,
            "ask_level": self._ask_level,
            "sim_internal_reward": reward,
            "sim_internal_mtm": mtm_next,
            "sim_internal_fill_gain_ticks": fill_gain,
            "sim_internal_inventory_penalty": penalty,
        }
        if terminated:
            info.update(self.episode_summary())
        return EnvStep(obs=obs, reward=reward, terminated=terminated, info=info)

    def episode_summary(self) -> dict[str, Any]:
        """Aggregate ``sim_internal_*`` accounting for the current episode."""
        n = max(1, self._n_decisions)
        out = _synthetic_envelope(self._cfg.lob.seed)
        out.update(
            {
                "horizon": float(self._cfg.horizon),
                "decision_interval": float(self._cfg.decision_interval),
                "inventory_cap": int(self._cfg.inventory_cap),
                "n_decisions": int(self._n_decisions),
                "n_fills": int(self._n_fills),
                "n_fills_bid": int(self._n_fills_bid),
                "n_fills_ask": int(self._n_fills_ask),
                "n_quote_suppressions": int(self._n_suppressed),
                "n_quote_clips": int(self._n_clips),
                "n_quotes_held": int(self._n_held),
                "n_quotes_reposted": int(self._n_reposted),
                "inventory_final": int(self._inventory),
                "max_abs_inventory": int(self._max_abs_inv),
                "mean_abs_inventory": float(self._abs_inv_sum / n),
                "sim_internal_reward_sum": float(self._reward_sum),
                "sim_internal_reward_mean": float(self._reward_sum / n),
                "sim_internal_fill_gain_ticks_sum": float(self._fill_gain_sum),
                "sim_internal_inventory_penalty_sum": float(self._penalty_sum),
                "sim_internal_mtm_pnl_final": float(self.mtm),
            }
        )
        return out


# ---------------------------------------------------------------------------
# C51 agent (torch-gated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C51AgentConfig:
    """C51 hyperparameters.

    The support defaults are the paper's Atari setting (Bellemare et al. 2017,
    Sec. 5.1: ``N = 51`` atoms on ``[-10, 10]``). ``target_update_every`` is
    the periodic hard sync of Mnih et al. (2015); ``double_dqn`` enables the
    decoupled target action selection of van Hasselt et al. (2016).
    ``risk_quantile`` (``None`` = mean) ranks actions by a lower quantile of
    the predicted return distribution instead of its mean — a distributional
    agent's risk-averse read-out. Exploration is epsilon-greedy with a linear
    decay (deterministic under ``seed``).
    """

    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    hidden: tuple[int, ...] = (128,)
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 32
    buffer_capacity: int = 20_000
    target_update_every: int = 100
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 2_000
    double_dqn: bool = True
    risk_quantile: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        _int_at_least(self.n_atoms, 2, "n_atoms")
        lo = float(self.v_min)
        hi = float(self.v_max)
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise ValueError(f"require v_min < v_max finite, got ({self.v_min!r}, {self.v_max!r})")
        widths = tuple(int(h) for h in self.hidden)
        if not widths or any(h < 1 for h in widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths, got {self.hidden!r}"
            )
        g = float(self.gamma)
        if not math.isfinite(g) or not 0.0 < g < 1.0:
            raise ValueError(f"gamma must lie in (0, 1), got {self.gamma!r}")
        _pos_finite(self.lr, "lr")
        bs = _int_at_least(self.batch_size, 1, "batch_size")
        cap = _int_at_least(self.buffer_capacity, 1, "buffer_capacity")
        if cap < bs:
            raise ValueError(f"buffer_capacity ({cap}) must be >= batch_size ({bs})")
        _int_at_least(self.target_update_every, 1, "target_update_every")
        e0 = _prob(self.eps_start, "eps_start")
        e1 = _prob(self.eps_end, "eps_end")
        if e1 > e0:
            raise ValueError(f"eps_end ({e1}) must be <= eps_start ({e0})")
        _int_at_least(self.eps_decay_steps, 1, "eps_decay_steps")
        if self.risk_quantile is not None:
            t = float(self.risk_quantile)
            if not math.isfinite(t) or t <= 0.0 or t >= 1.0:
                raise ValueError(
                    f"risk_quantile must lie in (0, 1) or be None, got {self.risk_quantile!r}"
                )
        _seed_int(self.seed)


class C51Agent:
    """Categorical distributional DQN over a fixed ``n_atoms`` support.

    An MLP maps an observation to ``n_actions * n_atoms`` logits; a softmax per
    action is the predicted return distribution ``Z(s, a)`` and its mean is
    ``Q(s, a)`` (Bellemare et al. 2017, Eq. 5). :meth:`learn` minimizes the
    categorical cross-entropy between ``Z(s, a)`` and the projected Bellman
    target (:meth:`project_target_distribution`, Algorithm 1), with the target
    action chosen by the *online* net and its distribution read from the
    *target* net when ``double_dqn`` (van Hasselt et al. 2016). The target net
    is hard-synced every ``target_update_every`` gradient steps.

    Determinism: ``torch.manual_seed`` and a private ``numpy`` Generator are
    seeded from ``config.seed`` and torch runs single-threaded on CPU, so an
    identical call sequence is bit-identical (GPU determinism is not claimed).
    """

    def __init__(self, obs_dim: int, n_actions: int, config: C51AgentConfig) -> None:
        torch = _torch()
        dim = _int_at_least(obs_dim, 1, "obs_dim")
        acts = _int_at_least(n_actions, 1, "n_actions")
        if not isinstance(config, C51AgentConfig):
            raise TypeError("config must be a C51AgentConfig")
        self._cfg = config
        self._obs_dim = dim
        self._n_actions = acts
        self._torch = torch
        torch.set_num_threads(1)
        self._rng = np.random.default_rng(int(config.seed) + 424_243)
        atoms_np, dz = atom_support(config.v_min, config.v_max, config.n_atoms)
        self._atoms_np = atoms_np
        self._dz = dz
        self._atoms = torch.as_tensor(atoms_np, dtype=torch.float32)
        with torch.random.fork_rng():
            torch.manual_seed(int(config.seed))
            self._online = self._new_net()
            self._target = self._new_net()
        self._n_updates = 0
        self._n_decisions = 0
        self._n_target_syncs = 0
        self.sync_target()
        self._opt = torch.optim.AdamW(self._online.parameters(), lr=float(config.lr))
        self._buffer = ReplayBuffer(
            int(config.buffer_capacity), dim, seed=int(config.seed) + 104_729
        )

    # -- construction --------------------------------------------------------

    def _new_net(self) -> Any:
        torch = self._torch
        layers: list[Any] = []
        d = self._obs_dim
        for h in self._cfg.hidden:
            layers.append(torch.nn.Linear(d, int(h)))
            layers.append(torch.nn.ReLU())
            d = int(h)
        layers.append(torch.nn.Linear(d, self._n_actions * int(self._cfg.n_atoms)))
        return torch.nn.Sequential(*layers)

    def _logits(self, net: Any, x: Any) -> Any:
        return net(x).view(-1, self._n_actions, int(self._cfg.n_atoms))

    # -- introspection -------------------------------------------------------

    @property
    def config(self) -> C51AgentConfig:
        return self._cfg

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def atoms(self) -> Array:
        return self._atoms_np.copy()

    @property
    def delta_z(self) -> float:
        return self._dz

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    @property
    def n_target_syncs(self) -> int:
        return self._n_target_syncs

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def epsilon(self) -> float:
        """Current epsilon of the linear decay schedule."""
        cfg = self._cfg
        frac = min(1.0, self._n_decisions / float(cfg.eps_decay_steps))
        return float(cfg.eps_end + (cfg.eps_start - cfg.eps_end) * (1.0 - frac))

    # -- policy --------------------------------------------------------------

    def _check_obs(self, obs: Array | Sequence[float]) -> Array:
        x = np.asarray(obs, dtype=np.float64).ravel()
        if x.shape != (self._obs_dim,):
            raise ValueError(f"obs must have shape ({self._obs_dim},), got {np.shape(obs)}")
        if not bool(np.all(np.isfinite(x))):
            raise ValueError("obs must be finite (NaN/inf rejected)")
        return x

    def action_distributions(self, obs: Array) -> Array:
        """Predicted return distributions ``(n_actions, n_atoms)``, no grad."""
        x = self._check_obs(obs)
        torch = self._torch
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            p = torch.softmax(self._logits(self._online, xt), dim=-1)
            out = p.squeeze(0).detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    def action_scores(self, obs: Array) -> Array:
        """Per-action selection scores: distribution mean, or ``risk_quantile``."""
        p = self.action_distributions(obs)
        if self._cfg.risk_quantile is None:
            return distribution_mean(p, self._atoms_np)
        return distribution_quantile(p, self._atoms_np, float(self._cfg.risk_quantile))

    def q_values(self, obs: Array) -> Array:
        """Per-action ``Q = E[Z]`` (always the mean, regardless of risk read-out)."""
        return distribution_mean(self.action_distributions(obs), self._atoms_np)

    def act(self, obs: Array, *, greedy: bool = False) -> int:
        """Pick an action index; epsilon-greedy unless ``greedy``.

        Non-greedy calls advance the epsilon schedule (one counted decision);
        greedy evaluation calls leave the agent untouched. Ties break to the
        lowest action index.
        """
        x = self._check_obs(obs)
        if greedy:
            return int(np.argmax(self.action_scores(x)))
        eps = self.epsilon
        self._n_decisions += 1
        if float(self._rng.random()) < eps:
            return int(self._rng.integers(self._n_actions))
        return int(np.argmax(self.action_scores(x)))

    # -- experience ----------------------------------------------------------

    def store_transition(
        self, obs: Array, action: int, reward: float, next_obs: Array, done: bool
    ) -> None:
        """Push one transition into the uniform replay buffer (fixed gamma)."""
        s = self._check_obs(obs)
        n = self._check_obs(next_obs)
        a = int(action)
        if isinstance(action, bool) or not 0 <= a < self._n_actions:
            raise ValueError(f"action must be an int in [0, {self._n_actions}), got {action!r}")
        r = float(reward)
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        self._buffer.push(s, a, r, n, float(self._cfg.gamma), bool(done))

    # -- learning ------------------------------------------------------------

    def sync_target(self) -> None:
        """Hard-copy the online parameters into the target network."""
        self._target.load_state_dict(self._online.state_dict())
        self._n_target_syncs += 1

    def project_target_distribution(self, probs: Any, rewards: Any, gammas: Any) -> Any:
        """Torch mirror of :func:`categorical_projection` (Algorithm 1).

        ``probs`` is ``(batch, n_atoms)`` of target masses, ``rewards`` and
        ``gammas`` are ``(batch,)``. Kept public so the torch path can be
        asserted equal to the numpy reference on the same inputs — the numpy
        function is the one that can be checked against hand-computed atoms.
        """
        torch = self._torch
        cfg = self._cfg
        n = int(cfg.n_atoms)
        z = self._atoms
        tz = torch.clamp(
            rewards.unsqueeze(1) + gammas.unsqueeze(1) * z.unsqueeze(0),
            float(cfg.v_min),
            float(cfg.v_max),
        )
        b = (tz - float(cfg.v_min)) / self._dz
        lo = b.floor().to(torch.long).clamp(0, n - 2)
        hi = lo + 1
        lo_f = lo.to(torch.float32)
        hi_f = hi.to(torch.float32)
        m = torch.zeros_like(probs)
        m.scatter_add_(1, lo, probs * (hi_f - b))
        m.scatter_add_(1, hi, probs * (b - lo_f))
        return m

    def learn(self) -> float | None:
        """One C51 gradient step; ``None`` while the buffer is below batch size.

        Cross-entropy between the online net's distribution for the taken
        action and the projected target (Bellemare et al. 2017, Eq. 11). With
        ``double_dqn`` the target action is ``argmax_a E[Z_online(s', a)]`` and
        the target distribution is ``Z_target(s', a*)``; otherwise both come
        from the target net. A non-finite loss raises rather than poisoning the
        weights (usually a reward scale far outside ``[v_min, v_max]``).
        """
        cfg = self._cfg
        if len(self._buffer) < cfg.batch_size:
            return None
        torch = self._torch
        batch = self._buffer.sample(int(cfg.batch_size))
        n = int(cfg.n_atoms)
        s = torch.as_tensor(batch["states"])
        a = torch.as_tensor(batch["actions"], dtype=torch.long)
        r = torch.as_tensor(batch["rewards"])
        s2 = torch.as_tensor(batch["next_states"])
        done = torch.as_tensor(batch["dones"], dtype=torch.float32)
        gam = torch.as_tensor(batch["gammas"]) * (1.0 - done)

        log_p = torch.log_softmax(self._logits(self._online, s), dim=-1)
        log_p_a = log_p.gather(1, a.view(-1, 1, 1).expand(-1, -1, n)).squeeze(1)
        with torch.no_grad():
            p_target = torch.softmax(self._logits(self._target, s2), dim=-1)
            if bool(cfg.double_dqn):
                p_online = torch.softmax(self._logits(self._online, s2), dim=-1)
                q2 = (p_online * self._atoms.unsqueeze(0)).sum(dim=-1)
            else:
                q2 = (p_target * self._atoms.unsqueeze(0)).sum(dim=-1)
            a_star = q2.argmax(dim=1)
            m = p_target.gather(1, a_star.view(-1, 1, 1).expand(-1, 1, n)).squeeze(1)
            target = self.project_target_distribution(m, r, gam)
        loss = -(target * log_p_a).sum(dim=-1).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "C51 cross-entropy loss went non-finite; check reward scale vs support"
            )
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        self._n_updates += 1
        if self._n_updates % int(cfg.target_update_every) == 0:
            self.sync_target()
        return float(loss.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# Policies (random / trained agent / AS-GLFT reference)
# ---------------------------------------------------------------------------


def random_action_policy(*, seed: int, n_actions: int) -> EnvPolicy:
    """Uniform-random quoting baseline (seeded; the learning-signal control)."""
    acts = _int_at_least(n_actions, 1, "n_actions")
    _seed_int(seed)
    rng = np.random.default_rng(int(seed))

    def policy(obs: Array, env: ZILobQuoteEnv) -> int:
        del obs, env  # a uniform baseline ignores the state by construction
        return int(rng.integers(acts))

    return policy


def agent_policy(agent: C51Agent, *, greedy: bool = True) -> EnvPolicy:
    """Wrap a :class:`C51Agent` as an :data:`EnvPolicy`."""
    if not isinstance(agent, C51Agent):
        raise TypeError("agent must be a C51Agent")

    def policy(obs: Array, env: ZILobQuoteEnv) -> int:
        del env
        return int(agent.act(obs, greedy=greedy))

    return policy


def nearest_quote_action(env: ZILobQuoteEnv, *, bid: float | None, ask: float | None) -> int:
    """Map closed-form quote prices onto the nearest discrete action cell.

    Offsets are recovered in ticks from the current touch (``d_bid =
    (best_bid - bid) / tick``, ``d_ask = (ask - best_ask) / tick``; either may
    be negative when the closed form quotes inside the touch) and matched to
    the cell minimizing squared offset distance, first index winning ties. A
    stood-down leg (``None``) maps to the widest available offset on that side.
    """
    if not isinstance(env, ZILobQuoteEnv):
        raise TypeError("env must be a ZILobQuoteEnv")
    cells = env.quote_actions
    tick = env.tick
    widest_bid = max(c[0] for c in cells)
    widest_ask = max(c[1] for c in cells)
    if bid is None:
        d_bid = float(widest_bid)
    else:
        p = float(bid)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError(f"bid must be positive and finite, got {bid!r}")
        d_bid = (env.sim.level_to_price(env.best_bid_level) - p) / tick
    if ask is None:
        d_ask = float(widest_ask)
    else:
        p = float(ask)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError(f"ask must be positive and finite, got {ask!r}")
        d_ask = (p - env.sim.level_to_price(env.best_ask_level)) / tick
    best_idx = 0
    best_dist = math.inf
    for i, (cb, ca) in enumerate(cells):
        dist = (float(cb) - d_bid) ** 2 + (float(ca) - d_ask) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return int(best_idx)


def classic_action_policy(
    *,
    kind: ClassicKind = "as",
    gamma: float,
    sigma: float,
    kappa: float,
    a_fill: float | None = None,
    tick: float | None = None,
) -> EnvPolicy:
    """Avellaneda-Stoikov or GLFT closed-form quotes as a discrete env policy.

    Reuses the lane B4-i ``QuotePolicy`` adapters (``as_policy`` /
    ``glft_policy``) — the single source of truth for those closed forms — and
    snaps their prices onto this environment's action grid with
    :func:`nearest_quote_action`, so a trained agent and a classic reference
    are compared under an identical step API, reward and seed. A stand-down
    (the GLFT adapter returns ``(None, None)`` under extreme skew, or ``tau``
    has expired) maps to the widest cell.
    """
    if kind not in ("as", "glft"):
        raise ValueError(f"kind must be 'as' or 'glft', got {kind!r}")
    if kind == "glft" and a_fill is None:
        raise ValueError("kind='glft' requires a_fill > 0")
    if kind == "as":
        quote_policy = as_policy(gamma=gamma, sigma=sigma, kappa=kappa, tick=tick)
    else:
        quote_policy = glft_policy(
            gamma=gamma,
            sigma=sigma,
            kappa=kappa,
            a_fill=_pos_finite(a_fill if a_fill is not None else 1.0, "a_fill"),
            tick=tick,
        )

    def policy(obs: Array, env: ZILobQuoteEnv) -> int:
        del obs
        state = env.mm_state()
        if state.mid is None or state.tau <= 0.0:
            return nearest_quote_action(env, bid=None, ask=None)
        bid, ask = quote_policy(state)
        return nearest_quote_action(env, bid=bid, ask=ask)

    return policy


# ---------------------------------------------------------------------------
# Episode runner, training loop, baseline comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C51TrainResult:
    """Training outcome of :func:`train_c51_on_zi_lob` (SYNTHETIC)."""

    agent: C51Agent = field(repr=False, compare=False)
    loss_curve: Array
    reward_curve: Array
    metrics: dict[str, float]
    n_decisions: int
    episodes: int
    seed: int
    label: str = "SYNTHETIC"


def _windowed_means(rewards: Sequence[float], window: int) -> list[float]:
    """Mean reward per consecutive ``window``-decision bucket (last is partial)."""
    x = [float(r) for r in rewards]
    if not x:
        return []
    return [float(np.mean(x[i : i + window])) for i in range(0, len(x), int(window))]


def run_env_episode(
    env: ZILobQuoteEnv,
    policy: EnvPolicy,
    *,
    seed: int | None = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """Run one episode of ``policy`` inside ``env``; return ``sim_internal_*`` rows.

    ``seed`` reseeds the wrapped simulator (paired-seed comparisons pass the
    same seed to every policy); ``max_decisions`` truncates early. The summary
    is the environment's :meth:`~ZILobQuoteEnv.episode_summary` plus
    ``sim_internal_reward_mean_per_decision`` — simulator-internal diagnostics
    on a SYNTHETIC engine, never market evidence and never a headline metric.
    """
    if not isinstance(env, ZILobQuoteEnv):
        raise TypeError("env must be a ZILobQuoteEnv")
    if not callable(policy):
        raise TypeError("policy must be callable")
    if max_decisions is not None:
        _int_at_least(max_decisions, 1, "max_decisions")
    obs = env.reset(seed=seed)
    limit = env.config.horizon / env.config.decision_interval
    budget = int(limit) + 1 if max_decisions is None else min(int(max_decisions), int(limit) + 1)
    steps = 0
    terminated = False
    while steps < budget:
        action = int(policy(obs, env))
        out = env.step(action)
        obs = out.obs
        steps += 1
        terminated = bool(out.terminated)
        if terminated:
            break
    summary = env.episode_summary()
    summary["n_steps_run"] = int(steps)
    summary["policy_seed"] = None if seed is None else int(seed)
    summary["terminated_by_horizon"] = terminated
    summary["truncated"] = bool(not terminated)
    return summary


def train_c51_on_zi_lob(
    *,
    env_config: ZILobQuoteEnvConfig,
    total_decisions: int,
    agent: C51Agent | None = None,
    agent_config: C51AgentConfig | None = None,
    learn_every: int = 1,
    reward_window: int = 100,
    train_seed: int = 0,
    flow_factory: Callable[[int], MOFlow] | None = None,
) -> C51TrainResult:
    """Train a :class:`C51Agent` inside the ZI-LOB step API (SYNTHETIC).

    One continuous interaction stream: the agent acts epsilon-greedy, every
    transition is stored, and :meth:`C51Agent.learn` runs every
    ``learn_every`` decisions; the episode restarts (with a fresh simulator
    seed derived from ``train_seed`` and the episode index) whenever the
    horizon is reached, so the buffer mixes many books. ``reward_window``
    buckets the training reward into the first/last-window means reported as
    ``sim_internal_reward_mean_first_window`` /
    ``sim_internal_reward_mean_last_window`` — the learning signal. Those keys
    are simulator-internal diagnostics, never headline metrics.
    """
    if not isinstance(env_config, ZILobQuoteEnvConfig):
        raise TypeError("env_config must be a ZILobQuoteEnvConfig")
    total = _int_at_least(total_decisions, 1, "total_decisions")
    every = _int_at_least(learn_every, 1, "learn_every")
    window = _int_at_least(reward_window, 1, "reward_window")
    _seed_int(train_seed)
    env = ZILobQuoteEnv(env_config, flow_factory=flow_factory)
    if agent is None:
        cfg = agent_config if agent_config is not None else C51AgentConfig(seed=int(train_seed))
        agent = C51Agent(env.obs_dim, env.n_actions, cfg)
    elif not isinstance(agent, C51Agent):
        raise TypeError("agent must be a C51Agent or None")
    elif agent_config is not None:
        raise ValueError("pass either agent or agent_config, not both")
    if agent.obs_dim != env.obs_dim or agent.n_actions != env.n_actions:
        raise ValueError(
            f"agent expects obs_dim={agent.obs_dim}, n_actions={agent.n_actions}; env provides "
            f"obs_dim={env.obs_dim}, n_actions={env.n_actions}"
        )

    losses: list[float] = []
    rewards: list[float] = []
    episode = 0
    decisions = 0
    obs = env.reset(seed=int(train_seed))
    while decisions < total:
        action = agent.act(obs)
        out = env.step(action)
        agent.store_transition(obs, action, out.reward, out.obs, out.terminated)
        obs = out.obs
        decisions += 1
        rewards.append(float(out.reward))
        if decisions % every == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        if out.terminated:
            episode += 1
            obs = env.reset(seed=int(train_seed) + 1_000_003 * episode)
    curve = _windowed_means(rewards, window)
    metrics = {
        "sim_internal_reward_mean_train": float(np.mean(rewards)) if rewards else float("nan"),
        "sim_internal_reward_mean_first_window": float(curve[0]) if len(curve) else float("nan"),
        "sim_internal_reward_mean_last_window": float(curve[-1]) if len(curve) else float("nan"),
        "sim_internal_reward_window_improvement": float(curve[-1] - curve[0])
        if len(curve)
        else float("nan"),
        "sim_internal_loss_mean_first": float(np.mean(losses[: max(1, len(losses) // 10)]))
        if losses
        else float("nan"),
        "sim_internal_loss_mean_last": float(np.mean(losses[-max(1, len(losses) // 10) :]))
        if losses
        else float("nan"),
        "n_decisions": float(decisions),
        "n_gradient_updates": float(agent.n_updates),
        "n_target_syncs": float(agent.n_target_syncs),
        "n_episodes": float(episode),
        "reward_window": float(window),
    }
    return C51TrainResult(
        agent=agent,
        loss_curve=np.asarray(losses, dtype=np.float64),
        reward_curve=np.asarray(curve, dtype=np.float64),
        metrics=metrics,
        n_decisions=int(decisions),
        episodes=int(episode),
        seed=int(train_seed),
        label="SYNTHETIC",
    )


def compare_c51_baselines(
    *,
    agent: C51Agent,
    env_config: ZILobQuoteEnvConfig,
    seeds: Sequence[int],
    max_decisions: int | None = None,
    flow_factory: Callable[[int], MOFlow] | None = None,
    classic_params: dict[str, Any] | None = None,
    include_classic: bool = True,
) -> dict[str, Any]:
    """Paired-seed evaluation: trained C51 vs random vs AS/GLFT references.

    Every policy runs through the *same* :class:`ZILobQuoteEnv` step API on the
    same seeds, so the comparison holds the book, the reward and the inventory
    cap fixed. The trained agent is evaluated greedily (``act(..., greedy=True)``,
    no exploration, no learning). Metric keys are ``sim_internal_*``
    simulator-internal diagnostics on a SYNTHETIC engine — never headline
    metrics, never market evidence, no live-trading claim.

    ``classic_params`` overrides the closed-form reference calibration
    (``gamma``, ``sigma``, ``kappa``, ``a_fill``, ``tick``); the defaults give
    both references a ~0.5 tick half-spread and ~1 tick of inventory skew at
    the Santa Fe calibration, so they quote *inside* this book's typical
    2-tick spread and actually participate instead of snapping to the widest
    cell of the action grid.
    """
    if not isinstance(agent, C51Agent):
        raise TypeError("agent must be a C51Agent")
    if not isinstance(env_config, ZILobQuoteEnvConfig):
        raise TypeError("env_config must be a ZILobQuoteEnvConfig")
    sd = [int(s) for s in seeds]
    if not sd:
        raise ValueError("seeds must be non-empty")
    for s in sd:
        _seed_int(s)
    params: dict[str, Any] = {
        "gamma": 0.01,
        "sigma": 2.0 * float(env_config.lob.tick),
        "kappa": 2.0 / float(env_config.lob.tick),
        "a_fill": 2.7e-4,
        "tick": float(env_config.lob.tick),
    }
    if classic_params is not None:
        for key, value in dict(classic_params).items():
            if key not in params:
                raise ValueError(
                    f"unknown classic_params key {key!r}; expected a subset of {sorted(params)}"
                )
            params[key] = value
    shared = {k: params[k] for k in ("gamma", "sigma", "kappa", "tick")}
    policies: dict[str, EnvPolicy] = {"c51": agent_policy(agent, greedy=True)}
    if include_classic:
        policies["as"] = classic_action_policy(kind="as", **shared)
        policies["glft"] = classic_action_policy(kind="glft", a_fill=params["a_fill"], **shared)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
    rows["random"] = []
    for i, s in enumerate(sd):
        for name, pol in policies.items():
            env = ZILobQuoteEnv(env_config, flow_factory=flow_factory)
            rows[name].append(run_env_episode(env, pol, seed=s, max_decisions=max_decisions))
        rand_env = ZILobQuoteEnv(env_config, flow_factory=flow_factory)
        rand = random_action_policy(seed=s + 17 * (i + 1), n_actions=rand_env.n_actions)
        rows["random"].append(run_env_episode(rand_env, rand, seed=s, max_decisions=max_decisions))

    def _mean(name: str, key: str) -> float:
        vals = [float(r[key]) for r in rows[name]]
        return float(np.mean(vals)) if vals else float("nan")

    keys = (
        "sim_internal_reward_mean",
        "sim_internal_reward_sum",
        "sim_internal_mtm_pnl_final",
        "sim_internal_inventory_penalty_sum",
        "mean_abs_inventory",
        "max_abs_inventory",
        "n_fills",
    )
    metrics: dict[str, float] = {}
    for name in rows:
        for key in keys:
            metrics[f"{key}_{name}"] = _mean(name, key)
    for name in rows:
        if name == "c51":
            continue
        metrics[f"sim_internal_reward_gap_c51_minus_{name}_mean"] = (
            metrics["sim_internal_reward_mean_c51"] - metrics[f"sim_internal_reward_mean_{name}"]
        )
    out: dict[str, Any] = _synthetic_envelope(sd[0])
    out.update(
        {
            "kind": "c51_vs_baseline_comparison",
            "n_seeds": float(len(sd)),
            "seeds": sd,
            "policies": sorted(rows),
            "horizon": float(env_config.horizon),
            "decision_interval": float(env_config.decision_interval),
            "inventory_cap": int(env_config.inventory_cap),
            "max_decisions": None if max_decisions is None else int(max_decisions),
            "classic_params": params if include_classic else None,
            "metrics": metrics,
            "rows": rows,
            "note": (
                "sim_internal_* keys are simulator-internal diagnostics of a SYNTHETIC "
                "zero-intelligence book — never headline metrics, never market evidence."
            ),
        }
    )
    return out
