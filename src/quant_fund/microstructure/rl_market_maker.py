"""Distributional-RL market maker (C51) on the ZI limit-order-book simulator.

**Labeled SYNTHETIC** research infrastructure (lane B4-ii): a categorical
distributional DQN (C51) market maker trained and evaluated inside the lane
B4-i zero-intelligence LOB (``microstructure.zi_lob_simulator``), following

- Moret, Lillo (2026). Deep learning of robust market making under
  regime-switching order flow. arXiv:2609.11614 (verified 2026-09-30 against
  the arXiv abstract + HTML full text). The paper trains a Rainbow-style
  distributional DQN in a ZI (Santa Fe) LOB; under stationary flow it beats
  GLFT across the observed risk-return frontier; under regime-switching flow
  a stationarily trained controller saturates inventory, and augmenting the
  state with (a) a Bayesian online change-point filter over the directional
  flow bias (Appendix A: Beta-Bernoulli run-length posterior) and (b) a
  queue-adjusted quote-exposure imbalance (Eqs. 12-13) restores profitability;
  an optional scenario-bandit step (their Algorithm C) is implemented in
  ``microstructure.scenario_bandit`` — a finite pool of exogenous regime
  plans sampled by difficulty-weighted softmax bandit during fine-tuning.
- Bellemare, Dabney, Munos (2017). A distributional perspective on
  reinforcement learning. *ICML 2017*, arXiv:1707.06887 — C51 atoms,
  categorical projection, cross-entropy loss.
- Hessel et al. (2018). Rainbow: combining improvements in deep RL. *AAAI*,
  arXiv:1710.02298 — the paper's backbone; see deviations below.
- van Hasselt, Guez, Silver (2016). Deep RL with double Q-learning. *AAAI*,
  arXiv:1509.06461. Wang et al. (2016). Dueling network architectures.
  *ICML*, arXiv:1511.06581. Schaul et al. (2016). Prioritized experience
  replay. *ICLR*, arXiv:1511.05952 (cited; NOT used — see deviations).
- Adams, MacKay (2007). Bayesian online changepoint detection.
  arXiv:0710.3742 — the run-length framework behind :class:`FlowBiasFilter`.

Paper formulation implemented here (arXiv:2609.11614 Sec. 3, Appendix A):

- **Decision clock**: throttled SMDP; one decision per ``decision_interval``
  simulated seconds (paper tau_c = 1 s), effective per-decision discount
  Gamma_t = gamma_event**N_t with N_t the number of book events in the
  interval (their Eqs. 9, 23-24).
- **State** (their Eq. 10, K=1, plus the auxiliary vector e_t of Sec. 6):
  touch spread, bid/ask depths at the touch and one level behind, touch depth
  imbalance, normalized inventory, and the MM's own queue-adjusted quote
  opportunities O_b/O_a (their Eq. 12). When ``aux_enabled``: the flow-bias
  belief (iota_hat, ell_bar) from the Bayesian filter and the quote-exposure
  imbalance x_t (their Eq. 13). See :func:`build_state_vector`.
- **Action grid** (their Eq. 15): six tick-offset pairs (delta_bid,
  delta_ask) relative to the current best quotes::

      bid = best_bid - delta_bid * tick,  ask = best_ask + delta_ask * tick
      A = {(-1,-1), (-1,0), (0,-1), (0,0), (0,+1), (+1,0)}

  delta = -1 posts one tick *inside* the touch (new best quote, front of a
  fresh FIFO queue); 0 joins the current best (back of its queue); +1 steps
  one tick deeper. The three extreme cells {(+1,+1), (+1,-1), (-1,+1)} are
  excluded per the paper's ablation (``FULL_ACTION_GRID`` keeps them for
  experiments). A safety layer clamps self-crossing quotes back to the touch
  and a post-hoc projection drops the bid (ask) leg at inventory >= +cap
  (<= -cap), so |inventory| <= cap holds for any learned policy.
- **Reward** (their Eq. 16, simulator-internal): fills revalued against the
  next-decision mid minus a quadratic inventory penalty and a one-sided
  inventory-wall penalty beyond ``wall_fraction * cap``. This is a
  ``sim_internal_*`` training signal, NEVER a headline metric.
- **Smart quoting**: resting orders persist across decisions; an order is
  canceled and reposted only when its target price level changes, so FIFO
  queue priority is treated as an economic resource (paper Sec. 3).

Deviations from the paper (deliberate, lane-brief driven):

1. **Epsilon-greedy exploration** (linear decay) instead of NoisyNet — the
   lane brief specifies epsilon-greedy; it is deterministic under a seed.
2. **Uniform replay** instead of prioritized experience replay — the brief
   specifies a replay buffer; PER is an orthogonal add-on left out to keep
   the module minimal and bit-deterministic.
3. **Single-stage auxiliary features**: the belief/exposure features are on
   during both stationary and regime training (the paper trains Algorithm A
   with e_t off, then fine-tunes Algorithm B with e_t on; that two-stage
   warm-start would require reshaping the input layer). Under stationary
   symmetric flow iota_hat stays near 0, so the features are inert there.
4. **FlowBiasFilter** implements the paper's Appendix A Beta-Bernoulli online
   filter — the Bernoulli-conjugate, truly incremental (O(max_run) per market
   order) sibling of ``models.changepoint.bocpd_gaussian`` (same Adams-MacKay
   run-length recursion, Gaussian Normal-Gamma batch form). The batch Gaussian
   BOCPD cannot serve as the per-MO online belief, so it is not reused as a
   code path; tests cross-check that both detectors localize the same change.

Composition: this module **composes** ``zi_lob_simulator`` without modifying
it — the event engine (:class:`~quant_fund.microstructure.zi_lob_simulator.
ZILobSimulator`), :class:`MarkovRegimeFlow`, the AS/GLFT policy adapters,
``book_phase_metrics`` and ``regime_flow_diagnostics`` are reused as the
single source of truth. :func:`run_rl_mm_session` is a ``run_mm_session``-
style runner (same accounting and honesty schema) extended with smart
quoting, action offsets, RL rewards/transitions, and the flow filter.

Honesty: every output is a SYNTHETIC correctness diagnostic on a synthetic
engine, never market evidence. All P&L-like keys are namespaced
``sim_internal_*``, must never be headlined, and there is no broker
connectivity or live-trading claim anywhere in this module. Tiny-budget tests
assert plumbing and bounded inventory, NOT that the trained policy beats
GLFT; :func:`rl_mm_benchmark` is the documented-optional full comparison for
the bench battery (long-running by design, excluded from the PR gate).

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models.deep_hedging``), so this module imports
cleanly without torch and every torch entry point raises ``ImportError`` with
install guidance. The numpy core (filter, buffer, state encoding, action
grid) is torch-free. Training is CPU single-thread and deterministic given
``seed`` (seeded torch + numpy; GPU determinism is not claimed).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MM_TAG,
    ZI_LOB_REVISION,
    AdversarialFlow,
    MarkovRegimeFlow,
    MMState,
    QuotePolicy,
    RegimeState,
    ScenarioRegimeFlow,
    Side,
    ZILobConfig,
    ZILobSimulator,
    as_policy,
    book_phase_metrics,
    glft_policy,
    regime_flow_diagnostics,
    santa_fe_config,
)

Array = NDArray[np.float64]

__all__ = [
    "FULL_ACTION_GRID",
    "PAPER_ACTION_GRID",
    "RL_MM_REVISION",
    "C51Config",
    "C51MarketMaker",
    "FlowBiasFilter",
    "RLStateSpec",
    "ReplayBuffer",
    "build_state_vector",
    "evaluate_rl_market_makers",
    "paper_regime_flow",
    "quote_exposure_imbalance",
    "rl_mm_benchmark",
    "run_rl_mm_session",
    "train_c51_market_maker",
    "validate_action_grid",
]

RL_MM_REVISION = "SYNTHETIC_C51_MM_v1"

# Base state features (always on) and auxiliary features (aux_enabled).
N_BASE_STATE_FEATURES = 9
N_AUX_STATE_FEATURES = 3

# Clip caps for normalized features (keep degenerate books from producing
# outlier inputs; deterministic, documented).
_SPREAD_NORM_CLIP = 8.0
_DEPTH_NORM_CLIP = 8.0
_RUNLEN_NORM_CLIP = 5.0


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (local; zi_lob_simulator's are private)
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


def _torch() -> Any:
    """Lazily import torch (optional ``nn`` extra); fail closed with guidance."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "the C51 RL market maker needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Action grid (paper Eq. 15)
# ---------------------------------------------------------------------------

ActionOffset = tuple[int, int]

#: The paper's six-action grid (arXiv:2609.11614 Eq. 15): tick offsets
#: ``(delta_bid, delta_ask)`` relative to the current best quotes. ``-1`` posts
#: one tick inside the touch, ``0`` joins the best quote, ``+1`` steps one tick
#: deeper. The extreme cells ``(+1,+1)``, ``(+1,-1)``, ``(-1,+1)`` are removed
#: per the paper's action-space ablation (they converge worse under
#: limited-budget non-stationary fine-tuning).
PAPER_ACTION_GRID: tuple[ActionOffset, ...] = (
    (-1, -1),
    (-1, 0),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, 0),
)

#: The full one-tick grid {-1, 0, +1}^2 (paper's nine-action ablation arm).
FULL_ACTION_GRID: tuple[ActionOffset, ...] = tuple(
    (db, da) for db in (-1, 0, 1) for da in (-1, 0, 1)
)


def validate_action_grid(grid: Sequence[ActionOffset]) -> tuple[ActionOffset, ...]:
    """Validate a quoting action grid; fail closed on malformed cells."""
    cells = tuple(grid)
    if len(cells) < 1:
        raise ValueError("action grid must contain at least one action")
    seen: set[ActionOffset] = set()
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise ValueError(
                f"action grid cells must be (delta_bid, delta_ask) pairs, got {cell!r}"
            )
        db, da = cell
        for v in (db, da):
            if isinstance(v, bool) or not isinstance(v, int) or abs(v) > 1:
                raise ValueError(f"action offsets must be ints in [-1, 1], got {cell!r}")
        pair: ActionOffset = (int(db), int(da))
        if pair in seen:
            raise ValueError(f"duplicate action grid cell {pair!r}")
        seen.add(pair)
    return cells


# ---------------------------------------------------------------------------
# Bayesian online flow-bias filter (paper Appendix A)
# ---------------------------------------------------------------------------


class FlowBiasFilter:
    """Beta-Bernoulli Bayesian online change-point filter over signed MO flow.

    Moret & Lillo (2026, arXiv:2609.11614) Appendix A, adapting Adams &
    MacKay (2007) run-length detection to the binary market-order side stream
    ``y_t`` (1 = buy, 0 = sell) observed on the MO clock:

    - within a directional segment, sides are i.i.d. Bernoulli(p) with a
      Beta(a0, b0) prior; a segment of run length ``ell`` has posterior
      Beta(a0 + buys(ell), b0 + sells(ell)) by conjugacy;
    - constant reset hazard ``h = 1 - exp(-1/tau_r)`` from Exp(1/tau_r)
      segment durations (their Eq. 25);
    - per observation the run-length posterior g(ell) is advanced by the
      continuation branch (predictive E = mu(ell) or 1 - mu(ell)) and the
      change-point branch (fresh segment scored under the prior predictive),
      then normalized.

    :meth:`belief` returns the controller summary ``b_t = (iota_hat, ell_bar)``
    (their Eqs. 11, A.3): the posterior directional flow bias ``iota_hat =
    2*p_hat - 1`` and the posterior expected run length ``ell_bar``. The run
    length is truncated at ``max_run`` (continuation mass beyond the cap is
    dropped and the posterior renormalized; change-point mass sees the full
    posterior). Deterministic: pure numpy arithmetic, no RNG.

    Relationship to ``models.changepoint.bocpd_gaussian``: same Adams-MacKay
    run-length framework, but Bernoulli-conjugate (correct for binary MO
    sides) and truly incremental — O(max_run) per update instead of a batch
    O(n^2) pass — so it can run inside a training loop. The Gaussian batch
    BOCPD is not reused as a code path; ``tests`` cross-check that both
    detectors localize the same changepoint on a sticky synthetic stream.
    """

    def __init__(
        self,
        *,
        tau_r: float = 60.0,
        a0: float = 1.0,
        b0: float = 1.0,
        max_run: int | None = None,
        hazard: float | None = None,
    ) -> None:
        t = _pos_finite(tau_r, "tau_r")
        self._a0 = _pos_finite(a0, "a0")
        self._b0 = _pos_finite(b0, "b0")
        if hazard is None:
            h = 1.0 - math.exp(-1.0 / t)
        else:
            h = float(hazard)
            if not math.isfinite(h) or not 0.0 < h < 1.0:
                raise ValueError(f"hazard must lie in (0, 1), got {hazard!r}")
        self._hazard = h
        if max_run is None:
            cap = int(min(4096, max(64, math.ceil(8.0 * t))))
        else:
            cap = _int_at_least(max_run, 1, "max_run")
        self._max_run = cap
        self._g = np.zeros(cap + 1, dtype=np.float64)
        self._g[0] = 1.0
        # Cumulative buy/sell counts; index t = number of MOs observed.
        self._cum_buys: list[int] = [0]
        self._cum_sells: list[int] = [0]
        self.n_mo = 0

    @property
    def hazard(self) -> float:
        return self._hazard

    @property
    def max_run(self) -> int:
        return self._max_run

    def observe(self, y: int) -> None:
        """Absorb one market-order side (1 = buy, 0 = sell); fail closed."""
        if isinstance(y, bool) or not isinstance(y, int) or y not in (0, 1):
            raise ValueError(f"y must be an int 0 (sell) or 1 (buy), got {y!r}")
        self._cum_buys.append(self._cum_buys[-1] + y)
        self._cum_sells.append(self._cum_sells[-1] + (1 - y))
        t = self.n_mo  # MOs observed before this one
        hi = min(self._max_run - 1, t)  # largest run length that can continue
        ells = np.arange(0, hi + 1)
        cb = np.asarray(self._cum_buys, dtype=np.float64)
        cs = np.asarray(self._cum_sells, dtype=np.float64)
        a_par = self._a0 + (cb[t] - cb[t - ells])
        b_par = self._b0 + (cs[t] - cs[t - ells])
        mu = a_par / (a_par + b_par)
        evidence = mu if y == 1 else 1.0 - mu
        g_new = np.zeros(self._max_run + 1, dtype=np.float64)
        g_new[1 : hi + 2] += self._g[: hi + 1] * (1.0 - self._hazard) * evidence
        prior_ev = self._a0 / (self._a0 + self._b0) if y == 1 else self._b0 / (self._a0 + self._b0)
        g_new[1] += self._hazard * prior_ev * float(self._g.sum())
        total = float(g_new.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("flow-bias filter posterior collapsed (non-finite or zero mass)")
        g_new /= total
        self._g = g_new
        self.n_mo = t + 1

    def belief(self) -> tuple[float, float]:
        """Controller summary ``(iota_hat, ell_bar)``; prior (0.0, 0.0)-ish
        before any MO (iota_hat = 2*a0/(a0+b0) - 1 with the uniform prior)."""
        t = self.n_mo
        ells = np.arange(0, self._max_run + 1)
        idx = np.clip(t - ells, 0, t)  # g has support only on ell <= t
        cb = np.asarray(self._cum_buys, dtype=np.float64)
        cs = np.asarray(self._cum_sells, dtype=np.float64)
        a_par = self._a0 + (cb[t] - cb[idx])
        b_par = self._b0 + (cs[t] - cs[idx])
        mu = a_par / (a_par + b_par)
        p_hat = float(np.dot(self._g, mu))
        ell_bar = float(np.dot(self._g, ells.astype(np.float64)))
        if not (math.isfinite(p_hat) and math.isfinite(ell_bar)):
            raise RuntimeError("flow-bias filter belief went non-finite")
        return (2.0 * p_hat - 1.0, ell_bar)

    def changepoint_mass(self) -> float:
        """Posterior mass on run length 1 (a change at the latest MO)."""
        return float(self._g[1]) if self.n_mo > 0 else 0.0


# ---------------------------------------------------------------------------
# Queue-adjusted quote-exposure imbalance (paper Eqs. 12-13)
# ---------------------------------------------------------------------------


def _opportunity(alive: bool, queue_ahead: int | None, name: str) -> float:
    if not alive:
        if queue_ahead is not None:
            raise ValueError(f"{name}_queue_ahead must be None when {name}_alive is False")
        return 0.0
    if queue_ahead is None:
        raise ValueError(f"{name}_queue_ahead is required when {name}_alive is True")
    qa = _int_at_least(queue_ahead, 0, f"{name}_queue_ahead")
    return 1.0 / (1.0 + qa)


def _opportunity_imbalance(o_bid: float, o_ask: float) -> float:
    return (o_bid - o_ask) / (1.0 + o_bid + o_ask)


def quote_exposure_imbalance(
    *,
    bid_alive: bool,
    bid_queue_ahead: int | None,
    ask_alive: bool,
    ask_queue_ahead: int | None,
) -> float:
    """Queue-adjusted quote-exposure imbalance x_t (paper Eqs. 12-13).

    ``O = 1{alive} / (1 + queue_ahead)`` per side, ``x = (O_b - O_a) / (1 +
    O_b + O_a)`` in ``[-1/2, 1/2]``: +1/2 when only a front-of-queue bid is
    alive, -1/2 for a front-of-queue ask only, near 0 when balanced or both
    buried. Tells the policy which side of the book is more vulnerable before
    the next market order arrives. Fail closed on invalid queue counts or
    alive/queue mismatches.
    """
    o_b = _opportunity(bool(bid_alive), bid_queue_ahead, "bid")
    o_a = _opportunity(bool(ask_alive), ask_queue_ahead, "ask")
    return _opportunity_imbalance(o_b, o_a)


# ---------------------------------------------------------------------------
# State specification + encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RLStateSpec:
    """State-vector specification for the C51 market maker.

    ``inventory_cap`` (the paper's hard bound q_max) doubles as the inventory
    normalization scale and the session gating cap; sessions fail closed when
    it disagrees with their ``inventory_cap``. ``filter_*`` configure the
    :class:`FlowBiasFilter` used for the auxiliary belief features.
    """

    aux_enabled: bool = True
    filter_tau_r: float = 60.0
    filter_a0: float = 1.0
    filter_b0: float = 1.0
    filter_max_run: int | None = None
    inventory_cap: int = 8
    depth_scale: float = 10.0
    spread_scale: float = 4.0

    def __post_init__(self) -> None:
        _pos_finite(self.filter_tau_r, "filter_tau_r")
        _pos_finite(self.filter_a0, "filter_a0")
        _pos_finite(self.filter_b0, "filter_b0")
        if self.filter_max_run is not None:
            _int_at_least(self.filter_max_run, 1, "filter_max_run")
        _int_at_least(self.inventory_cap, 1, "inventory_cap")
        _pos_finite(self.depth_scale, "depth_scale")
        _pos_finite(self.spread_scale, "spread_scale")

    @property
    def state_dim(self) -> int:
        return N_BASE_STATE_FEATURES + (N_AUX_STATE_FEATURES if self.aux_enabled else 0)


def build_state_vector(
    spec: RLStateSpec,
    *,
    spread_ticks: int,
    bid_depth0: int,
    bid_depth1: int,
    ask_depth0: int,
    ask_depth1: int,
    inventory: int,
    bid_opportunity: float,
    ask_opportunity: float,
    belief: tuple[float, float] | None = None,
) -> Array:
    """Encode decision-time observables into the agent state vector.

    Base features (9): normalized touch spread; bid/ask depths at the touch
    and one level behind (normalized); touch depth imbalance; inventory /
    cap; own bid/ask queue opportunities O_b, O_a (paper Eq. 12). Auxiliary
    features (3, ``aux_enabled`` only): flow-bias belief ``(iota_hat,
    ell_bar/tau_r)`` (paper Eq. 11) and the quote-exposure imbalance x_t
    (paper Eq. 13, recomputed from the opportunities). Deterministic; fails
    closed on out-of-domain inputs.
    """
    if not isinstance(spec, RLStateSpec):
        raise TypeError("spec must be an RLStateSpec")
    spr = _int_at_least(spread_ticks, 1, "spread_ticks")
    d0b = _int_at_least(bid_depth0, 0, "bid_depth0")
    d1b = _int_at_least(bid_depth1, 0, "bid_depth1")
    d0a = _int_at_least(ask_depth0, 0, "ask_depth0")
    d1a = _int_at_least(ask_depth1, 0, "ask_depth1")
    q = int(inventory)
    if isinstance(inventory, bool) or abs(q) > spec.inventory_cap:
        raise ValueError(
            f"inventory must satisfy |q| <= cap={spec.inventory_cap}, got {inventory!r}"
        )
    o_b = _prob(bid_opportunity, "bid_opportunity")
    o_a = _prob(ask_opportunity, "ask_opportunity")
    feats = [
        min(spr / spec.spread_scale, _SPREAD_NORM_CLIP),
        min(d0b / spec.depth_scale, _DEPTH_NORM_CLIP),
        min(d1b / spec.depth_scale, _DEPTH_NORM_CLIP),
        min(d0a / spec.depth_scale, _DEPTH_NORM_CLIP),
        min(d1a / spec.depth_scale, _DEPTH_NORM_CLIP),
        (d0b - d0a) / (d0b + d0a + 1.0),
        q / float(spec.inventory_cap),
        o_b,
        o_a,
    ]
    if spec.aux_enabled:
        if belief is None:
            raise ValueError("belief=(iota_hat, ell_bar) is required when aux_enabled")
        if len(belief) != 2 or not all(math.isfinite(float(v)) for v in belief):
            raise ValueError(f"belief must be two finite floats, got {belief!r}")
        iota = min(max(float(belief[0]), -1.0), 1.0)
        ell_n = min(max(float(belief[1]), 0.0) / spec.filter_tau_r, _RUNLEN_NORM_CLIP)
        feats += [iota, ell_n / _RUNLEN_NORM_CLIP, _opportunity_imbalance(o_b, o_a)]
    elif belief is not None:
        raise ValueError("belief must be None when aux_enabled is False")
    out = np.asarray(feats, dtype=np.float64)
    if not np.all(np.isfinite(out)):  # pragma: no cover - inputs validated above
        raise RuntimeError("state vector went non-finite")
    return out


# ---------------------------------------------------------------------------
# Uniform replay buffer (numpy core; torch-free)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer of SMDP transitions.

    Stores ``(state, action, reward, next_state, gamma_eff, done)`` with
    ``gamma_eff`` the effective per-decision discount ``gamma_event**N_t``
    (paper Sec. 3, SMDP form). Ring semantics: once full, the oldest
    transition is overwritten. Sampling is without replacement from a seeded
    ``numpy`` Generator, so identical push/sample sequences are bit-identical.
    """

    def __init__(self, capacity: int, state_dim: int, *, seed: int = 0) -> None:
        cap = _int_at_least(capacity, 1, "capacity")
        dim = _int_at_least(state_dim, 1, "state_dim")
        _seed_int(seed)
        self._capacity = cap
        self._dim = dim
        self._states = np.zeros((cap, dim), dtype=np.float32)
        self._next_states = np.zeros((cap, dim), dtype=np.float32)
        self._actions = np.zeros(cap, dtype=np.int64)
        self._rewards = np.zeros(cap, dtype=np.float32)
        self._gammas = np.zeros(cap, dtype=np.float32)
        self._dones = np.zeros(cap, dtype=bool)
        self._rng = np.random.default_rng(seed)
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def _check_vec(self, v: Array, name: str) -> Array:
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.shape != (self._dim,):
            raise ValueError(f"{name} must have shape ({self._dim},), got {np.shape(v)}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite")
        return arr

    def push(
        self,
        state: Array,
        action: int,
        reward: float,
        next_state: Array,
        gamma_eff: float,
        done: bool,
    ) -> None:
        s = self._check_vec(state, "state")
        n = self._check_vec(next_state, "next_state")
        a = int(action)
        if isinstance(action, bool) or a < 0:
            raise ValueError(f"action must be an int >= 0, got {action!r}")
        r = float(reward)
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        g = float(gamma_eff)
        if not math.isfinite(g) or not 0.0 < g <= 1.0:
            raise ValueError(f"gamma_eff must lie in (0, 1], got {gamma_eff!r}")
        i = self._pos
        self._states[i] = s.astype(np.float32)
        self._next_states[i] = n.astype(np.float32)
        self._actions[i] = a
        self._rewards[i] = r
        self._gammas[i] = g
        self._dones[i] = bool(done)
        self._pos = (i + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> dict[str, NDArray[Any]]:
        b = _int_at_least(batch_size, 1, "batch_size")
        if b > self._size:
            raise ValueError(f"cannot sample {b} transitions from a buffer of {self._size}")
        valid = self._capacity if self._size == self._capacity else self._size
        idx = self._rng.choice(valid, size=b, replace=False)
        return {
            "states": self._states[idx].copy(),
            "actions": self._actions[idx].copy(),
            "rewards": self._rewards[idx].copy(),
            "next_states": self._next_states[idx].copy(),
            "gammas": self._gammas[idx].copy(),
            "dones": self._dones[idx].copy(),
        }


# ---------------------------------------------------------------------------
# C51 agent (torch-gated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class C51Config:
    """C51 hyperparameters. Defaults follow the paper's Table 1 where the
    tiny-budget lane allows (paper: 101 atoms on [-3, 3], hidden 256,
    gamma_event = 1 - 1e-3, n_step = 3, AdamW lr 3e-4, batch 64, buffer 1e5,
    target sync every 2000 gradient steps). Exploration is epsilon-greedy
    (documented deviation from the paper's NoisyNet)."""

    n_atoms: int = 51
    v_min: float = -3.0
    v_max: float = 3.0
    hidden: tuple[int, ...] = (256,)
    gamma_event: float = 1.0 - 1e-3
    n_step: int = 3
    lr: float = 3e-4
    batch_size: int = 64
    buffer_capacity: int = 100_000
    target_update_every: int = 2_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    dueling: bool = True
    double_dqn: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        _int_at_least(self.n_atoms, 2, "n_atoms")
        lo = float(self.v_min)
        hi = float(self.v_max)
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError(f"require v_min < v_max finite, got ({self.v_min!r}, {self.v_max!r})")
        widths = tuple(int(h) for h in self.hidden)
        if not widths or any(h < 1 for h in widths):
            raise ValueError(
                f"hidden must be a non-empty sequence of positive widths, got {self.hidden!r}"
            )
        g = float(self.gamma_event)
        if not math.isfinite(g) or not 0.0 < g < 1.0:
            raise ValueError(f"gamma_event must lie in (0, 1), got {self.gamma_event!r}")
        _int_at_least(self.n_step, 1, "n_step")
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
        _seed_int(self.seed)


_PendingItem = tuple[Array, int, float, Array, float, bool]


class C51MarketMaker:
    """Categorical distributional DQN (C51) quoting agent for the ZI-LOB.

    A dueling MLP trunk maps the state vector (see :func:`build_state_vector`)
    to per-action categorical return distributions over ``n_atoms`` atoms on
    ``[v_min, v_max]``; Q(s, a) is the distribution mean (paper Eq. 21).
    Learning minimizes the categorical cross-entropy against the projected
    Bellman target (C51 projection Phi, paper Eq. 22): target atoms are
    shifted by ``r + gamma_eff * z`` (double-DQN action selection from the
    online net, evaluation from the target net), clipped to the support, and
    redistributed onto the neighboring atoms. ``n_step`` decision-level
    returns are folded with the SMDP event-clock discounts before storage.

    Exploration is epsilon-greedy with a linear decay over
    ``eps_decay_steps`` non-greedy decisions, drawn from a seeded numpy
    Generator; ``act(..., greedy=True)`` is the deterministic evaluation
    policy (argmax Q, first-index tie-break). Given ``seed``, torch and numpy
    randomness are fully seeded and CPU training is single-threaded, so
    identical call sequences give bit-identical results.
    """

    def __init__(
        self,
        state_spec: RLStateSpec,
        config: C51Config,
        action_grid: Sequence[ActionOffset] = PAPER_ACTION_GRID,
    ) -> None:
        torch = _torch()
        if not isinstance(state_spec, RLStateSpec):
            raise TypeError("state_spec must be an RLStateSpec")
        if not isinstance(config, C51Config):
            raise TypeError("config must be a C51Config")
        self._spec = state_spec
        self._cfg = config
        self._grid = validate_action_grid(action_grid)
        self._n_actions = len(self._grid)
        self._state_dim = state_spec.state_dim
        self._torch = torch
        torch.manual_seed(int(config.seed))
        torch.set_num_threads(1)
        self._rng = np.random.default_rng(config.seed + 7919)
        self._atoms_np = np.linspace(config.v_min, config.v_max, config.n_atoms)
        self._atoms = torch.as_tensor(self._atoms_np, dtype=torch.float32)
        self._dz = (config.v_max - config.v_min) / (config.n_atoms - 1)
        self._online = self._new_net()
        self._target = self._new_net()
        self.sync_target()
        params: list[Any] = []
        for mod in self._online.values():
            params.extend(mod.parameters())
        self._opt = torch.optim.AdamW(params, lr=float(config.lr))
        self._buffer = ReplayBuffer(
            config.buffer_capacity, self._state_dim, seed=config.seed + 104_729
        )
        self._pending: deque[_PendingItem] = deque()
        self._n_updates = 0
        self._n_decisions = 0
        self._total_stored = 0

    # -- construction helpers ------------------------------------------------

    def _new_net(self) -> dict[str, Any]:
        torch = self._torch
        layers: list[Any] = []
        d = self._state_dim
        for h in self._cfg.hidden:
            layers.append(torch.nn.Linear(d, int(h)))
            layers.append(torch.nn.ReLU())
            d = int(h)
        return {
            "trunk": torch.nn.Sequential(*layers),
            "value": torch.nn.Linear(d, int(self._cfg.n_atoms)),
            "adv": torch.nn.Linear(d, int(self._n_actions) * int(self._cfg.n_atoms)),
        }

    def _z_logits(self, net: dict[str, Any], x: Any) -> Any:
        h = net["trunk"](x)
        adv = net["adv"](h).view(-1, self._n_actions, self._cfg.n_atoms)
        if not self._cfg.dueling:
            return adv
        val = net["value"](h).unsqueeze(1)
        return val + adv - adv.mean(dim=1, keepdim=True)

    # -- introspection --------------------------------------------------------

    @property
    def spec(self) -> RLStateSpec:
        return self._spec

    @property
    def config(self) -> C51Config:
        return self._cfg

    @property
    def action_grid(self) -> tuple[ActionOffset, ...]:
        return self._grid

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def atoms(self) -> Array:
        return self._atoms_np.copy()

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    @property
    def total_stored(self) -> int:
        return self._total_stored

    @property
    def epsilon(self) -> float:
        """Current epsilon of the linear decay schedule."""
        cfg = self._cfg
        frac = min(1.0, self._n_decisions / float(cfg.eps_decay_steps))
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * (1.0 - frac)

    # -- policy ----------------------------------------------------------------

    def _check_state(self, state: Array) -> Array:
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.shape != (self._state_dim,):
            raise ValueError(f"state must have shape ({self._state_dim},), got {np.shape(state)}")
        if not np.all(np.isfinite(x)):
            raise ValueError("state must be finite")
        return x

    def q_values(self, state: Array) -> Array:
        """Per-action Q values (distribution means), eval-mode no-grad."""
        x = self._check_state(state)
        torch = self._torch
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
            z = self._z_logits(self._online, xt)
            p = torch.softmax(z, dim=-1)
            q = (p * self._atoms.unsqueeze(0)).sum(dim=-1)
            out = q.squeeze(0).detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64)

    def act(self, state: Array, *, greedy: bool = False) -> int:
        """Pick an action index; epsilon-greedy unless ``greedy``.

        Non-greedy calls advance the epsilon schedule (one counted decision);
        greedy evaluation calls leave the agent state untouched.
        """
        x = self._check_state(state)
        if greedy:
            return int(np.argmax(self.q_values(x)))
        eps = self.epsilon
        self._n_decisions += 1
        if float(self._rng.random()) < eps:
            return int(self._rng.integers(self._n_actions))
        return int(np.argmax(self.q_values(x)))

    # -- experience -------------------------------------------------------------

    def begin_episode(self) -> None:
        """Drop any pending n-step window (episode boundary; nothing folds
        across episodes)."""
        self._pending.clear()

    def store_transition(
        self,
        state: Array,
        action: int,
        reward: float,
        next_state: Array,
        gamma_eff: float,
        done: bool,
    ) -> None:
        """Accumulate one decision-level transition; fold n-step windows.

        Windows of ``n_step`` transitions are folded into a single buffer
        entry (rewards chained with the per-decision SMDP discounts). A
        ``done`` transition flushes every pending suffix window so no
        in-episode experience is dropped.
        """
        s = self._check_state(state)
        n = self._check_state(next_state)
        a = int(action)
        if isinstance(action, bool) or not 0 <= a < self._n_actions:
            raise ValueError(f"action must be an int in [0, {self._n_actions}), got {action!r}")
        r = float(reward)
        if not math.isfinite(r):
            raise ValueError(f"reward must be finite, got {reward!r}")
        g = float(gamma_eff)
        if not math.isfinite(g) or not 0.0 < g <= 1.0:
            raise ValueError(f"gamma_eff must lie in (0, 1], got {gamma_eff!r}")
        item: _PendingItem = (s, a, r, n, g, bool(done))
        self._pending.append(item)
        if item[5]:
            pending = list(self._pending)
            self._pending.clear()
            for i in range(len(pending)):
                self._fold_push(pending[i:])
        elif len(self._pending) >= self._cfg.n_step:
            window = [self._pending[i] for i in range(self._cfg.n_step)]
            self._fold_push(window)
            self._pending.popleft()

    def _fold_push(self, items: Sequence[_PendingItem]) -> None:
        r_acc = 0.0
        g_acc = 1.0
        for _, _, r, _, g, _ in items:
            r_acc += g_acc * r
            g_acc *= g
        s0, a0 = items[0][0], items[0][1]
        s_next, done = items[-1][3], items[-1][5]
        self._buffer.push(s0, a0, r_acc, s_next, g_acc, done)
        self._total_stored += 1

    # -- learning ----------------------------------------------------------------

    def sync_target(self) -> None:
        """Hard-copy online parameters into the target network."""
        for key, mod in self._online.items():
            self._target[key].load_state_dict(mod.state_dict())

    def learn(self) -> float | None:
        """One C51 gradient step; ``None`` while the buffer is below batch size.

        Categorical projection (Bellemare et al. 2017): target atoms are
        mapped through ``Tz = clamp(r + gamma_eff * (1 - done) * z, v_min,
        v_max)`` and their masses redistributed onto the bracketing atoms of
        the fixed support; the online net minimizes the cross-entropy between
        its predicted distribution for the taken action and that target.
        Double-DQN selects the target action with the online net. The target
        network is hard-synced every ``target_update_every`` updates.
        """
        cfg = self._cfg
        if len(self._buffer) < cfg.batch_size:
            return None
        torch = self._torch
        batch = self._buffer.sample(cfg.batch_size)
        s = torch.as_tensor(batch["states"])
        a = torch.as_tensor(batch["actions"], dtype=torch.long)
        r = torch.as_tensor(batch["rewards"])
        s2 = torch.as_tensor(batch["next_states"])
        gam = torch.as_tensor(batch["gammas"])
        done = torch.as_tensor(batch["dones"], dtype=torch.float32)
        atoms = self._atoms
        n_atoms = int(cfg.n_atoms)

        z = self._z_logits(self._online, s)
        log_p = torch.log_softmax(z, dim=-1)
        log_p_a = log_p.gather(1, a.view(-1, 1, 1).expand(-1, -1, n_atoms)).squeeze(1)
        with torch.no_grad():
            p2_target = torch.softmax(self._z_logits(self._target, s2), dim=-1)
            if cfg.double_dqn:
                z2_online = self._z_logits(self._online, s2)
                q2 = (torch.softmax(z2_online, dim=-1) * atoms).sum(dim=-1)
            else:
                q2 = (p2_target * atoms).sum(dim=-1)
            a_star = q2.argmax(dim=1)
            p2 = p2_target.gather(1, a_star.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)
            gamma_m = (gam * (1.0 - done)).unsqueeze(1)
            tz = torch.clamp(r.unsqueeze(1) + gamma_m * atoms.unsqueeze(0), cfg.v_min, cfg.v_max)
            b = (tz - cfg.v_min) / self._dz
            lo = b.floor().long().clamp(0, n_atoms - 2)
            hi = lo + 1
            lo_f = lo.to(torch.float32)
            hi_f = hi.to(torch.float32)
            m = torch.zeros_like(p2)
            m.scatter_add_(1, lo, p2 * (hi_f - b))
            m.scatter_add_(1, hi, p2 * (b - lo_f))
        loss = -(m * log_p_a).sum(dim=-1).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "C51 cross-entropy loss went non-finite; check reward scale vs support"
            )
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        self._n_updates += 1
        if self._n_updates % cfg.target_update_every == 0:
            self.sync_target()
        return float(loss.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# Regime flow helper (paper Sec. 5 generative model, two-state approximation)
# ---------------------------------------------------------------------------


def paper_regime_flow(
    *,
    seed: int,
    tau_r: float = 60.0,
    omega: float = 0.30,
    intensity_mult: float = 1.0,
) -> MarkovRegimeFlow:
    """Two-state Markov regime flow approximating the paper's Sec. 5 model.

    The paper draws i.i.d. regime durations ``L_k ~ Exp(1/tau_r)`` on the MO
    clock with ``p_buy,k ~ U[0.5 - omega, 0.5 + omega]``. The lane B4-i
    :class:`MarkovRegimeFlow` is the two-state fixed-width approximation:
    states ``(0.5 + omega, 0.5 - omega)`` with stay probability
    ``exp(-1/tau_r)`` (exponential durations of mean ``tau_r`` MOs per state).
    SYNTHETIC flow for correctness research, never a market-regime claim.
    """
    t = _pos_finite(tau_r, "tau_r")
    w = float(omega)
    if not math.isfinite(w) or w < 0.0 or w > 0.5:
        raise ValueError(f"omega must lie in [0, 0.5], got {omega!r}")
    im = _pos_finite(intensity_mult, "intensity_mult")
    _seed_int(seed)
    stay = math.exp(-1.0 / t)
    states = (
        RegimeState("buy_pressure", im, 0.5 + w),
        RegimeState("sell_pressure", im, 0.5 - w),
    )
    return MarkovRegimeFlow(states, (stay, stay), seed=seed)


# ---------------------------------------------------------------------------
# Session runner (run_mm_session-style, extended for the RL agent)
# ---------------------------------------------------------------------------


def run_rl_mm_session(
    *,
    config: ZILobConfig,
    horizon: float,
    agent: C51MarketMaker | None = None,
    policy: QuotePolicy | None = None,
    training: bool = False,
    learn_every: int = 1,
    decision_interval: float = 1.0,
    flow: MarkovRegimeFlow | ScenarioRegimeFlow | AdversarialFlow | None = None,
    inventory_cap: int | None = None,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
) -> dict[str, Any]:
    """Run one market-making session with a C51 agent or a classic QuotePolicy.

    ``run_mm_session``-style accounting (same keys and honesty schema) with
    the RL layer added: action-offset quoting with the paper's safety clamp
    and one-sided inventory gating, smart quoting (orders persist; cancel +
    repost only on a target-level change, preserving FIFO priority), the
    SMDP reward of paper Eq. 16 (``sim_internal_reward_*`` — simulator-
    internal training signal, never a headline metric), n-step transitions
    with event-clock discounts, and the auxiliary belief features (online
    flow filter + quote-exposure imbalance). Exactly one of ``agent`` /
    ``policy`` must be given; ``training=True`` (agent only) enables
    epsilon-greedy exploration, transition storage, and gradient steps every
    ``learn_every`` closed transitions. Evaluation mode acts greedily and
    never mutates agent experience.

    Fail closed: an agent session requires ``inventory_cap`` equal to the
    agent's ``spec.inventory_cap``; classic policies must return on-grid
    prices (the engine's ``price_to_level`` rejects off-grid quotes).

    Returns a SYNTHETIC diagnostic bundle; ``sim_internal_mtm_pnl_*`` is
    simulator-internal mark-to-market accounting, NEVER a headline metric and
    never market evidence. No live-trading claim.
    """
    if (agent is None) == (policy is None):
        raise ValueError("exactly one of agent / policy must be provided")
    if agent is not None and not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker")
    if policy is not None and not callable(policy):
        raise TypeError("policy must be callable")
    if training and agent is None:
        raise ValueError("training=True requires an agent")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    if flow is not None and not isinstance(
        flow, (MarkovRegimeFlow, ScenarioRegimeFlow, AdversarialFlow)
    ):
        raise TypeError(
            "flow must be a MarkovRegimeFlow / ScenarioRegimeFlow / AdversarialFlow or None"
        )
    h = _pos_finite(horizon, "horizon")
    di = _pos_finite(decision_interval, "decision_interval")
    si = _pos_finite(sample_interval, "sample_interval")
    phi = _nonneg_finite(reward_phi, "reward_phi")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    le = _int_at_least(learn_every, 1, "learn_every")
    cap: int | None = None
    if agent is not None:
        if inventory_cap is None:
            raise ValueError("inventory_cap is required in agent mode (hard bound q_max)")
        cap = _int_at_least(inventory_cap, 1, "inventory_cap")
        if cap != agent.spec.inventory_cap:
            raise ValueError(
                f"inventory_cap ({cap}) must equal agent.spec.inventory_cap "
                f"({agent.spec.inventory_cap})"
            )
    elif inventory_cap is not None:
        cap = _int_at_least(inventory_cap, 1, "inventory_cap")

    spec = agent.spec if agent is not None else None
    filt: FlowBiasFilter | None = None
    if agent is not None and spec is not None and spec.aux_enabled:
        filt = FlowBiasFilter(
            tau_r=spec.filter_tau_r,
            a0=spec.filter_a0,
            b0=spec.filter_b0,
            max_run=spec.filter_max_run,
        )

    sim = ZILobSimulator(config, flow=flow)
    inventory = 0
    cash = 0.0
    bid_oid: int | None = None
    ask_oid: int | None = None
    bid_level_live: int | None = None
    ask_level_live: int | None = None
    n_fills = n_fills_bid = n_fills_ask = 0
    n_cancels = n_clips = n_skipped = n_gated = 0
    n_decisions = 1  # the initial quote placement counts as decision 1
    queue_ahead_fills: list[int] = []
    fill_waits: list[float] = []
    inv_path: list[int] = []
    inv_times: list[float] = []
    mtm_path: list[float] = []
    max_abs_inv = 0
    last_mid_ref: list[float | None] = [sim.mid]
    samples: list[Any] = [sim.sample()]
    signs: list[float] = []
    trade_cursor = 0
    interval_fills: list[tuple[float, float]] = []
    events_at_prev = sim.n_events
    prev_state: Array | None = None
    prev_action: int | None = None
    action_hist = [0] * (agent.n_actions if agent is not None else 0)
    rewards: list[float] = []
    rewards_mo: list[int] = []
    losses: list[float] = []
    eps_initial = agent.epsilon if agent is not None else None
    wall = float(cap) * fw if cap is not None else 0.0
    penalty_sum = 0.0

    def _drain_trades() -> None:
        nonlocal trade_cursor, inventory, cash, bid_oid, ask_oid
        nonlocal bid_level_live, ask_level_live
        nonlocal n_fills, n_fills_bid, n_fills_ask, max_abs_inv
        while trade_cursor < len(sim.trades):
            tr = sim.trades[trade_cursor]
            trade_cursor += 1
            signs.append(1.0 if tr.aggressor == "buy" else -1.0)
            if filt is not None:
                filt.observe(1 if tr.aggressor == "buy" else 0)
            if tr.maker_tag != MM_TAG:
                continue
            if tr.maker_side == "buy":
                inventory += tr.qty
                cash -= tr.price * tr.qty
                n_fills_bid += 1
                interval_fills.append((float(tr.qty), tr.price))
            else:
                inventory -= tr.qty
                cash += tr.price * tr.qty
                n_fills_ask += 1
                interval_fills.append((-float(tr.qty), tr.price))
            n_fills += 1
            queue_ahead_fills.append(tr.maker_queue_ahead_at_submit)
            fill_waits.append(tr.t - tr.maker_t_submit)
            if tr.maker_order_id == bid_oid:
                bid_oid = None
                bid_level_live = None
            elif tr.maker_order_id == ask_oid:
                ask_oid = None
                ask_level_live = None
            max_abs_inv = max(max_abs_inv, abs(inventory))
        if isinstance(flow, AdversarialFlow):
            # Feedback channel: the adversary's picker reads the defender's
            # post-fill inventory when the pending boundary resolves.
            flow.note_inventory(float(inventory))

    def _smart_post(side: Side, level: int | None) -> None:
        """Keep the resting order when the target level is unchanged (FIFO
        priority preservation); cancel + repost only on a level change."""
        nonlocal bid_oid, ask_oid, bid_level_live, ask_level_live, n_cancels
        oid, live = (bid_oid, bid_level_live) if side == "buy" else (ask_oid, ask_level_live)
        if level is None:
            if oid is not None and sim.cancel_order(oid):
                n_cancels += 1
            oid, live = None, None
        elif oid is None or live != level:
            if oid is not None and sim.cancel_order(oid):
                n_cancels += 1
            oid = sim.submit_limit_order(side, sim.level_to_price(level), MM_TAG)
            live = level
        if side == "buy":
            bid_oid, bid_level_live = oid, live
        else:
            ask_oid, ask_level_live = oid, live

    def _observe_state() -> Array | None:
        ag, sp = agent, spec
        if ag is None or sp is None:
            return None
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if bb is None or ba is None:
            return None
        qa_b = sim.queue_position(bid_oid) if bid_oid is not None else None
        qa_a = sim.queue_position(ask_oid) if ask_oid is not None else None
        o_b = 1.0 / (1.0 + qa_b) if qa_b is not None else 0.0
        o_a = 1.0 / (1.0 + qa_a) if qa_a is not None else 0.0
        belief = filt.belief() if filt is not None else None
        return build_state_vector(
            sp,
            spread_ticks=ba - bb,
            bid_depth0=sim.depth_at("buy", bb),
            bid_depth1=sim.depth_at("buy", bb - 1),
            ask_depth0=sim.depth_at("sell", ba),
            ask_depth1=sim.depth_at("sell", ba + 1),
            inventory=inventory,
            bid_opportunity=o_b,
            ask_opportunity=o_a,
            belief=belief,
        )

    def _close_transition(next_state: Array, done: bool) -> None:
        """Settle the reward for the previous decision (paper Eq. 16) and, in
        training mode, store the SMDP transition and run gradient steps."""
        nonlocal prev_state, prev_action, events_at_prev, penalty_sum
        if prev_state is None or prev_action is None or agent is None:
            return
        mid_now = sim.mid
        if mid_now is not None:
            last_mid_ref[0] = mid_now
        m = last_mid_ref[0] if last_mid_ref[0] is not None else float("nan")
        q = float(inventory)
        fill_pnl = 0.0
        for dq, px in interval_fills:
            fill_pnl += dq * (m - px)
        wall_term = max(abs(q) - wall, 0.0) ** 2 if cap is not None else 0.0
        r = fill_pnl - phi * q * q - phi * wall_term
        penalty_sum += phi * q * q + phi * wall_term
        n_ev = max(sim.n_events - events_at_prev, 0)
        gamma_eff = float(agent.config.gamma_event) ** n_ev
        rewards.append(r)
        rewards_mo.append(sim.n_mo_arrivals)
        if training:
            agent.store_transition(prev_state, prev_action, r, next_state, gamma_eff, done)
            if done or len(rewards) % le == 0:
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
        interval_fills.clear()
        events_at_prev = sim.n_events
        prev_state = None
        prev_action = None

    def _requote_agent(state_now: Array) -> None:
        nonlocal prev_state, prev_action, n_clips, n_gated
        ag = agent
        if ag is None:  # pragma: no cover - agent-mode only
            return
        action = ag.act(state_now, greedy=not training)
        action_hist[action] += 1
        db, da = ag.action_grid[action]
        bb, ba = sim.best_bid_level, sim.best_ask_level
        if bb is None or ba is None:
            raise RuntimeError("agent requote with a one-sided book (state invariant violated)")
        bid_level = bb - db
        ask_level = ba + da
        if bid_level >= ask_level:
            # Safety layer (paper Sec. 3): never post crossing/marketable
            # quotes; fall back to joining the touch on both sides.
            n_clips += 1
            bid_level, ask_level = bb, ba
        post_bid = True
        post_ask = True
        if cap is not None:
            if inventory >= cap:
                post_bid = False
                n_gated += 1
            if inventory <= -cap:
                post_ask = False
                n_gated += 1
        _smart_post("buy", bid_level if post_bid else None)
        _smart_post("sell", ask_level if post_ask else None)
        prev_state = state_now
        prev_action = action

    def _requote_policy(pol: QuotePolicy) -> None:
        """Classic QuotePolicy path: run_mm_session clip/suppress semantics
        with smart quoting layered on top."""
        nonlocal n_clips, n_skipped
        mid = sim.mid
        if mid is None:
            n_skipped += 1
            return
        state = MMState(
            t=sim.t,
            mid=mid,
            best_bid=sim.best_bid,
            best_ask=sim.best_ask,
            inventory=inventory,
            tau=h - sim.t,
        )
        bid_px, ask_px = pol(state)
        if cap is not None:
            if inventory >= cap:
                bid_px = None
            if inventory <= -cap:
                ask_px = None
        tick = config.tick
        ba_px, bb_px = sim.best_ask, sim.best_bid
        if bid_px is not None and ba_px is not None and bid_px >= ba_px - 1e-12:
            bid_px = ba_px - tick
            n_clips += 1
            if bid_px <= 0.0:
                bid_px = None
        if ask_px is not None and bb_px is not None and ask_px <= bb_px + 1e-12:
            ask_px = bb_px + tick
            n_clips += 1
        if bid_px is not None and ask_px is not None and bid_px >= ask_px:
            n_skipped += 1
            return
        bid_level = sim.price_to_level(bid_px) if bid_px is not None else None
        ask_level = sim.price_to_level(ask_px) if ask_px is not None else None
        _smart_post("buy", bid_level)
        _smart_post("sell", ask_level)

    def _record_path() -> None:
        mid = sim.mid
        if mid is not None:
            last_mid_ref[0] = mid
        inv_path.append(inventory)
        inv_times.append(sim.t)
        if mid is not None:
            mtm_path.append(cash + inventory * mid)
        else:
            mtm_path.append(float("nan"))

    # -- initial decision, then the event loop --------------------------------
    if agent is not None:
        s_init = _observe_state()
        if s_init is None:
            n_skipped += 1
        else:
            _requote_agent(s_init)
    elif policy is not None:
        _requote_policy(policy)
    _record_path()
    next_decision = di
    next_sample = si
    while sim.t < h:
        sim.step()
        _drain_trades()
        if sim.t >= next_decision:
            n_decisions += 1
            if agent is not None:
                s_now = _observe_state()
                if s_now is None:
                    n_skipped += 1
                else:
                    _close_transition(s_now, done=False)
                    _requote_agent(s_now)
            elif policy is not None:
                _requote_policy(policy)
            _record_path()
            while next_decision <= sim.t:
                next_decision += di
        if sim.t >= next_sample:
            samples.append(sim.sample())
            while next_sample <= sim.t:
                next_sample += si

    # Terminal flush: settle the open transition as done (finite-horizon
    # episode), then one extra learn step to drain the pending n-step window.
    if agent is not None and prev_state is not None and prev_action is not None:
        s_end = _observe_state()
        _close_transition(s_end if s_end is not None else prev_state, done=True)
        if training:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
    samples.append(sim.sample())
    _record_path()

    final_mid = sim.mid if sim.mid is not None else last_mid_ref[0]
    final_mtm = cash + inventory * final_mid if final_mid is not None else float("nan")
    flow_diag: dict[str, Any] | None = None
    if len(signs) >= 52:
        flow_diag = regime_flow_diagnostics(signs)
    rw = np.asarray(rewards, dtype=np.float64) if rewards else None
    mean_abs_inv = (
        float(np.mean(np.abs(np.asarray(inv_path, dtype=np.float64)))) if inv_path else float("nan")
    )
    bundle: dict[str, Any] = {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "policy_kind": "c51_rl" if agent is not None else "classic_policy",
        "training": bool(training),
        "seed": config.seed,
        "horizon": h,
        "decision_interval": di,
        "inventory_cap": cap,
        "session_completed": bool(sim.t >= h),
        "n_events": sim.n_events,
        "n_decisions": n_decisions,
        "n_fills": n_fills,
        "n_fills_bid": n_fills_bid,
        "n_fills_ask": n_fills_ask,
        "n_mm_cancels": n_cancels,
        "n_quote_clips": n_clips,
        "n_skipped_decisions": n_skipped,
        "n_inventory_gated": n_gated,
        "inventory_final": inventory,
        "max_abs_inventory": max_abs_inv,
        "mean_abs_inventory": mean_abs_inv,
        "inventory_path": inv_path,
        "inventory_path_times": inv_times,
        # Simulator-internal accounting. Diagnostic only: never a headline
        # metric, never market evidence, no live-trading claim.
        "sim_internal_mtm_pnl_path": mtm_path,
        "sim_internal_mtm_pnl_final": float(final_mtm),
        "sim_internal_penalty_sum": float(penalty_sum),
        "mean_queue_ahead_at_fill": float(np.mean(queue_ahead_fills))
        if queue_ahead_fills
        else float("nan"),
        "mean_fill_wait_seconds": float(np.mean(fill_waits)) if fill_waits else float("nan"),
        "phase_metrics": book_phase_metrics(samples),
        "flow_diagnostics": flow_diag,
        "n_signs": len(signs),
        "event_counts": sim.event_counts(),
        # RL block (None / empty in classic-policy mode)
        "action_grid": [list(g) for g in agent.action_grid] if agent is not None else None,
        "action_histogram": action_hist if agent is not None else None,
        "sim_internal_reward_path": rewards if agent is not None else None,
        "sim_internal_reward_mo_index": rewards_mo if agent is not None else None,
        "sim_internal_reward_mean": float(rw.mean()) if rw is not None else None,
        "sim_internal_reward_min": float(rw.min()) if rw is not None else None,
        "sim_internal_reward_max": float(rw.max()) if rw is not None else None,
        "sim_internal_reward_final": float(rw[-1]) if rw is not None else None,
        "loss_curve": losses,
        "n_learn_steps": len(losses),
        "epsilon_initial": eps_initial,
        "epsilon_final": agent.epsilon if agent is not None else None,
        "n_updates_total": agent.n_updates if agent is not None else None,
        "filter_belief_final": list(filt.belief()) if filt is not None else None,
        "filter_n_mo": filt.n_mo if filt is not None else None,
    }
    return bundle


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_c51_market_maker(
    *,
    agent: C51MarketMaker,
    config: ZILobConfig,
    horizon: float,
    n_episodes: int = 4,
    seed_base: int = 0,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    learn_every: int = 1,
    flow_factory: Callable[[int], MarkovRegimeFlow | ScenarioRegimeFlow | None] | None = None,
) -> dict[str, Any]:
    """Train the agent over ``n_episodes`` simulator sessions.

    Each episode runs :func:`run_rl_mm_session` in training mode on a
    per-episode seeded config (``seed_base + i``) and, optionally, a
    per-episode flow built by ``flow_factory(seed)`` (e.g.
    :func:`paper_regime_flow` for regime-switching training; ``None`` keeps
    the stationary flow). Episodes are sequential; the agent's replay buffer,
    epsilon schedule, and target network persist across them. Rewards are the
    simulator-internal signal of paper Eq. 16 (``sim_internal_*`` keys, never
    headline metrics). Deterministic given the agent and config seeds.
    """
    if not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    n_ep = _int_at_least(n_episodes, 1, "n_episodes")
    _seed_int(seed_base)
    le = _int_at_least(learn_every, 1, "learn_every")
    _pos_finite(horizon, "horizon")
    phi = _nonneg_finite(reward_phi, "reward_phi")
    fw = float(reward_wall_fraction)
    if not math.isfinite(fw) or not 0.0 < fw < 1.0:
        raise ValueError(f"reward_wall_fraction must lie in (0, 1), got {reward_wall_fraction!r}")
    if flow_factory is not None and not callable(flow_factory):
        raise TypeError("flow_factory must be callable or None")
    episodes: list[dict[str, Any]] = []
    all_losses: list[float] = []
    for i in range(n_ep):
        ep_seed = seed_base + i
        cfg_i = replace(config, seed=ep_seed)
        flow_i = flow_factory(ep_seed) if flow_factory is not None else None
        if flow_i is not None and not isinstance(flow_i, (MarkovRegimeFlow, ScenarioRegimeFlow)):
            raise TypeError(
                "flow_factory must return a MarkovRegimeFlow / ScenarioRegimeFlow or None"
            )
        agent.begin_episode()
        bundle = run_rl_mm_session(
            config=cfg_i,
            horizon=horizon,
            agent=agent,
            training=True,
            learn_every=le,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            inventory_cap=agent.spec.inventory_cap,
            reward_phi=reward_phi,
            reward_wall_fraction=reward_wall_fraction,
            flow=flow_i,
        )
        all_losses.extend(bundle["loss_curve"])
        episodes.append(
            {
                "seed": bundle["seed"],
                "session_completed": bundle["session_completed"],
                "n_decisions": bundle["n_decisions"],
                "n_fills": bundle["n_fills"],
                "max_abs_inventory": bundle["max_abs_inventory"],
                "inventory_final": bundle["inventory_final"],
                "sim_internal_mtm_pnl_final": bundle["sim_internal_mtm_pnl_final"],
                "sim_internal_reward_mean": bundle["sim_internal_reward_mean"],
                "epsilon_final": bundle["epsilon_final"],
                "n_inventory_gated": bundle["n_inventory_gated"],
            }
        )
    loss_arr = np.asarray(all_losses, dtype=np.float64) if all_losses else None
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "c51_training_run",
        "flow": "regime_factory" if flow_factory is not None else "stationary",
        "n_episodes": n_ep,
        "episodes": episodes,
        "loss_curve": all_losses,
        "loss_initial_mean": float(loss_arr[:10].mean()) if loss_arr is not None else None,
        "loss_final_mean": float(loss_arr[-10:].mean()) if loss_arr is not None else None,
        "n_updates": agent.n_updates,
        "n_decisions_total": agent.n_decisions,
        "n_transitions_stored": agent.total_stored,
        "epsilon_final": agent.epsilon,
        "training_budget": {
            "horizon": float(horizon),
            "n_episodes": n_ep,
            "decision_interval": float(decision_interval),
            "learn_every": le,
            "reward_phi": phi,
            "reward_wall_fraction": fw,
            "seed_base": int(seed_base),
        },
    }


# ---------------------------------------------------------------------------
# Evaluation harness: trained C51 vs AS vs GLFT, stationary vs regime flow
# ---------------------------------------------------------------------------

_SESSION_ROW_KEYS = (
    "seed",
    "session_completed",
    "max_abs_inventory",
    "mean_abs_inventory",
    "inventory_final",
    "sim_internal_mtm_pnl_final",
    "sim_internal_mtm_pnl_path_mean",
    "sim_internal_mtm_pnl_path_std",
    "sim_internal_mtm_pnl_path_min",
    "sim_internal_mtm_pnl_path_max",
    "n_fills",
    "n_decisions",
    "n_inventory_gated",
    "mean_spread_ticks",
    "phase",
    "regime_detected",
)


def _session_row(bundle: dict[str, Any]) -> dict[str, Any]:
    path = np.asarray(bundle["sim_internal_mtm_pnl_path"], dtype=np.float64)
    finite = path[np.isfinite(path)]
    fd = bundle["flow_diagnostics"]
    return {
        "seed": bundle["seed"],
        "session_completed": bool(bundle["session_completed"]),
        "max_abs_inventory": int(bundle["max_abs_inventory"]),
        "mean_abs_inventory": float(bundle["mean_abs_inventory"]),
        "inventory_final": int(bundle["inventory_final"]),
        "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
        "sim_internal_mtm_pnl_path_mean": float(finite.mean()) if finite.size else float("nan"),
        "sim_internal_mtm_pnl_path_std": float(finite.std()) if finite.size else float("nan"),
        "sim_internal_mtm_pnl_path_min": float(finite.min()) if finite.size else float("nan"),
        "sim_internal_mtm_pnl_path_max": float(finite.max()) if finite.size else float("nan"),
        "n_fills": int(bundle["n_fills"]),
        "n_decisions": int(bundle["n_decisions"]),
        "n_inventory_gated": int(bundle["n_inventory_gated"]),
        "mean_spread_ticks": float(bundle["phase_metrics"]["mean_spread_ticks"]),
        "phase": str(bundle["phase_metrics"]["phase"]),
        "regime_detected": None if fd is None else bool(fd["regime_detected"]),
    }


def evaluate_rl_market_makers(
    *,
    agent: C51MarketMaker,
    config: ZILobConfig,
    horizon: float,
    n_seeds: int = 2,
    seed_base: int = 101,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    as_gamma: float = 0.002,
    as_sigma: float = 0.02,
    as_kappa: float = 1000.0,
    glft_gamma: float = 1.0,
    glft_sigma: float = 0.02,
    glft_kappa: float = 1000.0,
    glft_a_fill: float = 1.0,
    regime_tau_r: float = 60.0,
    regime_omega: float = 0.30,
) -> dict[str, Any]:
    """Paired-seed evaluation: trained C51 vs Avellaneda-Stoikov vs GLFT.

    Runs every policy through :func:`run_rl_mm_session` (identical accounting,
    smart quoting, and gating mechanics for all three; the closed forms are
    the ``zi_lob_simulator`` adapters — single source of truth) on matched
    seeds under (a) stationary flow and (b) regime-switching flow
    (:func:`paper_regime_flow`). The trained agent acts greedily; no agent
    state is mutated. Reports inventory saturation, session completion,
    simulator-internal MTM path stats, fill counts, and book phase metrics.

    SYNTHETIC policy comparison on a synthetic engine: ``sim_internal_*``
    keys are diagnostics, never headline metrics, never market evidence. A
    trained policy beating GLFT at paper budgets is the paper's claim, not
    something tiny-budget callers should infer from this bundle.
    """
    if not isinstance(agent, C51MarketMaker):
        raise TypeError("agent must be a C51MarketMaker")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    ns = _int_at_least(n_seeds, 1, "n_seeds")
    _seed_int(seed_base)
    h = _pos_finite(horizon, "horizon")
    tau_r = _pos_finite(regime_tau_r, "regime_tau_r")
    omega = float(regime_omega)
    if not math.isfinite(omega) or omega < 0.0 or omega > 0.5:
        raise ValueError(f"regime_omega must lie in [0, 0.5], got {regime_omega!r}")
    cap = agent.spec.inventory_cap
    tick = config.tick
    policies: dict[str, QuotePolicy | None] = {
        "c51": None,
        "as": as_policy(gamma=as_gamma, sigma=as_sigma, kappa=as_kappa, tick=tick),
        "glft": glft_policy(
            gamma=glft_gamma, sigma=glft_sigma, kappa=glft_kappa, a_fill=glft_a_fill, tick=tick
        ),
    }
    sessions: dict[str, list[dict[str, Any]]] = {}
    for flow_name in ("stationary", "regime"):
        for i in range(ns):
            ep_seed = seed_base + i
            cfg_i = replace(config, seed=ep_seed)
            flow_i = (
                None
                if flow_name == "stationary"
                else paper_regime_flow(
                    seed=seed_base + 100_003 + i, tau_r=regime_tau_r, omega=regime_omega
                )
            )
            for pol_name in ("c51", "as", "glft"):
                pol = policies[pol_name]
                bundle = run_rl_mm_session(
                    config=cfg_i,
                    horizon=h,
                    agent=agent if pol is None else None,
                    policy=pol,
                    training=False,
                    decision_interval=decision_interval,
                    sample_interval=sample_interval,
                    inventory_cap=cap,
                    reward_phi=reward_phi,
                    reward_wall_fraction=reward_wall_fraction,
                    flow=flow_i,
                )
                sessions.setdefault(f"{pol_name}_{flow_name}", []).append(_session_row(bundle))

    metrics: dict[str, float] = {}
    for combo, rows in sessions.items():
        n = float(len(rows))
        completed = sum(1.0 for r in rows if r["session_completed"])
        saturated = sum(1.0 for r in rows if r["max_abs_inventory"] >= cap)
        detected = [r["regime_detected"] for r in rows if r["regime_detected"] is not None]
        finals = np.asarray([r["sim_internal_mtm_pnl_final"] for r in rows], dtype=np.float64)
        metrics[f"{combo}_session_completion_rate"] = completed / n
        metrics[f"{combo}_inventory_saturation_rate"] = saturated / n
        metrics[f"{combo}_max_abs_inventory_mean"] = float(
            np.mean([r["max_abs_inventory"] for r in rows])
        )
        metrics[f"{combo}_mean_abs_inventory_mean"] = float(
            np.mean([r["mean_abs_inventory"] for r in rows])
        )
        metrics[f"{combo}_inventory_final_abs_mean"] = float(
            np.mean([abs(r["inventory_final"]) for r in rows])
        )
        # P&L-like aggregates lead with the simulator-internal namespace so the
        # honesty scan (pnl keys must start with sim_internal_) holds verbatim.
        metrics[f"sim_internal_mtm_pnl_final_mean_{combo}"] = float(finals.mean())
        metrics[f"sim_internal_mtm_pnl_final_std_{combo}"] = float(finals.std())
        metrics[f"sim_internal_mtm_pnl_path_mean_{combo}"] = float(
            np.mean([r["sim_internal_mtm_pnl_path_mean"] for r in rows])
        )
        metrics[f"sim_internal_mtm_pnl_path_std_{combo}"] = float(
            np.mean([r["sim_internal_mtm_pnl_path_std"] for r in rows])
        )
        metrics[f"sim_internal_mtm_pnl_path_min_{combo}"] = float(
            np.mean([r["sim_internal_mtm_pnl_path_min"] for r in rows])
        )
        metrics[f"{combo}_n_fills_mean"] = float(np.mean([r["n_fills"] for r in rows]))
        metrics[f"{combo}_n_decisions_mean"] = float(np.mean([r["n_decisions"] for r in rows]))
        metrics[f"{combo}_n_inventory_gated_mean"] = float(
            np.mean([r["n_inventory_gated"] for r in rows])
        )
        metrics[f"{combo}_mean_spread_ticks_mean"] = float(
            np.mean([r["mean_spread_ticks"] for r in rows])
        )
        metrics[f"{combo}_regime_detected_rate"] = (
            float(np.mean([1.0 if d else 0.0 for d in detected])) if detected else float("nan")
        )
    for flow_name in ("stationary", "regime"):
        for other in ("glft", "as"):
            gap = (
                metrics[f"sim_internal_mtm_pnl_final_mean_c51_{flow_name}"]
                - metrics[f"sim_internal_mtm_pnl_final_mean_{other}_{flow_name}"]
            )
            metrics[f"sim_internal_mtm_pnl_gap_c51_minus_{other}_{flow_name}_mean"] = gap
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "c51_vs_classic_evaluation",
        "note": (
            "simulator-internal synthetic policy comparison; never headline "
            "metrics, never market evidence, no live-trading claim"
        ),
        "n_seeds": ns,
        "seed_base": int(seed_base),
        "horizon": h,
        "inventory_cap": cap,
        "decision_interval": float(decision_interval),
        "regime_tau_r": tau_r,
        "regime_omega": omega,
        "policies": ("c51", "as", "glft"),
        "flows": ("stationary", "regime"),
        "sessions": sessions,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Documented-optional full-budget benchmark (bench battery entry point)
# ---------------------------------------------------------------------------


def rl_mm_benchmark(
    *,
    config: ZILobConfig | None = None,
    horizon: float = 3000.0,
    train_episodes: int = 40,
    eval_seeds: int = 8,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    reward_phi: float = 1e-3,
    reward_wall_fraction: float = 0.5,
    learn_every: int = 1,
    regime_tau_r: float = 60.0,
    regime_omega: float = 0.30,
    state_spec: RLStateSpec | None = None,
    agent_config: C51Config | None = None,
    train_under_regime: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Full-comparison benchmark: train C51, then evaluate vs AS and GLFT.

    Documented-optional and long-running by design — NOT part of the PR-gate
    test suite (the ``slow``-marked test exercises a scaled-down version of
    the same code path). Intended as the bench-battery entry point for lane
    B4-ii; at paper-scale budgets (101 atoms, hidden 256, buffer 1e5,
    thousands of episodes, horizon 3000 s) this runs for hours. All outputs
    are SYNTHETIC simulator-internal diagnostics, never headline metrics,
    never market evidence, no live-trading claim.

    Trains under regime-switching flow by default (auxiliary belief features
    on — the single-stage deviation documented in the module docstring), then
    evaluates the greedy policy against AS/GLFT on stationary and
    regime-switching flows with paired seeds.
    """
    _seed_int(seed)
    cfg = config if config is not None else santa_fe_config(seed=seed)
    if not isinstance(cfg, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    spec = (
        state_spec
        if state_spec is not None
        else RLStateSpec(aux_enabled=True, filter_tau_r=regime_tau_r)
    )
    c51_cfg = (
        agent_config
        if agent_config is not None
        else C51Config(
            n_atoms=101,
            hidden=(256,),
            gamma_event=1.0 - 1e-3,
            n_step=3,
            lr=3e-4,
            batch_size=64,
            buffer_capacity=100_000,
            target_update_every=2_000,
            eps_decay_steps=max(1, int(train_episodes * horizon / decision_interval // 2)),
            seed=seed,
        )
    )
    if not isinstance(c51_cfg, C51Config):
        raise TypeError("agent_config must be a C51Config")
    agent = C51MarketMaker(spec, c51_cfg)
    factory: Callable[[int], MarkovRegimeFlow | None] | None = None
    if train_under_regime:

        def _regime_factory(s: int) -> MarkovRegimeFlow:
            return paper_regime_flow(seed=s, tau_r=regime_tau_r, omega=regime_omega)

        factory = _regime_factory

    training = train_c51_market_maker(
        agent=agent,
        config=cfg,
        horizon=horizon,
        n_episodes=train_episodes,
        seed_base=seed,
        decision_interval=decision_interval,
        sample_interval=sample_interval,
        reward_phi=reward_phi,
        reward_wall_fraction=reward_wall_fraction,
        learn_every=learn_every,
        flow_factory=factory,
    )
    evaluation = evaluate_rl_market_makers(
        agent=agent,
        config=cfg,
        horizon=horizon,
        n_seeds=eval_seeds,
        seed_base=seed + 7_777_777,
        decision_interval=decision_interval,
        sample_interval=sample_interval,
        reward_phi=reward_phi,
        reward_wall_fraction=reward_wall_fraction,
        regime_tau_r=regime_tau_r,
        regime_omega=regime_omega,
    )
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": RL_MM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "c51_benchmark",
        "train_under_regime": bool(train_under_regime),
        "training": training,
        "evaluation": evaluation,
        "metrics": evaluation["metrics"],
        "bench_keys": sorted(evaluation["metrics"]),
    }
