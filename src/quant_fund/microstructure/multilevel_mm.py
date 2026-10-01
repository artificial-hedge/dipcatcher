"""Multi-level market making: deep-set encoder + logistic-normal allocations.

**Labeled SYNTHETIC** research infrastructure (wave-17 lane): an actor-critic
market maker that posts *distributions* of unit limit orders across multiple
price levels, following

- Cheridito, Weiss (2026). Multi-Level Market Making with Reinforcement
  Learning. arXiv:2608.18195 (verified 2026-09-30 against the arXiv abstract +
  PDF full text). The agent allocates up to ``M`` lots per decision over the
  simplex ``S^{2(K+1)}``: hold fraction ``a_0``, market buy ``a_1``, limit buys
  ``a_{2..K+1}`` at ``0..K-1`` ticks below the best bid, market sell
  ``a_{K+2}``, and limit sells ``a_{K+3..2(K+1)}`` at ``0..K-1`` ticks above
  the best ask (their Sec. 3.2). Allocations are sampled from a multivariate
  **logistic-normal** policy ``h(X)``, ``X ~ N(mu_theta(s), Diag(exp(theta_v)))``
  with the additive log-ratio (ALR) density of Aitchison & Shen 1980 (their
  Eqs. 8-9); a **deep-set** encoder maps the agent's variable-length resting
  order set to fixed-dimensional per-level embeddings by per-order embedding
  ``f^o_phi`` + mean pooling inside each price level (their Sec. 4.1, Zaheer
  et al. 2017); the critic estimates ``V`` and the advantage is the
  returns-to-go minus the value estimate (their Eqs. 10-11); training uses the
  combined actor+critic loss (their Eq. 14). **Potential-based reward
  shaping**: the per-decision reward carries the telescoping term
  ``Q_{n+1} p_{n+1} - Q_n p_n`` (their Eq. 5) — a Ng-Harada-Russell (1999)
  potential ``Phi(s) = Q p`` at discount 1 — so it cancels over the episode
  (their Eq. 7) while densifying the learning signal and preserving the
  optimal policy class.

Supporting citations:

- Aitchison, Shen (1980). Logistic-normal distributions: some properties and
  uses. *Biometrika* 67(2):261-272 — ALR density + Jacobian used in
  :func:`logistic_normal_logpdf`.
- Zaheer et al. (2017). Deep sets. *NeurIPS 30*, arXiv:1703.06114 —
  permutation-invariant pooling (``deep_set_pool``).
- Ng, Harada, Russell (1999). Policy invariance under reward transformations.
  *ICML*, citeseerx 10.1.1.48.694 — potential-based shaping
  ``F = gamma*Phi(s') - Phi(s)``; ``shaped_optimal_q`` implements the exact
  invariance ``Q'*(s,a) = Q*(s,a) - Phi(s)`` (argmax preserving).
- Balinski, Young (2010). *Fair Representation*, 2nd ed. — Hamilton (largest
  remainder) apportionment ``hamilton_apportionment`` (paper Sec. 3.2).
- Sutton et al. (1999). Policy gradient methods. *NeurIPS* — Eq. 12.
- Kingma, Ba (2014). Adam. arXiv:1412.6980; Saxe et al. (2013) orthogonal
  init, arXiv:1312.6120 — the paper's optimizer/initialization (App. A.1).

Composition (nothing here modifies the composed modules):

- ``microstructure.zi_lob_simulator`` is the venue: ``ZILobSimulator``'s
  price-time FIFO engine hosts the agent's unit-lot orders, ``MM_TAG`` marks
  maker fills, and ``glft_policy``/``as_policy`` are the classic contrast arms
  in :func:`evaluate_multilevel_mm`. The simulator is **unit-size** only, so a
  paper order of ``w`` lots becomes ``w`` resting unit orders; queue position
  ``q`` is the engine's FIFO index. The paper's three trader populations
  (noise / tactical / strategic, its Sec. 5) are represented here by the
  composed engine's ZI flow plus optional :class:`MarkovRegimeFlow`
  directional modulation — the ZI engine does not expose per-side LO
  placement/cancellation streams, so the interval flow features use the
  observable aggregate counters (documented deviation, see
  ``build_base_features``).
- ``microstructure.rl_market_maker`` (lane B4-ii) is the *single-quote* C51
  RL market maker on the same engine. This module does NOT reuse its discrete
  action grid / replay buffer / C51 head: the paper's policy is a continuous
  simplex distribution (logistic-normal) trained by vanilla actor-critic, an
  incompatible architecture. The session bookkeeping pattern (smart quoting,
  ``sim_internal_*`` namespacing, paired-seed evaluation) is mirrored for
  consistency.

Honesty: every output is a SYNTHETIC correctness diagnostic on a synthetic
engine, never market evidence. All P&L-like keys are namespaced
``sim_internal_*``, must never be headlined, and there is no broker
connectivity or live-trading claim anywhere in this module. Rewards are the
simulator-internal training signal of paper Eq. 5 (a *shaped* objective), not
a performance metric. Tiny-budget tests assert plumbing, determinism, and
learning mechanics, NOT paper-scale results; ``multilevel_mm_benchmark`` is
the documented-optional full-budget entry point.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch` (mirrors ``models.deep_hedging``), so this module imports
cleanly without torch and every torch entry point raises ``ImportError`` with
install guidance. The numpy core (apportionment, logistic-normal density,
shaping helpers, feature encoding, the session runner, classic and random
policies) is torch-free. Given ``seed``, torch and numpy randomness are fully
seeded and CPU training is single-threaded: identical call sequences give
bit-identical results (GPU determinism is not claimed).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from quant_fund.microstructure.zi_lob_simulator import (
    MM_TAG,
    ZI_LOB_REVISION,
    MarkovRegimeFlow,
    MMState,
    QuotePolicy,
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
IntArray = NDArray[np.int64]

__all__ = [
    "MLMM_REVISION",
    "MMObs",
    "MultiLevelMMConfig",
    "MultiLevelMMAgent",
    "MultiLevelSpec",
    "RandomAllocationPolicy",
    "TrajectoryStep",
    "action_component_map",
    "alr_transform",
    "build_base_features",
    "collect_order_elements",
    "deep_set_pool",
    "evaluate_multilevel_mm",
    "hamilton_apportionment",
    "logistic_normal_logpdf",
    "logistic_normal_sample",
    "mm_reward",
    "multilevel_mm_benchmark",
    "potential_shaping",
    "run_multilevel_mm_session",
    "shape_episode_rewards",
    "shaped_optimal_q",
    "simplex_transform",
    "terminal_position_limit",
    "terminal_reward",
    "train_multilevel_mm",
]

MLMM_REVISION = "SYNTHETIC_MLMM_v1"


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


def _finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {x!r}")
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
            "the multi-level market maker needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Problem specification and the action simplex
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiLevelSpec:
    """Problem dimensions for the multi-level market maker.

    ``n_levels`` is the paper's ``K``: how many ticks below the best bid /
    above the best ask the agent's limit orders may sit (action components
    ``a_{2..K+1}`` / ``a_{K+3..2(K+1)}``). ``lots`` is ``M``: the maximum
    integer lot budget allocated each decision. ``inventory_cap`` is the hard
    position bound used for post-hoc gating and feature normalization; the
    paper's terminal limit is ``ceil(terminal_nu * lots)`` handled by the
    runner. ``emb_dim`` is the deep-set order embedding width (paper: 2).
    ``depth_scale``/``queue_scale`` are the paper's volume / queue-position
    normalization constants (paper App. A.2 uses 100 for both).
    """

    n_levels: int = 3
    lots: int = 4
    inventory_cap: int = 8
    emb_dim: int = 2
    depth_scale: float = 100.0
    queue_scale: float = 100.0

    def __post_init__(self) -> None:
        _int_at_least(self.n_levels, 1, "n_levels")
        _int_at_least(self.lots, 1, "lots")
        _int_at_least(self.inventory_cap, 1, "inventory_cap")
        _int_at_least(self.emb_dim, 1, "emb_dim")
        _pos_finite(self.depth_scale, "depth_scale")
        _pos_finite(self.queue_scale, "queue_scale")

    @property
    def n_components(self) -> int:
        """Simplex dimension ``2(K+1) + 1``: hold + market + K limits per side."""
        return 2 * self.n_levels + 3

    @property
    def n_logits(self) -> int:
        """Gaussian logit dimension ``2(K+1)`` (reference component excluded)."""
        return 2 * self.n_levels + 2

    @property
    def n_order_slots(self) -> int:
        """Deep-set level slots: ``K`` bid-side + ``K`` ask-side."""
        return 2 * self.n_levels

    @property
    def n_base_features(self) -> int:
        """Scalar + kappa feature count: ``(2K + 8) + 2(K+1) = 4K + 10``."""
        return 4 * self.n_levels + 10

    @property
    def state_dim(self) -> int:
        """Full encoded state: ``2K*emb_dim + 4K + 10``."""
        return 2 * self.n_levels * self.emb_dim + self.n_base_features


@dataclass(frozen=True)
class ActionComponent:
    """One component of the action simplex (paper Sec. 3.2)."""

    kind: Literal["hold", "market", "limit"]
    side: Side | None
    level_offset: int | None  # ticks below best bid / above best ask (limit legs)


def action_component_map(n_levels: int) -> tuple[ActionComponent, ...]:
    """Map each simplex component to its order-flow meaning (paper Sec. 3.2).

    Layout: ``a_0`` hold, ``a_1`` market buy, ``a_{2+k}`` limit buy ``k`` ticks
    below the best bid for ``k = 0..K-1``, ``a_{K+2}`` market sell, and
    ``a_{K+3+k}`` limit sell ``k`` ticks above the best ask. With ``K = 1``
    this collapses to single-quote placement: one limit-bid component (the
    best bid) and one limit-ask component (the best ask).
    """
    k = _int_at_least(n_levels, 1, "n_levels")
    out: list[ActionComponent] = [
        ActionComponent("hold", None, None),
        ActionComponent("market", "buy", None),
    ]
    for off in range(k):
        out.append(ActionComponent("limit", "buy", off))
    out.append(ActionComponent("market", "sell", None))
    for off in range(k):
        out.append(ActionComponent("limit", "sell", off))
    return tuple(out)


def hamilton_apportionment(weights: Sequence[float] | Array, lots: int) -> IntArray:
    """Round simplex weights to an integer lot split (paper Sec. 3.2).

    Hamilton / largest-remainder method (Balinski & Young 2010): quotas
    ``w_k * M`` are floored and the remaining lots are dealt one each to the
    largest fractional remainders, ties broken by the lowest component index
    (deterministic). Weights must be non-negative and finite with positive
    sum; they are renormalized so the output always sums to ``lots``.
    """
    m = _int_at_least(lots, 1, "lots")
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be non-negative, got {weights!r}")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    w = w / total
    quotas = w * m
    base = np.floor(quotas).astype(np.int64)
    remainder = m - int(base.sum())
    order = np.argsort(-(quotas - base), kind="stable")
    for i in range(remainder):
        base[int(order[i])] += 1
    return base


# ---------------------------------------------------------------------------
# Logistic-normal allocations (paper Sec. 4.2, Aitchison & Shen 1980)
# ---------------------------------------------------------------------------


def simplex_transform(x: Sequence[float] | Array) -> Array:
    """Logistic map ``h: R^d -> S^{d+1}`` (paper Eq. before 8).

    ``a_k = exp(x_k) / (1 + sum_l exp(x_l))`` for ``k = 1..d`` and the
    reference component ``a_0 = 1 / (1 + sum_l exp(x_l))``. The output is a
    strictly-positive point of the open simplex (all components in (0, 1),
    summing to 1). Computed in a max-shifted form for numerical stability.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("x must be non-empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("x must be finite")
    z = np.concatenate(([0.0], v))
    z = z - float(z.max())
    e = np.exp(z)
    return e / float(e.sum())


def alr_transform(a: Sequence[float] | Array) -> Array:
    """Additive log-ratio inverse of :func:`simplex_transform`.

    ``y_k = log(a_k / a_0)`` for ``k = 1..d`` — the exact inverse of ``h`` on
    the open simplex, so ``alr_transform(simplex_transform(x)) == x``
    componentwise. Fail closed: entries must be strictly positive (the open
    simplex) and sum to 1.
    """
    v = np.asarray(a, dtype=np.float64).ravel()
    if v.size < 2:
        raise ValueError(f"a must have >= 2 components, got {v.size}")
    if not np.all(np.isfinite(v)):
        raise ValueError("a must be finite")
    if np.any(v <= 0.0):
        raise ValueError("a must be strictly positive (open simplex)")
    s = float(v.sum())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"a must sum to 1, got sum={s!r}")
    return np.asarray(np.log(v[1:] / v[0]), dtype=np.float64)


def logistic_normal_logpdf(a: Sequence[float] | Array, mean: Array, logvar: Array) -> float:
    """Closed-form logistic-normal log density (Aitchison & Shen 1980).

    If ``X ~ N(mu, Diag(exp(logvar)))`` then ``a = h(X)`` has density

        p(a) = N(alr(a); mu, Sigma) / prod_{k=0}^{d} a_k ,

    the Jacobian of the ALR map being ``prod_k a_k`` over all ``d + 1``
    components including the reference ``a_0``. This is the density behind the
    policy-gradient term ``log pi_theta(a|s)`` of the paper's Eq. 12/14.
    """
    y = alr_transform(a)
    mu = np.asarray(mean, dtype=np.float64).ravel()
    lv = np.asarray(logvar, dtype=np.float64).ravel()
    d = y.size
    if mu.shape != (d,) or lv.shape != (d,):
        raise ValueError(f"mean/logvar must have shape ({d},), got {mu.shape} and {lv.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(lv)):
        raise ValueError("mean and logvar must be finite")
    av = np.asarray(a, dtype=np.float64).ravel()
    var = np.exp(lv)
    resid = (y - mu) ** 2 / var
    log_n = -0.5 * (d * math.log(2.0 * math.pi) + float(lv.sum()) + float(resid.sum()))
    return float(log_n - float(np.log(av).sum()))


def logistic_normal_sample(mean: Array, logvar: Array, rng: np.random.Generator) -> Array:
    """Draw ``a = h(X)`` with ``X ~ N(mu, Diag(exp(logvar)))``; seeded RNG."""
    mu = np.asarray(mean, dtype=np.float64).ravel()
    lv = np.asarray(logvar, dtype=np.float64).ravel()
    if mu.size == 0:
        raise ValueError("mean must be non-empty")
    if mu.shape != lv.shape:
        raise ValueError(f"mean and logvar shapes must agree, got {mu.shape} / {lv.shape}")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(lv)):
        raise ValueError("mean and logvar must be finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    z = rng.standard_normal(mu.size)
    return simplex_transform(mu + np.exp(0.5 * lv) * z)


# ---------------------------------------------------------------------------
# Potential-based reward shaping (Ng, Harada, Russell 1999; paper Sec. 3.3)
# ---------------------------------------------------------------------------


def potential_shaping(phi_now: float, phi_next: float, gamma: float = 1.0) -> float:
    """Shaping bonus ``F = gamma * Phi(s') - Phi(s)``.

    The paper's per-decision reward carries ``Phi(s) = Q p`` (inventory valued
    at mid) at ``gamma = 1`` — the ``Q_{n+1} p_{n+1} - Q_n p_n`` term of Eq. 5.
    Fail closed on non-finite potentials or a discount outside ``(0, 1]``.
    """
    p0 = _finite(phi_now, "phi_now")
    p1 = _finite(phi_next, "phi_next")
    g = float(gamma)
    if not math.isfinite(g) or not 0.0 < g <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma!r}")
    return g * p1 - p0


def shape_episode_rewards(
    rewards: Sequence[float] | Array,
    potentials: Sequence[float] | Array,
    gamma: float = 1.0,
) -> Array:
    """Apply potential shaping to a length-``N`` reward sequence.

    ``potentials`` has length ``N + 1`` (``Phi(s_0) .. Phi(s_N)``); returns
    ``r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t)``. Invariance (documented,
    tested): the discounted shaped return is ``R' = R + gamma^N Phi(s_N) -
    Phi(s_0)`` — an offset independent of the action sequence, so the optimal
    policy class is preserved.
    """
    r = np.asarray(rewards, dtype=np.float64).ravel()
    phi = np.asarray(potentials, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("rewards must be non-empty")
    if phi.size != r.size + 1:
        raise ValueError(
            f"potentials must have length n_rewards + 1 ({r.size + 1}), got {phi.size}"
        )
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(phi)):
        raise ValueError("rewards and potentials must be finite")
    g = float(gamma)
    if not math.isfinite(g) or not 0.0 < g <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma!r}")
    return r + g * phi[1:] - phi[:-1]


def shaped_optimal_q(q_values: Sequence[float] | Array, phi_state: float) -> Array:
    """Optimal Q under the shaped reward: ``Q'*(s, a) = Q*(s, a) - Phi(s)``.

    Ng-Harada-Russell (1999) invariance: adding ``gamma*Phi(s') - Phi(s)`` to
    every reward shifts the optimal action-value at ``s`` by the constant
    ``-Phi(s)``, so ``argmax_a`` is exactly preserved. This helper implements
    the shift; tests assert the argmax invariance.
    """
    q = np.asarray(q_values, dtype=np.float64).ravel()
    if q.size == 0:
        raise ValueError("q_values must be non-empty")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_values must be finite")
    p = _finite(phi_state, "phi_state")
    return q - p


# ---------------------------------------------------------------------------
# Rewards (paper Eqs. 5-7)
# ---------------------------------------------------------------------------


def mm_reward(
    *,
    cash_flow: float,
    q_prev: int,
    p_prev: float,
    q_next: int,
    p_next: float,
    inv_gamma: float,
    lots: int,
) -> float:
    """Per-decision reward, paper Eq. 5.

    ``r = (cash_flow + (Q_{n+1} p_{n+1} - Q_n p_n) - gamma |Q_{n+1}|) / M``
    where ``cash_flow`` is signed fill cash over ``(t_n, t_{n+1}]``, ``Q`` the
    lot inventory, ``p`` the mid. The ``Q'p' - Qp`` term is the potential
    shaping ``Phi = Q p`` (telescopes over the episode, Eq. 7); the
    ``-gamma |Q'|`` term is the inventory penalty. Simulator-internal
    (``sim_internal_*``) training signal, never a headline metric.
    """
    cf = _finite(cash_flow, "cash_flow")
    p0 = _pos_finite(p_prev, "p_prev")
    p1 = _pos_finite(p_next, "p_next")
    g = _nonneg_finite(inv_gamma, "inv_gamma")
    m = _int_at_least(lots, 1, "lots")
    if isinstance(q_prev, bool) or isinstance(q_next, bool):
        raise ValueError("inventories must be ints")
    return float((cf + (q_next * p1 - q_prev * p0) - g * abs(q_next)) / m)


def terminal_position_limit(lots: int, nu: float) -> int:
    """Paper Sec. 3.3 terminal bound ``ceil(nu * M)``, ``nu in [0, 1]``."""
    m = _int_at_least(lots, 1, "lots")
    v = _prob(nu, "nu")
    return int(math.ceil(v * m))


def terminal_reward(
    *,
    q_n: int,
    p_n: float,
    q_plus: int,
    mo_cash: float,
    lots: int,
) -> float:
    """Terminal reward ``g(s_N)``, paper Eq. 6.

    Inventory beyond ``ceil(nu M)`` is liquidated by a market order; ``mo_cash``
    is the signed cash actually received (it walks the book), and the leftover
    ``Q_{N+}`` is valued at the terminal mid: ``g = (p_N (Q_{N+} - Q_N) +
    MO_nu) / M``. Fail closed on inconsistent liquidation accounting
    (``|q_plus|`` must not exceed ``|q_n|`` with matching sign).
    """
    p = _pos_finite(p_n, "p_n")
    c = _finite(mo_cash, "mo_cash")
    m = _int_at_least(lots, 1, "lots")
    if isinstance(q_n, bool) or isinstance(q_plus, bool):
        raise ValueError("inventories must be ints")
    if abs(q_plus) > abs(q_n) or (q_plus != 0 and (q_plus > 0) != (q_n > 0)):
        raise ValueError(
            f"q_plus must be a partial liquidation of q_n, got q_n={q_n}, q_plus={q_plus}"
        )
    return float((p * (q_plus - q_n) + c) / m)


# ---------------------------------------------------------------------------
# State features: deep-set order encoding + normalized scalars
# ---------------------------------------------------------------------------


def deep_set_pool(features: Sequence[Sequence[float]] | Array, slots: Array, n_slots: int) -> Array:
    """Mean-pool element embeddings into ``n_slots`` level buckets (flattened).

    Per Zaheer et al. (2017) and the paper's Sec. 4.1: each element is embedded
    by a shared ``f^o_phi`` and the embeddings inside each price level are
    averaged; empty levels map to zeros. Permutation-invariant by construction:
    rows inside each slot are canonically sorted before averaging, so
    shuffling the rows of ``(features, slots)`` leaves the output bitwise
    identical — this is the property the tests assert.
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim == 1:
        f = f.reshape(-1, 1) if f.size else np.zeros((0, 1))
    if f.ndim != 2:
        raise ValueError(f"features must be 2-D (n_elems, d), got shape {f.shape}")
    s = np.asarray(slots, dtype=np.int64).ravel()
    if s.shape[0] != f.shape[0]:
        raise ValueError(f"slots length {s.shape[0]} must match n_elems {f.shape[0]}")
    k = _int_at_least(n_slots, 1, "n_slots")
    if s.size and (int(s.min()) < 0 or int(s.max()) >= k):
        raise ValueError(f"slots must lie in [0, {k}), got range {s.min()}..{s.max()}")
    if not np.all(np.isfinite(f)):
        raise ValueError("features must be finite")
    pooled = np.zeros((k, f.shape[1]), dtype=np.float64)
    if s.size:
        for j in range(k):
            rows = f[s == j]
            if rows.shape[0]:
                # Canonical (lexicographic) row order inside a slot: the mean
                # is mathematically unchanged but bitwise identical under
                # element permutations — determinism the tests pin down.
                order = np.lexsort(rows.T[::-1])
                pooled[j] = rows[order].mean(axis=0)
    return pooled.reshape(-1)


def collect_order_elements(
    orders: dict[int, tuple[Side, int]],
    sim: ZILobSimulator,
    spec: MultiLevelSpec,
) -> tuple[Array, IntArray]:
    """Encode the agent's resting orders as deep-set elements (paper Sec. 4.1).

    Each live order contributes the normalized pair ``(q / queue_scale,
    w / lots)``: ``q`` is the engine's FIFO queue position + 1 (paper's
    ``q - 1`` ahead convention) and ``w = 1`` (unit-lot engine — a paper order
    of ``w`` lots is ``w`` unit orders here, so the pooled feature reads the
    mean queue position per level; documented adaptation). The slot index is
    the price level ``0..K-1`` per side (orders deeper than ``K-1`` ticks are
    clamped into the boundary slot, mirroring the kappa ``K+1``-th bucket);
    buy slots are ``0..K-1``, sell slots ``K..2K-1``.
    """
    if not isinstance(sim, ZILobSimulator):
        raise TypeError("sim must be a ZILobSimulator")
    if not isinstance(spec, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    bb, ba = sim.best_bid_level, sim.best_ask_level
    if bb is None or ba is None:
        raise ValueError("book must be two-sided to encode orders")
    feats: list[list[float]] = []
    slots: list[int] = []
    k = spec.n_levels
    for oid, (side, lvl) in orders.items():
        qp = sim.queue_position(oid)
        if qp is None:  # raced fill/cancel between snapshot and encoding
            continue
        offset = (bb - lvl) if side == "buy" else (lvl - ba)
        slot = min(max(offset, 0), k - 1) + (0 if side == "buy" else k)
        feats.append([(qp + 1) / spec.queue_scale, 1.0 / spec.lots])
        slots.append(slot)
    f = np.asarray(feats, dtype=np.float64).reshape(-1, 2)
    s = np.asarray(slots, dtype=np.int64)
    return f, s


def kappa_fractions(
    orders: dict[int, tuple[Side, int]],
    sim: ZILobSimulator,
    spec: MultiLevelSpec,
) -> Array:
    """Resting-order fractions ``kappa`` per level (paper Eq. 4).

    ``kappa[b, k]`` (``k = 0..K-1``) is the fraction of the ``M``-lot budget
    resting ``k`` ticks below the best bid; ``kappa[b, K]`` covers orders at
    least ``K`` ticks below; the ask half mirrors it. Vector length ``2(K+1)``.
    """
    if not isinstance(spec, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    k = spec.n_levels
    counts = np.zeros(2 * (k + 1), dtype=np.float64)
    bb, ba = sim.best_bid_level, sim.best_ask_level
    if bb is not None and ba is not None:
        for oid, (side, lvl) in orders.items():
            if not sim.order_alive(oid):  # zombie: a ZI cancel removed it
                continue
            offset = (bb - lvl) if side == "buy" else (lvl - ba)
            bucket = min(max(offset, 0), k)
            idx = bucket if side == "buy" else (k + 1) + bucket
            counts[idx] += 1.0
    return counts / float(spec.lots)


def build_base_features(
    spec: MultiLevelSpec,
    *,
    bid_ret: float,
    ask_ret: float,
    bid_volumes: Sequence[float] | Array,
    ask_volumes: Sequence[float] | Array,
    mid_drift: float,
    mo_imbalance: float,
    lo_rate: float,
    cxl_rate: float,
    t_frac: float,
    q_norm: float,
    kappa: Sequence[float] | Array,
) -> Array:
    """Assemble the non-encoded state slice (paper Sec. 3.1 + App. A.2).

    Scalars (``2K + 8``): best bid/ask returns relative to the first observed
    quotes (x100), the ``K`` bid + ``K`` ask level volumes over
    ``depth_scale``, the mid-price return over the last decision interval
    (x100), and the observable interval flows. **Deviation (documented)**: the
    composed ZI engine does not expose per-side LO placement/cancellation
    streams, so the paper's signed ``Delta L``/``Delta C`` are replaced by the
    observable aggregates ``lo_rate`` (LO arrivals per second relative to the
    ZI baseline intensity) and ``cxl_rate`` (cancellations per second relative
    to ``theta * mean depth``); ``mo_imbalance`` is the exact signed MO
    imbalance (buys - sells over total) since trades carry aggressor tags.
    Then ``t_frac`` and the inventory ``q_norm = Q / M``. The ``kappa`` block
    (``2(K+1)``) follows, matching paper Eq. 4.
    """
    if not isinstance(spec, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    k = spec.n_levels
    bv = np.asarray(bid_volumes, dtype=np.float64).ravel()
    av = np.asarray(ask_volumes, dtype=np.float64).ravel()
    if bv.shape != (k,) or av.shape != (k,):
        raise ValueError(
            f"level volumes must have {k} entries per side, got {bv.shape} / {av.shape}"
        )
    if not np.all(np.isfinite(bv)) or not np.all(np.isfinite(av)):
        raise ValueError("level volumes must be finite")
    if np.any(bv < 0.0) or np.any(av < 0.0):
        raise ValueError("level volumes must be non-negative")
    kp = np.asarray(kappa, dtype=np.float64).ravel()
    if kp.shape != (2 * (k + 1),):
        raise ValueError(f"kappa must have {2 * (k + 1)} entries, got {kp.shape}")
    if not np.all(np.isfinite(kp)) or np.any(kp < 0.0) or np.any(kp > 1.0 + 1e-9):
        raise ValueError("kappa entries must lie in [0, 1]")
    out = np.concatenate(
        [
            [_finite(bid_ret, "bid_ret"), _finite(ask_ret, "ask_ret")],
            bv / spec.depth_scale,
            av / spec.depth_scale,
            [
                _finite(mid_drift, "mid_drift"),
                _finite(mo_imbalance, "mo_imbalance"),
                _nonneg_finite(lo_rate, "lo_rate"),
                _nonneg_finite(cxl_rate, "cxl_rate"),
            ],
            [_prob(t_frac, "t_frac"), _finite(q_norm, "q_norm")],
            kp,
        ]
    )
    if out.shape != (spec.n_base_features,):  # pragma: no cover - length checked above
        raise RuntimeError(f"base features mis-shaped: {out.shape} != ({spec.n_base_features},)")
    if not np.all(np.isfinite(out)):
        raise ValueError("base features went non-finite")
    return out


@dataclass(frozen=True)
class MMObs:
    """One decision-time observation: raw context + feature blocks.

    ``base_features`` (``4K+10``) holds the normalized scalars and the kappa
    block; ``order_feats`` (``m x 2``) and ``order_slots`` (``m``) are the
    deep-set elements of :func:`collect_order_elements`. Raw ``t``, ``mid``,
    ``best_bid``, ``best_ask``, ``inventory``, ``tau`` accompany it for
    policy ergonomics — policies see exactly what the encoder sees.
    """

    t: float
    mid: float
    best_bid: float
    best_ask: float
    inventory: int
    tau: float
    base_features: Array
    order_feats: Array
    order_slots: IntArray


AllocationPolicy = Callable[[MMObs], Array]


class RandomAllocationPolicy:
    """Seeded uniform-random allocation policy (contrast arm).

    Draws independent uniforms on the ``2K+3`` components and normalizes — a
    maximally uninformed simplex baseline sharing the exact execution path of
    the learned policy. Deterministic given ``seed``.
    """

    def __init__(self, spec: MultiLevelSpec, *, seed: int = 0) -> None:
        if not isinstance(spec, MultiLevelSpec):
            raise TypeError("spec must be a MultiLevelSpec")
        _seed_int(seed)
        self._spec = spec
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: MMObs) -> Array:
        if not isinstance(obs, MMObs):
            raise TypeError("obs must be an MMObs")
        w = self._rng.random(self._spec.n_components)
        return w / float(w.sum())


# ---------------------------------------------------------------------------
# The torch-gated agent: deep-set encoder + logistic-normal actor + critic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiLevelMMConfig:
    """Hyperparameters of the actor-critic agent (paper App. A.1).

    Paper values: encoder ``f^o_phi`` a one-layer 2-node ReLU network (here
    ``encoder_hidden=()`` + ``emb_dim`` on the spec), actor ``f^m`` and critic
    ``f^V`` two hidden layers x 128 tanh nodes, Adam lr 5e-4, critic loss
    weight ``c_V = 0.5``, inventory penalty ``inv_gamma = 0.01`` (paper Eq. 5),
    terminal position fraction ``nu`` (paper Sec. 3.3). Defaults below shrink
    the hidden widths for test budgets; the paper's are recoverable by
    passing ``hidden_actor=(128, 128)`` / ``hidden_critic=(128, 128)``.
    """

    hidden_actor: tuple[int, ...] = (64, 64)
    hidden_critic: tuple[int, ...] = (64, 64)
    encoder_hidden: tuple[int, ...] = ()
    lr: float = 5e-4
    critic_coef: float = 0.5
    inv_gamma: float = 0.01
    terminal_nu: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name, widths in (
            ("hidden_actor", self.hidden_actor),
            ("hidden_critic", self.hidden_critic),
            ("encoder_hidden", self.encoder_hidden),
        ):
            ws = tuple(int(h) for h in widths)
            if any(h < 1 for h in ws):
                raise ValueError(f"{name} widths must be positive ints, got {widths!r}")
            object.__setattr__(self, name, ws)
        if not self.hidden_actor or not self.hidden_critic:
            raise ValueError("hidden_actor and hidden_critic must be non-empty")
        _pos_finite(self.lr, "lr")
        _pos_finite(self.critic_coef, "critic_coef")
        _nonneg_finite(self.inv_gamma, "inv_gamma")
        _prob(self.terminal_nu, "terminal_nu")
        _seed_int(self.seed)


@dataclass(frozen=True)
class TrajectoryStep:
    """One acted decision of a collected episode.

    ``reward`` accumulates the Eq. 5 interval rewards earned while ``action``
    stood: normally one decision interval, but skipped decisions (one-sided
    book) fold their intervals into the standing action's step so the
    returns-to-go stay exact (documented adaptation — the paper's grid never
    skips).
    """

    obs: MMObs
    action: Array  # sampled simplex weights, length n_components
    reward: float  # paper Eq. 5 reward accumulated while the action stood


class MultiLevelMMAgent:
    """Actor-critic agent: deep-set order encoder + logistic-normal policy.

    Forward path (paper Sec. 4): each resting order's ``(q, w)`` features go
    through the shared embedding net ``f^o_phi`` (phi), are mean-pooled inside
    their price-level slot (permutation-invariant — reordering the order set
    leaves the encoding identical), concatenated with the scalar/kappa base
    features, and read by the actor head ``f^m`` (the ``2(K+1)`` Gaussian
    logits) and critic head ``f^V``. The covariance is the paper's
    state-independent ``Diag(exp(theta_v))``; the actor output layer starts
    with bias ``(1,...,1)`` and orthogonal-gain-1e-5 weights, so initial
    allocations are near-uniform over the trade components and ``a_0`` starts
    unlikely (paper App. A.1). ``learn`` applies the combined loss of paper
    Eq. 14 with returns-to-go advantages (Eq. 11).
    """

    def __init__(self, spec: MultiLevelSpec, config: MultiLevelMMConfig) -> None:
        torch = _torch()
        if not isinstance(spec, MultiLevelSpec):
            raise TypeError("spec must be a MultiLevelSpec")
        if not isinstance(config, MultiLevelMMConfig):
            raise TypeError("config must be a MultiLevelMMConfig")
        self._spec = spec
        self._cfg = config
        self._torch = torch
        torch.manual_seed(int(config.seed))
        torch.set_num_threads(1)
        self._rng = np.random.default_rng(config.seed + 1543)
        self._n_updates = 0

        def _mlp(d_in: int, hidden: tuple[int, ...], d_out: int, act: Any) -> Any:
            mods: list[Any] = []
            d = d_in
            for h in hidden:
                mods += [torch.nn.Linear(d, h), act()]
                d = h
            mods.append(torch.nn.Linear(d, d_out))
            return torch.nn.Sequential(*mods)

        # Encoder f^o_phi: paper has one layer -> Linear(2, emb) + ReLU.
        if config.encoder_hidden:
            self._phi = _mlp(2, config.encoder_hidden, spec.emb_dim, torch.nn.ReLU)
        else:
            self._phi = torch.nn.Sequential(torch.nn.Linear(2, spec.emb_dim), torch.nn.ReLU())
        self._actor = _mlp(spec.state_dim, config.hidden_actor, spec.n_logits, torch.nn.Tanh)
        self._critic = _mlp(spec.state_dim, config.hidden_critic, 1, torch.nn.Tanh)
        self._theta_v = torch.nn.Parameter(torch.zeros(spec.n_logits))
        # Initialization (paper App. A.1): orthogonal gain sqrt(2) everywhere
        # except the actor output layer (gain 1e-5, bias = 1).
        for mod in (self._phi, self._actor, self._critic):
            for layer in mod:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                    torch.nn.init.zeros_(layer.bias)
        out_layer = self._actor[-1]
        torch.nn.init.orthogonal_(out_layer.weight, gain=1e-5)
        with torch.no_grad():
            out_layer.bias.fill_(1.0)
        params: list[Any] = (
            list(self._phi.parameters())
            + list(self._actor.parameters())
            + list(self._critic.parameters())
            + [self._theta_v]
        )
        self._opt = torch.optim.Adam(params, lr=float(config.lr))

    # -- plumbing -----------------------------------------------------------

    @property
    def spec(self) -> MultiLevelSpec:
        return self._spec

    @property
    def config(self) -> MultiLevelMMConfig:
        return self._cfg

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def _check_obs(self, obs: MMObs) -> None:
        if not isinstance(obs, MMObs):
            raise TypeError("obs must be an MMObs")
        b = np.asarray(obs.base_features, dtype=np.float64).ravel()
        f = np.asarray(obs.order_feats, dtype=np.float64)
        s = np.asarray(obs.order_slots, dtype=np.int64).ravel()
        if b.shape != (self._spec.n_base_features,):
            raise ValueError(
                f"base_features must have shape ({self._spec.n_base_features},), got {b.shape}"
            )
        if f.ndim != 2 or f.shape[1] != 2 or f.shape[0] != s.shape[0]:
            raise ValueError(
                f"order_feats must be (m, 2) matching order_slots, got {f.shape} / {s.shape}"
            )
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(f)):
            raise ValueError("observation features must be finite")

    def _encode(self, obs_list: Sequence[MMObs]) -> Any:
        """Batch observations to the (B, state_dim) encoded state tensor."""
        torch = self._torch
        rows: list[Any] = []
        k, emb = self._spec.n_levels, self._spec.emb_dim
        for obs in obs_list:
            self._check_obs(obs)
            feats = torch.as_tensor(
                np.asarray(obs.order_feats, dtype=np.float64), dtype=torch.float32
            )
            slots = np.asarray(obs.order_slots, dtype=np.int64).ravel()
            z = self._phi(feats)  # (m, emb)
            pooled = torch.zeros(2 * k, emb)
            for j in range(2 * k):
                mask = torch.as_tensor(slots == j)
                if bool(mask.any()):
                    pooled[j] = z[mask].mean(dim=0)
            base = torch.as_tensor(
                np.asarray(obs.base_features, dtype=np.float64), dtype=torch.float32
            )
            rows.append(torch.cat([pooled.reshape(-1), base]))
        return torch.stack(rows)

    def mu_logits(self, obs_list: Sequence[MMObs]) -> Array:
        """Actor mean logits ``mu_theta(s)`` per observation, eval-mode."""
        torch = self._torch
        with torch.no_grad():
            out = self._actor(self._encode(obs_list))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)

    def values(self, obs_list: Sequence[MMObs]) -> Array:
        """Critic value estimate ``V(s)`` per observation, eval-mode."""
        torch = self._torch
        with torch.no_grad():
            out = self._critic(self._encode(obs_list)).squeeze(-1)
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float64)

    def log_probs(self, obs_list: Sequence[MMObs], actions: Array) -> Array:
        """Logistic-normal log ``pi(a|s)`` per (obs, action) pair, eval-mode."""
        torch = self._torch
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] != len(obs_list) or a.shape[1] != self._spec.n_components:
            raise ValueError(f"actions must be (n_obs, {self._spec.n_components}), got {a.shape}")
        if np.any(a <= 0.0) or not np.all(np.isfinite(a)):
            raise ValueError("actions must be strictly positive simplex points")
        if not np.allclose(a.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("actions must sum to 1")
        with torch.no_grad():
            lp = self._log_prob_t(obs_list, torch.as_tensor(a, dtype=torch.float32))
        return np.asarray(lp.detach().cpu().numpy(), dtype=np.float64)

    # -- action -------------------------------------------------------------

    def _weights_from_mu(self, obs: MMObs, z: Array | None) -> Array:
        mu = self.mu_logits([obs])[0]
        logvar = np.asarray(self._theta_v.detach().cpu().numpy(), dtype=np.float64)
        x = mu if z is None else mu + np.exp(0.5 * logvar) * z
        return simplex_transform(x)

    def act(
        self,
        obs: MMObs,
        *,
        rng: np.random.Generator | None = None,
        sample: bool = True,
    ) -> tuple[Array, IntArray]:
        """Produce (simplex weights, integer lots) for one observation.

        ``sample=True`` draws the paper's stochastic policy ``a ~ pi`` using
        ``rng`` (or the agent's seeded stream when ``rng`` is None);
        ``sample=False`` returns the deterministic mean-logit action ``h(mu)``
        — a documented eval convenience, NOT the exact logistic-normal mean
        (which has no closed form, paper Eq. 9 context). Lots come from
        :func:`hamilton_apportionment`.
        """
        self._check_obs(obs)
        z: Array | None = None
        if sample:
            gen = rng if rng is not None else self._rng
            if not isinstance(gen, np.random.Generator):
                raise TypeError("rng must be a numpy Generator")
            z = gen.standard_normal(self._spec.n_logits)
        w = self._weights_from_mu(obs, z)
        return w, hamilton_apportionment(w, self._spec.lots)

    # -- learning -------------------------------------------------------------

    def _log_prob_t(self, obs_list: Sequence[MMObs], actions: Any) -> Any:
        """Differentiable log pi(a|s) for a batched simplex action tensor."""
        torch = self._torch
        enc = self._encode(obs_list)
        mu = self._actor(enc)
        eps = torch.tensor(1e-8, dtype=torch.float32)
        a = torch.clamp(actions, min=float(eps))
        y = torch.log(a[:, 1:] / a[:, :1])  # ALR map, reference component a_0
        var = torch.exp(self._theta_v)
        d = self._spec.n_logits
        log_n = -0.5 * (
            d * math.log(2.0 * math.pi) + self._theta_v.sum() + (((y - mu) ** 2) / var).sum(dim=1)
        )
        return log_n - torch.log(a).sum(dim=1)

    def evaluate_loss(
        self, episodes: Sequence[tuple[Sequence[TrajectoryStep], float]]
    ) -> dict[str, float]:
        """Loss components on a trajectory batch without a gradient step.

        ``episodes`` is a list of ``(steps, terminal_g)`` pairs — the paper's
        trajectory tuple (their Eq. 13): per-step (obs, action, reward) plus
        the terminal reward ``g``. Returns-to-go and advantages follow Eqs.
        10-11; the terms are the two halves of the Eq. 14 loss.
        """
        obs_list, act_list, adv_list, ret_list = self._episode_batch(episodes)
        torch = self._torch
        enc = self._encode(obs_list)
        with torch.no_grad():
            v = self._critic(enc).squeeze(-1)
        ret = torch.as_tensor(np.asarray(ret_list, dtype=np.float64), dtype=torch.float32)
        adv = torch.as_tensor(np.asarray(adv_list, dtype=np.float64), dtype=torch.float32)
        acts = torch.as_tensor(np.stack(act_list), dtype=torch.float32)
        with torch.no_grad():
            lp = self._log_prob_t(obs_list, acts)
        critic_mse = float(torch.mean((v - ret) ** 2).detach().cpu().numpy())
        actor_term = float(torch.mean(adv * lp).detach().cpu().numpy())
        total = -actor_term + self._cfg.critic_coef * critic_mse
        return {"actor_term": actor_term, "critic_mse": critic_mse, "total": float(total)}

    def _episode_batch(
        self, episodes: Sequence[tuple[Sequence[TrajectoryStep], float]]
    ) -> tuple[list[MMObs], list[Array], list[float], list[float]]:
        if len(episodes) == 0:
            raise ValueError("episodes must be non-empty")
        obs_list: list[MMObs] = []
        act_list: list[Array] = []
        adv_list: list[float] = []
        ret_list: list[float] = []
        for steps, g_term in episodes:
            if len(steps) == 0:
                raise ValueError("each episode must contain >= 1 step")
            g = _finite(g_term, "terminal_g")
            rewards = np.asarray([s.reward for s in steps], dtype=np.float64)
            if not np.all(np.isfinite(rewards)):
                raise ValueError("episode rewards must be finite")
            # Undiscounted returns-to-go incl. the terminal reward (Eq. 11).
            rets = np.cumsum(rewards[::-1])[::-1] + g
            v_hat = self.values([s.obs for s in steps])
            for step, ret in zip(steps, rets, strict=True):
                a = np.asarray(step.action, dtype=np.float64).ravel()
                if a.shape != (self._spec.n_components,):
                    raise ValueError(
                        f"action must have {self._spec.n_components} components, got {a.shape}"
                    )
                obs_list.append(step.obs)
                act_list.append(a)
                adv_list.append(float(ret) - float(v_hat[len(ret_list)]))
                ret_list.append(float(ret))
        return obs_list, act_list, adv_list, ret_list

    def learn(self, episodes: Sequence[tuple[Sequence[TrajectoryStep], float]]) -> float:
        """One combined actor+critic gradient step (paper Eq. 14).

        ``-(1/tauN) sum A(s,a) log pi(a|s) + c_V (1/tauN) sum (V(s) - ret)^2``
        over every step of the collected episodes; advantages use the frozen
        critic estimate (detached), matching the paper's estimator.
        """
        obs_list, act_list, adv_list, ret_list = self._episode_batch(episodes)
        torch = self._torch
        enc = self._encode(obs_list)
        v = self._critic(enc).squeeze(-1)
        ret = torch.as_tensor(np.asarray(ret_list, dtype=np.float64), dtype=torch.float32)
        adv = torch.as_tensor(np.asarray(adv_list, dtype=np.float64), dtype=torch.float32)
        acts = torch.as_tensor(np.stack(act_list), dtype=torch.float32)
        lp = self._log_prob_t(obs_list, acts)
        loss = -(adv * lp).mean() + self._cfg.critic_coef * ((v - ret) ** 2).mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("actor-critic loss went non-finite")
        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        self._n_updates += 1
        return float(loss.detach().cpu().numpy())


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Session runner: allocations into the ZI engine
# ---------------------------------------------------------------------------


def _resolve_spec(
    spec: MultiLevelSpec | None, agent: MultiLevelMMAgent | None, *, mode: str
) -> MultiLevelSpec | None:
    if agent is not None:
        if spec is not None and spec != agent.spec:
            raise ValueError("spec must be omitted or equal to agent.spec")
        return agent.spec
    if mode == "allocation" and spec is None:
        raise ValueError("spec is required with an allocation_policy")
    if spec is not None and not isinstance(spec, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    return spec


class _MMExec:
    """Mutable execution state for one :func:`run_multilevel_mm_session` call.

    Holds the engine plus all session bookkeeping (inventory/cash, counters,
    interval accumulators, interval-stat snapshots, the standing allocation's
    open trajectory step). Methods implement the paper's execution semantics:
    market legs inject unit-by-unit with self-trade prevention (the engine has
    no STP flag — a unit whose fill would consume the MM's own front-of-queue
    order is skipped), and limit legs realloc to per-level target counts —
    FIFO tail cancelled first, deficits posted fresh (the paper's
    reallocation rule).
    """

    def __init__(
        self,
        *,
        sim: ZILobSimulator,
        config: ZILobConfig,
        spec: MultiLevelSpec | None,
        mode: str,
        agent: MultiLevelMMAgent | None,
        allocation_policy: AllocationPolicy | None,
        policy: QuotePolicy | None,
        horizon: float,
        decision_interval: float,
        sample_interval: float,
        inventory_cap: int | None,
        inv_gamma: float,
        terminal_nu: float,
        training: bool,
        eval_sample: bool,
        return_trajectory: bool,
        seed: int,
    ) -> None:
        self.sim = sim
        self.cfg = config
        self.sp = spec
        self.mode = mode
        self.agent = agent
        self.allocation_policy = allocation_policy
        self.policy = policy
        self.h = horizon
        self.di = decision_interval
        self.si = sample_interval
        self.cap = inventory_cap
        self.g_inv = inv_gamma
        self.nu = terminal_nu
        self.training = training
        self.eval_sample = eval_sample
        self.return_trajectory = return_trajectory
        self.act_rng = np.random.default_rng(seed + 911_912)
        self.m_lots = float(spec.lots if spec is not None else 1)
        # Order-flow state.
        self.inventory = 0
        self.cash = 0.0
        self.self_orders: dict[int, tuple[Side, int]] = {}
        self.own_mo_trades: set[int] = set()  # trades from the MM's own MOs
        self.trade_cursor = 0
        # Counters.
        self.n_fills = self.n_fills_bid = self.n_fills_ask = 0
        self.n_mo_fills = self.n_mo_injected = self.n_cancels = 0
        self.n_kept = self.n_posted = 0
        self.n_clips = self.n_skipped = self.n_gated = 0
        self.n_decisions = self.n_self_blocked = 0
        self.terminal_lots = 0
        self.terminal_shortfall = 0
        # Paths and diagnostics.
        self.inv_path: list[int] = []
        self.inv_times: list[float] = []
        self.mtm_path: list[float] = []
        self.samples: list[Any] = [sim.sample()]
        self.signs: list[float] = []
        self.fill_waits: list[float] = []
        self.queue_ahead_fills: list[int] = []
        self.max_abs_inv = 0
        self.last_mid: float | None = sim.mid
        self.p0_bid: float | None = None
        self.p0_ask: float | None = None
        # Interval accumulators / snapshots.
        self.interval_cash = 0.0
        self.interval_mo_buy = 0
        self.interval_mo_sell = 0
        # (n_lo_arrivals, n_cancellations, own n_cancels) at the last settle;
        # the engine counts the MM's own cancels inside n_cancellations, so
        # the ZI-only delta subtracts the own-cancel delta.
        self.counters_prev = (0, 0, 0)
        self.depth_time = 0.0
        self.depth_weighted = 0.0
        self.t_prev_decision = 0.0
        self.st_lo_rate = 0.0
        self.st_cxl_rate = 0.0
        self.st_mo_imb = 0.0
        self.st_mid_prev: float | None = None
        self.prev_mid: float | None = None
        # Reward / trajectory bookkeeping.
        self.rewards: list[float] = []
        self.alloc_weight_hist: list[Array] = []
        self.trajectory: list[TrajectoryStep] = []
        self.have_prev = False
        self.q_prev = 0
        self.p_prev = 0.0
        self.open_obs: MMObs | None = None
        self.open_action: Array | None = None
        self.open_reward = 0.0
        self.gamma_penalty_total = 0.0
        self.cash_flow_total = 0.0
        self.q_n = 0
        self.q_plus = 0
        self.mo_cash = 0.0
        self.g_term = 0.0
        self.final_mid = 0.0

    # -- fills ----------------------------------------------------------------

    def drain_trades(self) -> None:
        """Apply new engine trades to the MM's cash/inventory and flow stats."""
        sim = self.sim
        while self.trade_cursor < len(sim.trades):
            idx = self.trade_cursor
            tr = sim.trades[idx]
            self.trade_cursor += 1
            if idx in self.own_mo_trades:
                continue  # the MM's own market leg: not exogenous ZI flow
            self.signs.append(1.0 if tr.aggressor == "buy" else -1.0)
            if tr.maker_tag == MM_TAG and tr.maker_order_id in self.self_orders:
                if tr.maker_side == "buy":
                    self.inventory += tr.qty
                    self.cash -= tr.price * tr.qty
                    self.interval_cash -= tr.price * tr.qty
                    self.n_fills_bid += 1
                else:
                    self.inventory -= tr.qty
                    self.cash += tr.price * tr.qty
                    self.interval_cash += tr.price * tr.qty
                    self.n_fills_ask += 1
                self.n_fills += 1
                self.queue_ahead_fills.append(tr.maker_queue_ahead_at_submit)
                self.fill_waits.append(tr.t - tr.maker_t_submit)
                self.self_orders.pop(tr.maker_order_id, None)
                self.max_abs_inv = max(self.max_abs_inv, abs(self.inventory))
            if tr.maker_tag != MM_TAG:
                # ZI-maker trade: the aggressor stream for flow features.
                if tr.aggressor == "buy":
                    self.interval_mo_buy += 1
                else:
                    self.interval_mo_sell += 1

    # -- observation ------------------------------------------------------------

    def observe(self) -> MMObs | None:
        """Build the decision-time observation, or None on a one-sided book."""
        sim = self.sim
        sp = self.sp
        if sp is None:
            return None
        bb_l, ba_l = sim.best_bid_level, sim.best_ask_level
        mid = sim.mid
        if bb_l is None or ba_l is None or mid is None:
            return None
        bb, ba = sim.level_to_price(bb_l), sim.level_to_price(ba_l)
        p0b, p0a = self.p0_bid, self.p0_ask
        if p0b is None or p0a is None:
            p0b, p0a = bb, ba
            self.p0_bid, self.p0_ask = p0b, p0a
        k = sp.n_levels
        bid_vols = np.asarray(
            [sim.depth_at("buy", bb_l - off) for off in range(k)], dtype=np.float64
        )
        ask_vols = np.asarray(
            [sim.depth_at("sell", ba_l + off) for off in range(k)], dtype=np.float64
        )
        mid_drift = (
            0.0
            if self.st_mid_prev is None or self.st_mid_prev <= 0.0
            else 100.0 * (mid - self.st_mid_prev) / self.st_mid_prev
        )
        feats, slots = collect_order_elements(self.self_orders, sim, sp)
        base = build_base_features(
            sp,
            bid_ret=100.0 * (bb - p0b) / p0b,
            ask_ret=100.0 * (ba - p0a) / p0a,
            bid_volumes=bid_vols,
            ask_volumes=ask_vols,
            mid_drift=mid_drift,
            mo_imbalance=self.st_mo_imb,
            lo_rate=self.st_lo_rate,
            cxl_rate=self.st_cxl_rate,
            t_frac=min(sim.t / self.h, 1.0),
            q_norm=self.inventory / float(sp.lots),
            kappa=kappa_fractions(self.self_orders, sim, sp),
        )
        return MMObs(
            t=sim.t,
            mid=mid,
            best_bid=bb,
            best_ask=ba,
            inventory=self.inventory,
            tau=self.h - sim.t,
            base_features=base,
            order_feats=feats,
            order_slots=slots,
        )

    # -- order management -------------------------------------------------------

    def cancel_oid(self, oid: int) -> None:
        # Engine-side cancellation may fail for zombie entries (a ZI
        # cancellation event already removed the order); the bookkeeping
        # entry is dropped either way.
        if self.sim.cancel_order(oid):
            self.n_cancels += 1
        self.self_orders.pop(oid, None)

    def reconcile_level(self, side: Side, level: int, target: int) -> None:
        mine: list[int] = []
        for oid in sorted(
            o for o, (sd, lv) in self.self_orders.items() if sd == side and lv == level
        ):
            if self.sim.order_alive(oid):
                mine.append(oid)
            else:
                # Zombie: a ZI cancellation event removed it from the book —
                # it must not count toward the keep set nor the kappa vector.
                self.self_orders.pop(oid, None)
        keep = min(len(mine), target)
        for oid in mine[keep:]:  # lowest priority = FIFO tail (latest posted)
            self.cancel_oid(oid)
        for _ in range(target - keep):
            oid = self.sim.submit_limit_order(side, self.sim.level_to_price(level), MM_TAG)
            self.self_orders[oid] = (side, level)
            self.n_posted += 1
        self.n_kept += keep

    def reconcile_to(self, targets: dict[tuple[Side, int], int]) -> None:
        """Realloc resting orders to per-level target counts.

        Levels absent from ``targets`` are fully cancelled; within a level the
        FIFO tail (lowest queue priority) is cancelled first and deficits are
        posted as fresh unit orders — the paper's reallocation rule.
        """
        merged = dict(targets)
        for sd, lv in set(self.self_orders.values()):
            merged.setdefault((sd, lv), 0)
        for (sd, lv), tgt in merged.items():
            self.reconcile_level(sd, lv, max(int(tgt), 0))

    def front_is_own(self, side: Side) -> bool:
        """True when a market ``side`` unit would fill the MM's own order."""
        lvl = self.sim.best_ask_level if side == "buy" else self.sim.best_bid_level
        if lvl is None:
            return False
        opp: Side = "sell" if side == "buy" else "buy"
        return any(
            sd == opp and lv == lvl and self.sim.queue_position(oid) == 0
            for oid, (sd, lv) in self.self_orders.items()
        )

    def inject_stp(self, side: Side, qty: int) -> None:
        """Inject a market leg unit-by-unit with self-trade prevention."""
        for _ in range(qty):
            if self.front_is_own(side):
                self.n_self_blocked += 1
                continue
            self.n_mo_injected += 1
            n0 = len(self.sim.trades)
            trs = self.sim.inject_market_order(side, 1)
            self.own_mo_trades.update(range(n0, len(self.sim.trades)))
            for tr in trs:
                px = tr.price * tr.qty
                if side == "buy":
                    self.inventory += tr.qty
                    self.cash -= px
                    self.interval_cash -= px
                else:
                    self.inventory -= tr.qty
                    self.cash += px
                    self.interval_cash += px
                self.n_mo_fills += 1
                self.max_abs_inv = max(self.max_abs_inv, abs(self.inventory))

    def execute(self, lots: IntArray) -> None:
        """Execute one simplex action: market legs then limit-leg realloc."""
        sim = self.sim
        sp = self.sp
        if sp is None:
            raise RuntimeError("allocation execution requires a MultiLevelSpec")
        k = sp.n_levels
        bb_l, ba_l = sim.best_bid_level, sim.best_ask_level
        if bb_l is None or ba_l is None:
            return
        lv = np.asarray(lots, dtype=np.int64).ravel()
        # Post-hoc inventory gating (paper's hard bound, applied post-action).
        if self.cap is not None and self.inventory >= self.cap:
            lv[1] = 0
            lv[2 : k + 2] = 0
            self.n_gated += 1
        if self.cap is not None and self.inventory <= -self.cap:
            lv[k + 2] = 0
            lv[k + 3 :] = 0
            self.n_gated += 1
        # Market legs first (the paper's action executes immediately); the
        # limit-leg anchors stay pinned to the pre-action best quotes.
        if int(lv[1]) > 0:
            self.inject_stp("buy", int(lv[1]))
        if int(lv[k + 2]) > 0:
            self.inject_stp("sell", int(lv[k + 2]))
        targets: dict[tuple[Side, int], int] = {}
        for off in range(k):
            if int(lv[2 + off]) > 0:
                targets[("buy", bb_l - off)] = int(lv[2 + off])
            if int(lv[k + 3 + off]) > 0:
                targets[("sell", ba_l + off)] = int(lv[k + 3 + off])
        self.reconcile_to(targets)

    # -- decisions ----------------------------------------------------------------

    def record_path(self) -> None:
        mid = self.sim.mid
        if mid is not None:
            self.last_mid = mid
        self.inv_path.append(self.inventory)
        self.inv_times.append(self.sim.t)
        self.mtm_path.append(self.cash + self.inventory * mid if mid is not None else float("nan"))

    def close_open_step(self) -> None:
        """Finalize the standing allocation's trajectory step."""
        if self.open_action is not None and self.open_obs is not None:
            self.trajectory.append(
                TrajectoryStep(
                    obs=self.open_obs,
                    action=self.open_action,
                    reward=self.open_reward,
                )
            )
        self.open_obs, self.open_action, self.open_reward = None, None, 0.0

    def act_allocation(self, obs: MMObs) -> Array:
        """Sample the simplex allocation and execute its lot legs."""
        sp = self.sp
        if sp is None:
            raise RuntimeError("allocation modes require a MultiLevelSpec")
        self.n_decisions += 1
        if self.mode == "agent":
            if self.agent is None:
                raise RuntimeError("agent mode requires an agent")
            w, lots = self.agent.act(
                obs, rng=self.act_rng, sample=(self.training or self.eval_sample)
            )
        else:
            if self.allocation_policy is None:
                raise RuntimeError("allocation mode requires an allocation_policy")
            w = np.asarray(self.allocation_policy(obs), dtype=np.float64).ravel()
            if w.shape != (sp.n_components,):
                raise ValueError(
                    f"allocation_policy must return {sp.n_components} weights, got {w.shape}"
                )
            lots = hamilton_apportionment(w, sp.lots)
        self.alloc_weight_hist.append(w)
        self.execute(lots)
        return np.asarray(w, dtype=np.float64)

    def act_policy(self) -> None:
        """Classic QuotePolicy arm: one smart-quoted unit order per side."""
        sim = self.sim
        if self.policy is None:
            raise RuntimeError("policy mode requires a policy")
        mid = sim.mid
        bb_l, ba_l = sim.best_bid_level, sim.best_ask_level
        if mid is None or bb_l is None or ba_l is None:
            self.n_skipped += 1
            return
        self.n_decisions += 1
        bb, ba = sim.level_to_price(bb_l), sim.level_to_price(ba_l)
        st = MMState(
            t=sim.t,
            mid=mid,
            best_bid=bb,
            best_ask=ba,
            inventory=self.inventory,
            tau=max(self.h - sim.t, 0.0),
        )
        bid_px, ask_px = self.policy(st)
        targets: dict[tuple[Side, int], int] = {}
        tick = self.cfg.tick
        if bid_px is not None and (self.cap is None or self.inventory < self.cap):
            bp = min(float(bid_px), ba - tick)
            if bp < float(bid_px):
                self.n_clips += 1
            if bp > 0.0:
                targets[("buy", sim.price_to_level(bp))] = 1
        if ask_px is not None and (self.cap is None or self.inventory > -self.cap):
            ap = max(float(ask_px), bb + tick)
            if ap > float(ask_px):
                self.n_clips += 1
            targets[("sell", sim.price_to_level(ap))] = 1
        self.reconcile_to(targets)

    def settle_and_act(self, done: bool) -> None:
        """Close the open interval's Eq. 5 reward, then observe and act.

        The potential points ``(q_prev, p_prev)`` are always stored at the
        pre-action boundary state, so the ``Q p`` terms telescope exactly over
        the episode (paper Eq. 7) regardless of market-leg fills. Skipped
        decisions (one-sided book) leave the standing allocation — and its
        open trajectory step, which keeps accruing interval rewards — in
        place.
        """
        sim = self.sim
        mid_now = sim.mid if sim.mid is not None else self.last_mid
        if self.have_prev and mid_now is not None and self.p_prev > 0.0:
            r = mm_reward(
                cash_flow=self.interval_cash,
                q_prev=self.q_prev,
                p_prev=self.p_prev,
                q_next=self.inventory,
                p_next=mid_now,
                inv_gamma=self.g_inv,
                lots=int(self.m_lots),
            )
            self.rewards.append(r)
            self.gamma_penalty_total += self.g_inv * abs(self.inventory)
            self.cash_flow_total += self.interval_cash
            if self.open_action is not None:
                self.open_reward += r
        # Snapshot the just-ended interval's flow stats for the next obs.
        dt_i = max(sim.t - self.t_prev_decision, 1e-9)
        mean_depth = (
            self.depth_weighted / self.depth_time if self.depth_time > 0 else float(sim.total_depth)
        )
        lo_delta = sim.n_lo_arrivals - self.counters_prev[0]
        cxl_delta = (sim.n_cancellations - self.counters_prev[1]) - (
            self.n_cancels - self.counters_prev[2]
        )
        mo_tot = self.interval_mo_buy + self.interval_mo_sell
        self.st_mo_imb = (self.interval_mo_buy - self.interval_mo_sell) / max(mo_tot, 1)
        self.st_lo_rate = (lo_delta / dt_i) / max(2.0 * self.cfg.lam * self.cfg.band, 1e-9)
        self.st_cxl_rate = (max(float(cxl_delta), 0.0) / dt_i) / max(
            self.cfg.theta_cxl * max(mean_depth, 1.0), 1e-9
        )
        # Reset interval accumulators; store the pre-action potential point.
        self.interval_cash = 0.0
        self.interval_mo_buy = self.interval_mo_sell = 0
        self.counters_prev = (sim.n_lo_arrivals, sim.n_cancellations, self.n_cancels)
        self.depth_time = 0.0
        self.depth_weighted = 0.0
        self.t_prev_decision = sim.t
        self.st_mid_prev = self.prev_mid
        self.prev_mid = mid_now
        if mid_now is not None:
            self.have_prev = True
            self.q_prev, self.p_prev = self.inventory, mid_now
        if done:
            self.close_open_step()
            return
        if self.mode == "policy":
            self.act_policy()
            return
        obs = self.observe()
        if obs is None:
            self.n_skipped += 1
            return
        w = self.act_allocation(obs)
        if self.training or self.return_trajectory:
            self.close_open_step()
            self.open_obs, self.open_action, self.open_reward = obs, w, 0.0

    # -- main loop ----------------------------------------------------------------

    def run(self) -> None:
        """Drive the simulator to the horizon, then run terminal liquidation."""
        sim = self.sim
        self.settle_and_act(done=False)  # initial decision at t = 0
        self.record_path()
        next_decision = self.di
        next_sample = self.si
        while sim.t < self.h:
            t0 = sim.t
            sim.step()
            dt_step = sim.t - t0
            self.depth_time += dt_step
            self.depth_weighted += sim.total_depth * dt_step
            self.drain_trades()
            if sim.t >= next_decision:
                self.settle_and_act(done=False)
                self.record_path()
                while next_decision <= sim.t:
                    next_decision += self.di
            if sim.t >= next_sample:
                self.samples.append(sim.sample())
                while next_sample <= sim.t:
                    next_sample += self.si
        self._terminal()

    def _terminal(self) -> None:
        """Paper Sec. 3.3: settle, pull quotes, liquidate excess, score g."""
        sim = self.sim
        self.drain_trades()
        self.settle_and_act(done=True)  # last interval at pre-liquidation state
        for oid in sorted(self.self_orders, key=int, reverse=True):
            self.cancel_oid(oid)  # pull remaining quotes (no self-trades)
        self.q_n = self.inventory
        if self.sp is not None:
            limit = terminal_position_limit(self.sp.lots, self.nu)
        else:
            limit = self.cap if self.cap is not None else abs(self.q_n)
        excess = abs(self.q_n) - limit
        if excess > 0:
            side_t: Side = "sell" if self.q_n > 0 else "buy"
            n0 = len(sim.trades)
            for tr in sim.inject_market_order(side_t, excess):
                px = tr.price * tr.qty
                self.mo_cash += px if side_t == "sell" else -px
                self.cash += px if side_t == "sell" else -px
                self.inventory += -tr.qty if side_t == "sell" else tr.qty
                self.terminal_lots += tr.qty
            self.own_mo_trades.update(range(n0, len(sim.trades)))
            self.terminal_shortfall = excess - self.terminal_lots
        self.q_plus = self.inventory if excess > 0 else self.q_n
        final_mid = sim.mid if sim.mid is not None else self.last_mid
        if final_mid is None:  # pragma: no cover - seeded books are defined
            raise RuntimeError("terminal mid undefined; cannot settle the episode")
        self.final_mid = final_mid
        self.g_term = terminal_reward(
            q_n=self.q_n,
            p_n=final_mid,
            q_plus=self.q_plus,
            mo_cash=self.mo_cash,
            lots=int(self.m_lots),
        )
        self.record_path()
        self.samples.append(sim.sample())
        self.drain_trades()


def run_multilevel_mm_session(
    *,
    config: ZILobConfig,
    horizon: float,
    agent: MultiLevelMMAgent | None = None,
    allocation_policy: AllocationPolicy | None = None,
    policy: QuotePolicy | None = None,
    spec: MultiLevelSpec | None = None,
    training: bool = False,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    flow: MarkovRegimeFlow | None = None,
    inventory_cap: int | None = None,
    inv_gamma: float = 0.01,
    terminal_nu: float = 1.0,
    eval_sample: bool = True,
    return_trajectory: bool = False,
) -> dict[str, Any]:
    """Run one multi-level market-making session on the ZI engine.

    ``run_mm_session``-style accounting extended to the paper's simplex
    actions: every ``decision_interval`` simulated seconds the policy produces
    a ``2K+3`` allocation; market legs execute via ``inject_market_order``
    (immediate fills at book prices, unit-by-unit with self-trade prevention)
    and limit legs are reconciled level-by-level — resting orders whose
    absolute level stays targeted keep their FIFO position, surplus orders
    are cancelled lowest-priority-first (the paper's reallocating rule), and
    deficits are posted as new unit orders. Post-hoc inventory gating (the
    paper's hard bound): at ``inventory >= cap`` all buy legs are dropped, at
    ``<= -cap`` all sell legs. Per-decision rewards are the shaped Eq. 5
    signal; the terminal Eq. 6 liquidation is applied at the horizon. Exactly
    one of ``agent`` / ``allocation_policy`` / ``policy`` must be given;
    ``training=True`` requires the agent.

    Returns a SYNTHETIC diagnostic bundle; ``sim_internal_*`` keys are
    simulator-internal accounting, NEVER headline metrics or market evidence.
    No live-trading claim.
    """
    n_given = sum(x is not None for x in (agent, allocation_policy, policy))
    if n_given != 1:
        raise ValueError("exactly one of agent / allocation_policy / policy must be given")
    if agent is not None and not isinstance(agent, MultiLevelMMAgent):
        raise TypeError("agent must be a MultiLevelMMAgent")
    if allocation_policy is not None and not callable(allocation_policy):
        raise TypeError("allocation_policy must be callable")
    if policy is not None and not callable(policy):
        raise TypeError("policy must be callable")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    if flow is not None and not isinstance(flow, MarkovRegimeFlow):
        raise TypeError("flow must be a MarkovRegimeFlow or None")
    if training and agent is None:
        raise ValueError("training=True requires an agent")
    mode = (
        "agent"
        if agent is not None
        else ("allocation" if allocation_policy is not None else "policy")
    )
    sp = _resolve_spec(spec, agent, mode=mode)
    h = _pos_finite(horizon, "horizon")
    di = _pos_finite(decision_interval, "decision_interval")
    si = _pos_finite(sample_interval, "sample_interval")
    g_inv = _nonneg_finite(inv_gamma, "inv_gamma")
    nu = _prob(terminal_nu, "terminal_nu")
    cap: int | None = None
    if inventory_cap is not None:
        cap = _int_at_least(inventory_cap, 1, "inventory_cap")
    elif agent is not None:
        cap = agent.spec.inventory_cap
    if agent is not None and cap != agent.spec.inventory_cap:
        raise ValueError(
            f"inventory_cap ({cap}) must equal agent.spec.inventory_cap "
            f"({agent.spec.inventory_cap})"
        )

    ex = _MMExec(
        sim=ZILobSimulator(config, flow=flow),
        config=config,
        spec=sp,
        mode=mode,
        agent=agent,
        allocation_policy=allocation_policy,
        policy=policy,
        horizon=h,
        decision_interval=di,
        sample_interval=si,
        inventory_cap=cap,
        inv_gamma=g_inv,
        terminal_nu=nu,
        training=training,
        eval_sample=eval_sample,
        return_trajectory=return_trajectory,
        seed=config.seed,
    )
    ex.run()

    # Telescope diagnostic (paper Eq. 7): sum of shaped rewards + g must equal
    # the unshaped cash sum + terminal inventory value + liquidation cash.
    rsum = float(np.sum(ex.rewards)) if ex.rewards else 0.0
    rhs = (
        ex.cash_flow_total - ex.gamma_penalty_total + ex.q_plus * ex.final_mid + ex.mo_cash
    ) / ex.m_lots
    telescope_resid = rsum + ex.g_term - rhs
    cash_resid = ex.cash - (ex.cash_flow_total + ex.mo_cash)  # == 0 by construction
    flow_diag: dict[str, Any] | None = None
    if len(ex.signs) >= 52:
        flow_diag = regime_flow_diagnostics(ex.signs)
    rw = np.asarray(ex.rewards, dtype=np.float64) if ex.rewards else None
    mean_alloc = (
        np.asarray(ex.alloc_weight_hist, dtype=np.float64).mean(axis=0)
        if ex.alloc_weight_hist
        else None
    )
    final_mtm = ex.cash + ex.inventory * ex.final_mid
    bundle: dict[str, Any] = {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": MLMM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "policy_kind": mode,
        "training": bool(training),
        "seed": config.seed,
        "horizon": h,
        "decision_interval": di,
        "inventory_cap": cap,
        "inv_gamma": g_inv,
        "terminal_nu": nu,
        "n_levels": sp.n_levels if sp is not None else None,
        "lots": sp.lots if sp is not None else None,
        "session_completed": bool(ex.sim.t >= h),
        "n_events": ex.sim.n_events,
        "n_decisions": ex.n_decisions,
        "n_fills": ex.n_fills,
        "n_fills_bid": ex.n_fills_bid,
        "n_fills_ask": ex.n_fills_ask,
        "n_market_orders_injected": ex.n_mo_injected,
        "n_market_fills": ex.n_mo_fills,
        "n_mm_cancels": ex.n_cancels,
        "n_mm_posted": ex.n_posted,
        "n_mm_kept": ex.n_kept,
        "n_quote_clips": ex.n_clips,
        "n_skipped_decisions": ex.n_skipped,
        "n_inventory_gated": ex.n_gated,
        "n_self_trade_blocked": ex.n_self_blocked,
        "n_terminal_liquidation_lots": ex.terminal_lots,
        "n_terminal_liquidation_shortfall": ex.terminal_shortfall,
        "inventory_final": ex.inventory,
        "inventory_terminal_plus": ex.q_plus,
        "max_abs_inventory": ex.max_abs_inv,
        "mean_abs_inventory": float(np.mean(np.abs(np.asarray(ex.inv_path, dtype=np.float64))))
        if ex.inv_path
        else float("nan"),
        "inventory_path": ex.inv_path,
        "inventory_path_times": ex.inv_times,
        # Simulator-internal accounting. Diagnostic only: never a headline
        # metric, never market evidence, no live-trading claim.
        "sim_internal_mtm_pnl_path": ex.mtm_path,
        "sim_internal_mtm_pnl_final": float(final_mtm),
        "sim_internal_reward_path": ex.rewards,
        "sim_internal_reward_mean": float(rw.mean()) if rw is not None else None,
        "sim_internal_reward_sum": float(rsum),
        "sim_internal_terminal_g": float(ex.g_term),
        "sim_internal_telescope_residual": float(telescope_resid),
        "sim_internal_cash_consistency_residual": float(cash_resid),
        "terminal_liquidation_cash": float(ex.mo_cash),
        "mean_queue_ahead_at_fill": float(np.mean(ex.queue_ahead_fills))
        if ex.queue_ahead_fills
        else float("nan"),
        "mean_fill_wait_seconds": float(np.mean(ex.fill_waits)) if ex.fill_waits else float("nan"),
        "allocation_component_mean": mean_alloc.tolist() if mean_alloc is not None else None,
        "action_components": [
            {"kind": c.kind, "side": c.side, "level_offset": c.level_offset}
            for c in action_component_map(sp.n_levels)
        ]
        if sp is not None
        else None,
        "phase_metrics": book_phase_metrics(ex.samples),
        "flow_diagnostics": flow_diag,
        "n_signs": len(ex.signs),
        "event_counts": ex.sim.event_counts(),
        "n_updates_total": agent.n_updates if agent is not None else None,
    }
    if return_trajectory:
        bundle["trajectory"] = ex.trajectory
        bundle["trajectory_terminal_g"] = float(ex.g_term)
    return bundle


# ---------------------------------------------------------------------------
# Training loop + evaluation harness
# ---------------------------------------------------------------------------


def train_multilevel_mm(
    *,
    agent: MultiLevelMMAgent,
    config: ZILobConfig,
    horizon: float,
    n_episodes: int = 4,
    episodes_per_update: int = 1,
    seed_base: int = 0,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    inv_gamma: float = 0.01,
    terminal_nu: float = 1.0,
    eval_sample: bool = True,
    flow_factory: Callable[[int], MarkovRegimeFlow | None] | None = None,
) -> dict[str, Any]:
    """Train the agent over ``n_episodes`` simulator sessions.

    Each episode runs :func:`run_multilevel_mm_session` in training mode on a
    per-episode seed (``seed_base + i``), collecting the trajectory tuple of
    paper Eq. 13; after every ``episodes_per_update`` episodes one combined
    gradient step (Eq. 14) is taken — ``episodes_per_update`` is the paper's
    batch size ``tau`` split along the episode axis. Deterministic given the
    agent/config seeds. Rewards are the simulator-internal Eq. 5 signal
    (``sim_internal_*``), never headline metrics.
    """
    if not isinstance(agent, MultiLevelMMAgent):
        raise TypeError("agent must be a MultiLevelMMAgent")
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    n_ep = _int_at_least(n_episodes, 1, "n_episodes")
    epu = _int_at_least(episodes_per_update, 1, "episodes_per_update")
    _seed_int(seed_base)
    _pos_finite(horizon, "horizon")
    _nonneg_finite(inv_gamma, "inv_gamma")
    _prob(terminal_nu, "terminal_nu")
    if flow_factory is not None and not callable(flow_factory):
        raise TypeError("flow_factory must be callable or None")
    episodes: list[dict[str, Any]] = []
    pending: list[tuple[Sequence[TrajectoryStep], float]] = []
    losses: list[float] = []
    for i in range(n_ep):
        ep_seed = seed_base + i
        flow_i = flow_factory(ep_seed) if flow_factory is not None else None
        if flow_i is not None and not isinstance(flow_i, MarkovRegimeFlow):
            raise TypeError("flow_factory must return a MarkovRegimeFlow or None")
        bundle = run_multilevel_mm_session(
            config=replace(config, seed=ep_seed),
            horizon=horizon,
            agent=agent,
            training=True,
            decision_interval=decision_interval,
            sample_interval=sample_interval,
            flow=flow_i,
            inventory_cap=agent.spec.inventory_cap,
            inv_gamma=inv_gamma,
            terminal_nu=terminal_nu,
            eval_sample=eval_sample,
            return_trajectory=True,
        )
        pending.append((bundle["trajectory"], bundle["trajectory_terminal_g"]))
        if len(pending) >= epu:
            losses.append(agent.learn(pending))
            pending.clear()
        episodes.append(
            {
                "seed": bundle["seed"],
                "session_completed": bundle["session_completed"],
                "n_decisions": bundle["n_decisions"],
                "n_fills": bundle["n_fills"],
                "n_market_fills": bundle["n_market_fills"],
                "max_abs_inventory": bundle["max_abs_inventory"],
                "inventory_final": bundle["inventory_final"],
                "sim_internal_mtm_pnl_final": bundle["sim_internal_mtm_pnl_final"],
                "sim_internal_reward_mean": bundle["sim_internal_reward_mean"],
                "n_inventory_gated": bundle["n_inventory_gated"],
            }
        )
    if pending:
        losses.append(agent.learn(pending))
    loss_arr = np.asarray(losses, dtype=np.float64) if losses else None
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": MLMM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "multilevel_mm_training_run",
        "flow": "regime_factory" if flow_factory is not None else "stationary",
        "n_episodes": n_ep,
        "episodes_per_update": epu,
        "episodes": episodes,
        "loss_curve": losses,
        "loss_initial_mean": float(loss_arr[:10].mean()) if loss_arr is not None else None,
        "loss_final_mean": float(loss_arr[-10:].mean()) if loss_arr is not None else None,
        "n_updates": agent.n_updates,
        "training_budget": {
            "horizon": float(horizon),
            "n_episodes": n_ep,
            "decision_interval": float(decision_interval),
            "inv_gamma": float(inv_gamma),
            "terminal_nu": float(terminal_nu),
            "seed_base": int(seed_base),
        },
    }


_SESSION_ROW_KEYS = (
    "seed",
    "session_completed",
    "max_abs_inventory",
    "mean_abs_inventory",
    "inventory_final",
    "sim_internal_mtm_pnl_final",
    "n_fills",
    "n_market_fills",
    "n_decisions",
    "n_inventory_gated",
    "mean_spread_ticks",
    "phase",
)


def _session_row(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": bundle["seed"],
        "session_completed": bool(bundle["session_completed"]),
        "max_abs_inventory": int(bundle["max_abs_inventory"]),
        "mean_abs_inventory": float(bundle["mean_abs_inventory"]),
        "inventory_final": int(bundle["inventory_final"]),
        "sim_internal_mtm_pnl_final": float(bundle["sim_internal_mtm_pnl_final"]),
        "n_fills": int(bundle["n_fills"]),
        "n_market_fills": int(bundle["n_market_fills"]),
        "n_decisions": int(bundle["n_decisions"]),
        "n_inventory_gated": int(bundle["n_inventory_gated"]),
        "mean_spread_ticks": float(bundle["phase_metrics"]["mean_spread_ticks"]),
        "phase": str(bundle["phase_metrics"]["phase"]),
    }


def evaluate_multilevel_mm(
    *,
    config: ZILobConfig,
    horizon: float,
    agent: MultiLevelMMAgent | None = None,
    spec: MultiLevelSpec | None = None,
    n_seeds: int = 2,
    seed_base: int = 101,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    inv_gamma: float = 0.01,
    terminal_nu: float = 1.0,
    eval_sample: bool = True,
    glft_gamma: float = 1.0,
    glft_sigma: float = 0.02,
    glft_kappa: float = 1000.0,
    glft_a_fill: float = 1.0,
    as_gamma: float = 0.002,
    as_sigma: float = 0.02,
    as_kappa: float = 1000.0,
    inventory_cap: int | None = None,
    arms: Sequence[str] = ("agent", "glft", "random"),
) -> dict[str, Any]:
    """Paired-seed evaluation: the agent vs GLFT / AS / random allocation.

    Every arm runs through :func:`run_multilevel_mm_session` on identical
    simulator seeds (same ZI event tape), so cross-arm gaps isolate the
    policy. ``agent`` requires a trained :class:`MultiLevelMMAgent`; ``glft``
    and ``as`` are the ``zi_lob_simulator`` closed-form adapters (single-quote
    baselines); ``random`` is :class:`RandomAllocationPolicy` — a uniform
    simplex arm through the SAME multi-level execution path, the honest
    contrast for the allocation machinery. The agent samples actions from a
    per-session seeded stream (paper: evaluation also samples), so the arm is
    deterministic given the trained parameters.

    SYNTHETIC simulator-internal comparison: ``sim_internal_*`` keys are
    diagnostics, never headline metrics, never market evidence.
    """
    if not isinstance(config, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    arm_set = tuple(arms)
    if not arm_set or any(a not in ("agent", "glft", "as", "random") for a in arm_set):
        raise ValueError(f"arms must be a non-empty subset of agent/glft/as/random, got {arms!r}")
    if "agent" in arm_set and not isinstance(agent, MultiLevelMMAgent):
        raise TypeError("agent must be a MultiLevelMMAgent when the 'agent' arm is enabled")
    if "random" in arm_set and agent is None and spec is None:
        raise ValueError("spec (or agent.spec) is required for the 'random' arm")
    sp = agent.spec if agent is not None else spec
    if sp is not None and not isinstance(sp, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    ns = _int_at_least(n_seeds, 1, "n_seeds")
    _seed_int(seed_base)
    h = _pos_finite(horizon, "horizon")
    cap = inventory_cap
    if agent is not None:
        cap = agent.spec.inventory_cap
    sessions: dict[str, list[dict[str, Any]]] = {}
    tick = config.tick
    for i in range(ns):
        ep_seed = seed_base + i
        cfg_i = replace(config, seed=ep_seed)
        for arm in arm_set:
            kw: dict[str, Any] = {}
            if arm == "agent":
                kw = {"agent": agent}
            elif arm == "glft":
                kw = {
                    "policy": glft_policy(
                        gamma=glft_gamma,
                        sigma=glft_sigma,
                        kappa=glft_kappa,
                        a_fill=glft_a_fill,
                        tick=tick,
                    )
                }
            elif arm == "as":
                kw = {
                    "policy": as_policy(gamma=as_gamma, sigma=as_sigma, kappa=as_kappa, tick=tick)
                }
            else:
                if sp is None:  # pragma: no cover - guarded by arm validation
                    raise RuntimeError("random arm requires a spec")
                kw = {
                    "allocation_policy": RandomAllocationPolicy(sp, seed=seed_base + 555 + i),
                    "spec": sp,
                }
            bundle = run_multilevel_mm_session(
                config=cfg_i,
                horizon=h,
                training=False,
                decision_interval=decision_interval,
                sample_interval=sample_interval,
                inventory_cap=cap,
                inv_gamma=inv_gamma,
                terminal_nu=terminal_nu,
                eval_sample=eval_sample,
                **kw,
            )
            sessions.setdefault(arm, []).append(_session_row(bundle))
    metrics: dict[str, float] = {}
    for arm, rows in sessions.items():
        n = float(len(rows))
        finals = np.asarray([r["sim_internal_mtm_pnl_final"] for r in rows], dtype=np.float64)
        metrics[f"{arm}_session_completion_rate"] = float(
            sum(1.0 for r in rows if r["session_completed"]) / n
        )
        metrics[f"{arm}_n_fills_mean"] = float(np.mean([r["n_fills"] for r in rows]))
        metrics[f"{arm}_n_market_fills_mean"] = float(np.mean([r["n_market_fills"] for r in rows]))
        metrics[f"{arm}_max_abs_inventory_mean"] = float(
            np.mean([r["max_abs_inventory"] for r in rows])
        )
        metrics[f"{arm}_mean_abs_inventory_mean"] = float(
            np.mean([r["mean_abs_inventory"] for r in rows])
        )
        metrics[f"sim_internal_mtm_pnl_final_mean_{arm}"] = float(finals.mean())
        metrics[f"sim_internal_mtm_pnl_final_std_{arm}"] = float(finals.std())
    if "agent" in arm_set:
        for other in arm_set:
            if other == "agent":
                continue
            gap = (
                metrics["sim_internal_mtm_pnl_final_mean_agent"]
                - metrics[f"sim_internal_mtm_pnl_final_mean_{other}"]
            )
            metrics[f"sim_internal_mtm_pnl_gap_agent_minus_{other}_mean"] = float(gap)
            metrics[f"n_fills_gap_agent_minus_{other}_mean"] = float(
                metrics["agent_n_fills_mean"] - metrics[f"{other}_n_fills_mean"]
            )
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": MLMM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "multilevel_mm_vs_classic_evaluation",
        "note": (
            "simulator-internal synthetic policy comparison; never headline "
            "metrics, never market evidence, no live-trading claim"
        ),
        "arms": arm_set,
        "n_seeds": ns,
        "seed_base": int(seed_base),
        "horizon": h,
        "inventory_cap": cap,
        "decision_interval": float(decision_interval),
        "sessions": sessions,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Documented-optional full-budget benchmark (bench battery entry point)
# ---------------------------------------------------------------------------


def multilevel_mm_benchmark(
    *,
    config: ZILobConfig | None = None,
    spec: MultiLevelSpec | None = None,
    agent_config: MultiLevelMMConfig | None = None,
    horizon: float = 3000.0,
    train_episodes: int = 40,
    eval_seeds: int = 8,
    decision_interval: float = 1.0,
    sample_interval: float = 25.0,
    inv_gamma: float = 0.01,
    terminal_nu: float = 1.0,
    episodes_per_update: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Full-comparison benchmark: train the agent, then evaluate vs baselines.

    Documented-optional and long-running by design — NOT part of the PR-gate
    test suite. At paper-scale budgets (hidden 128x2, tau=1280 episodes per
    update, H=800 steps) this runs for hours; the defaults here are the
    reduced lane form. All outputs are SYNTHETIC simulator-internal
    diagnostics, never headline metrics, never market evidence.
    """
    _seed_int(seed)
    cfg = config if config is not None else santa_fe_config(seed=seed)
    if not isinstance(cfg, ZILobConfig):
        raise TypeError("config must be a ZILobConfig")
    sp = spec if spec is not None else MultiLevelSpec()
    if not isinstance(sp, MultiLevelSpec):
        raise TypeError("spec must be a MultiLevelSpec")
    acfg = agent_config if agent_config is not None else MultiLevelMMConfig(seed=seed)
    if not isinstance(acfg, MultiLevelMMConfig):
        raise TypeError("agent_config must be a MultiLevelMMConfig")
    agent = MultiLevelMMAgent(sp, acfg)
    training = train_multilevel_mm(
        agent=agent,
        config=cfg,
        horizon=horizon,
        n_episodes=train_episodes,
        episodes_per_update=episodes_per_update,
        seed_base=seed,
        decision_interval=decision_interval,
        sample_interval=sample_interval,
        inv_gamma=inv_gamma,
        terminal_nu=terminal_nu,
    )
    evaluation = evaluate_multilevel_mm(
        agent=agent,
        config=cfg,
        horizon=horizon,
        n_seeds=eval_seeds,
        seed_base=seed + 7_777_777,
        decision_interval=decision_interval,
        sample_interval=sample_interval,
        inv_gamma=inv_gamma,
        terminal_nu=terminal_nu,
    )
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "agent_revision": MLMM_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "multilevel_mm_benchmark",
        "training": training,
        "evaluation": evaluation,
        "metrics": evaluation["metrics"],
        "bench_keys": sorted(evaluation["metrics"]),
    }
