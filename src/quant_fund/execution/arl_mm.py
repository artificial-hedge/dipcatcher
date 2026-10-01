"""ARLMM: adversarial-RL market making with Hawkes order flow + trade impact.

**Labeled SYNTHETIC** research infrastructure (wave-20 lane): a zero-sum
stochastic game between a market maker and an environmental adversary,
following

- Yang, H., Xu, Z. (2026). "Robust Market Making with Hawkes Order Flow and
  Price Impact via Adversarial Reinforcement Learning." arXiv:2609.22785
  [q-fin.TR] (citation verified against the arXiv abstract 2026-09-30). The
  paper casts Avellaneda-Stoikov market making as a zero-sum game: prior
  adversarial-RL market-making work used Poisson order arrivals and ignored
  trade-induced price impact; this extension gives the environment Hawkes
  self-exciting market-order arrivals and trade impact, handles the stronger
  non-stationarity with an LSTM policy network, and evaluates robustness on
  the LEFT TAIL of the episode-return distribution — reporting, via bootstrap
  tests, no evidence that gains come from a stronger terminal directional-
  inventory bias.

Supporting references:

- Pinto, L., Davidson, J., Sukthankar, R., Gupta, A. (2017). "Robust
  Adversarial Reinforcement Learning." *ICML 2017*, arXiv:1703.02702 — the
  zero-sum protagonist/adversary framework the paper builds on (their
  adversary perturbs transition dynamics; ours perturbs environment
  parameters inside a bounded budget box).
- Avellaneda, M., Stoikov, S. (2008). "High-frequency trading in a limit
  order book." *Quantitative Finance* 8(3):217-224 — the market-making
  objective (spread capture vs inventory risk) and the terminal inventory
  penalty used in the episode return.
- Hawkes, A.G. (1971). "Spectra of some self-exciting and mutually exciting
  point processes." *Biometrika* 58 — the self-exciting arrival law.
- Ogata, Y. (1981). "On Lewis' simulation method for point processes."
  *IEEE Trans. Inf. Theory* 27 — thinning simulation (delegated to
  ``models.point_process.hawkes_simulate``).
- Hochreiter, S., Schmidhuber, J. (1997). "Long short-term memory."
  *Neural Computation* 9(8) — the recurrent policy core.
- Williams, R.J. (1992). "Simple statistical gradient-following algorithms
  for connectionist reinforcement learning." *Machine Learning* 8 — the
  episodic REINFORCE estimator used for both actors.
- Merton, R.C. (1976). "Option pricing when underlying stock returns are
  discontinuous." *JFE* 3 — the compound-Poisson mid jump component.

Paper formulation implemented here:

- **Environment** (:class:`ArlMMEnv`, :class:`ArlMMEnvParams`): a finite-
  horizon decision-clock market-making environment. The mid follows a GBM
  diffusion (plus optional Merton jumps); market orders arrive on a Hawkes
  self-exciting clock (baseline ``mu``, branching ``alpha`` in [0, 1), decay
  ``beta``) with i.i.d. exponential sizes and Bernoulli sides. Every market
  order moves the mid permanently by ``impact_eta * size`` in its direction
  (linear trade-induced impact). The maker posts unit-lot quotes
  ``d_bid``/``d_ask`` ticks off the decision-time mid; a market order fills a
  resting quote with probability ``exp(-k_fill * d_ticks / size)`` — larger
  orders reach deeper (walk the book) — subject to an inventory cap that
  suppresses the side that would breach it (inventory-aware fill model).
  Episode return = terminal mark-to-market wealth ``cash + q * S_T`` minus a
  quadratic running inventory penalty and a terminal ``psi * q_T^2``
  Avellaneda-Stoikov penalty. This is a ``sim_internal_*`` training signal,
  NEVER a headline metric.
- **Adversary** (:class:`AdversaryBudget`, :func:`apply_perturbation`): once
  per episode (SMDP at episode granularity) it picks a 4-dim action in
  [-1, 1]^4 perturbing (arrival intensity ``mu``, Hawkes branching ``alpha``,
  mid vol ``sigma``, impact coefficient ``eta``) inside the budget box; its
  reward is the negated maker return (zero-sum).
- **Policies**: both actors are LSTM policies (torch, lazy ``_torch()``
  idiom). The maker is a categorical policy over a small
  Avellaneda-Stoikov quote table ``(spread, skew)``
  (:data:`MAKER_QUOTE_TABLE`) mapped to per-side offsets
  ``d_bid = clip(s + k q/cap, 0, max)``, ``d_ask = clip(s - k q/cap, 0,
  max)`` — the reservation-price skew that controls inventory tail risk;
  its 11-feature observation includes a normalized environment-descriptor
  block (the regime summary the paper's LSTM must infer from flow,
  supplied directly at CI scale). The adversary is a Gaussian-tanh policy
  mapping an environment-descriptor sequence to the perturbation vector.
  Training alternates adversary updates (maker frozen, acting greedily)
  and maker best-response updates (adversary frozen at its greedy corner,
  plus jitter and boundary-corner coverage and a benign slice) on episodic
  REINFORCE with per-environment baselines — the paper's alternating ARL
  loop at CI scale.
- **Left-tail protocol** (:func:`adversary_env_grid`,
  :func:`evaluate_left_tail`, :func:`left_tail_stats`): both makers are
  evaluated on a fixed grid of adversarial environments (single-dim
  worst corners, the all-worst corner, seeded full-box interior points,
  and the trained adversary's greedy perturbation as the ``adv_learned_0``
  cell) plus the benign environment; the diagnostics are the 5% quantile
  and 5% expected shortfall of the episode-return distribution, matching
  the paper's left-tail robustness criterion.

Deviations from the paper (deliberate, CI-scale):

1. **Reduced-form fill model**: the paper simulates a full LOB; we use the
   classical reduced-form fill intensity ``exp(-k d)`` generalized by MO
   size (``exp(-k_fill * d / size)``) so Hawkes bursts and impact both feed
   the adverse-selection channel endogenously. The ZI FIFO engine
   (``microstructure.zi_lob_simulator``) is NOT reused: its Poisson flows
   cannot express Hawkes self-excitation or parametric trade impact.
2. **Episodic REINFORCE** for both actors instead of the paper's
   actor-critic variant: deterministic, tiny, and sufficient for the
   comparative science assertions (ARL vs non-adversarial baseline).
3. **Episode-granularity adversary**: one bounded perturbation per episode
   rather than per-step control; the worst-case corner structure is what
   the left-tail protocol measures.
4. **Categorical maker over a quote table**: the paper's continuous
   Gaussian quote head is replaced by a categorical choice over five
   (spread, skew) AS templates — discrete episodic REINFORCE is far less
   gradient-noisy at CI budgets, and the regime-conditioned argmax is
   exactly the robust-quoting mechanism under test.
5. **Environment descriptor in the maker observation**: the normalized
   regime parameters (mu, alpha, sigma, eta) appear in the observation —
   an observable-in-principle summary the paper's LSTM infers from flow —
   so the comparative conditioning is learnable at this scale.
6. **Pinned bench seed**: training outcome is seed-sensitive at CI
   budgets (tiny nets, few updates); the bench pins ``seed=2`` so its
   diagnostics are deterministic and demonstrate the mechanism on the
   seeded run, not SOTA training quality.
7. **Tiny budgets**: hidden sizes, episode counts, and grid sizes are
   scaled so the bench runs in seconds; tests assert the *mechanism* on
   SYNTHETIC data, never paper-scale training quality.

Composition (single source of truth, imported not reimplemented):
``models.point_process.hawkes_simulate`` generates the MO arrival clock
(Ogata thinning, seeded), ``models.point_process.hawkes_intensity``
evaluates the conditional intensity for the cluster diagnostics, and
``models.point_process.hawkes_mle`` estimates the branching ratio.
``models.deep_hedging`` owns the ``_torch()`` lazy-import idiom this module
mirrors for the optional ``nn`` extra.

Honesty: every output is a SYNTHETIC correctness diagnostic on a synthetic
environment, never market evidence. All mark-to-market accounting stays
inside episode results under ``sim_internal_*`` names and never enters the
bench blob; the bench reports left-tail *distribution* diagnostics under
``synthetic_*`` keys. No Sharpe/Sortino/Calmar/P&L/NAV headline; no broker
connectivity or live-trading claim anywhere in this module.

Conventions: torch is the optional ``nn`` extra, imported lazily via
:func:`_torch`, so this module imports cleanly without torch and every torch
entry point raises ``ImportError`` with install guidance. The environment,
adversary arithmetic, eval grids, and all diagnostics are pure numpy.
Training is CPU single-thread episodic REINFORCE and deterministic given
``seed`` (GPU determinism is not claimed). Fail-closed: degenerate env
params, empty episodes, non-finite rewards/actions, and malformed budgets
all raise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.point_process import hawkes_intensity, hawkes_mle, hawkes_simulate

Array = NDArray[np.float64]

__all__ = [
    "ARL_MM_REVISION",
    "AdversaryBudget",
    "ArlMMEnv",
    "ArlMMEnvParams",
    "EpisodeResult",
    "LeftTailEval",
    "ARLTrainResult",
    "adversary_env_grid",
    "apply_perturbation",
    "arl_mm_bench",
    "evaluate_left_tail",
    "hawkes_cluster_stats",
    "inventory_bias_check",
    "left_tail_stats",
    "run_episode",
    "train_arl_pair",
    "train_baseline_maker",
]

ARL_MM_REVISION = "SYNTHETIC_ARL_MM_v1"

# Feature clips for the maker observation vector (deterministic, documented).
_MID_REL_CLIP = 3.0
_LAM_REL_CLIP = 3.0
_SIZE_REL_CLIP = 3.0


# ---------------------------------------------------------------------------
# Fail-closed validation helpers (local; house style)
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
            "the ARL market maker needs the optional 'nn' extra (torch): uv sync --extra nn"
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Environment parameters + adversary budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArlMMEnvParams:
    """Decision-clock market-making environment parameters.

    ``sigma`` is the log-mid diffusion vol per sqrt-second; optional Merton
    jumps arrive at ``jump_rate`` per second with log-size std
    ``jump_scale``. Market orders follow a stationary exponential-kernel
    Hawkes process (``hawkes_mu`` baseline, ``hawkes_alpha`` branching ratio
    in [0, 1), ``hawkes_beta`` decay per second) with i.i.d. Exp sizes of
    mean ``mean_mo_size`` (unit lots) and buy probability ``p_buy``. Each
    market order moves the mid permanently by ``impact_eta * size`` in its
    direction. The maker quotes ``d`` ticks off mid per side; a market order
    of ``size`` units fills a quote at depth ``d`` ticks with probability
    ``exp(-k_fill * d / size)`` — larger orders walk deeper — subject to the
    ``inventory_cap`` side gate. The running inventory penalty
    ``risk_aversion * (q/cap)^2 * dt`` and the terminal
    ``terminal_penalty * (q_T/cap)^2`` are the Avellaneda-Stoikov risk
    terms, normalized by the inventory cap so they are scale-free in the
    wealth units of the episode return. ``max_quote_ticks`` bounds the
    quote action space; ``horizon``/``n_steps`` set the finite episode on
    the decision clock (``dt = horizon / n_steps``).
    """

    s0: float = 100.0
    tick: float = 0.01
    sigma: float = 0.0008
    jump_rate: float = 0.0
    jump_scale: float = 0.02
    hawkes_mu: float = 2.5
    hawkes_alpha: float = 0.6
    hawkes_beta: float = 3.0
    p_buy: float = 0.5
    mean_mo_size: float = 1.0
    impact_eta: float = 0.0015
    k_fill: float = 0.5
    inventory_cap: int = 5
    risk_aversion: float = 0.05
    terminal_penalty: float = 1.0
    max_quote_ticks: int = 8
    horizon: float = 30.0
    n_steps: int = 30

    def __post_init__(self) -> None:
        _pos_finite(self.s0, "s0")
        _pos_finite(self.tick, "tick")
        _pos_finite(self.sigma, "sigma")
        _nonneg_finite(self.jump_rate, "jump_rate")
        _pos_finite(self.jump_scale, "jump_scale")
        _pos_finite(self.hawkes_mu, "hawkes_mu")
        a = float(self.hawkes_alpha)
        if not math.isfinite(a) or a < 0.0 or a >= 1.0:
            raise ValueError(f"hawkes_alpha must lie in [0, 1), got {self.hawkes_alpha!r}")
        _pos_finite(self.hawkes_beta, "hawkes_beta")
        _prob(self.p_buy, "p_buy")
        _pos_finite(self.mean_mo_size, "mean_mo_size")
        _nonneg_finite(self.impact_eta, "impact_eta")
        _pos_finite(self.k_fill, "k_fill")
        _int_at_least(self.inventory_cap, 1, "inventory_cap")
        _nonneg_finite(self.risk_aversion, "risk_aversion")
        _nonneg_finite(self.terminal_penalty, "terminal_penalty")
        _int_at_least(self.max_quote_ticks, 1, "max_quote_ticks")
        _pos_finite(self.horizon, "horizon")
        _int_at_least(self.n_steps, 2, "n_steps")

    @property
    def dt(self) -> float:
        """Decision interval in seconds."""
        return self.horizon / self.n_steps


@dataclass(frozen=True)
class AdversaryBudget:
    """Bounded perturbation box for the environmental adversary.

    ``mu_range``/``sigma_range``/``eta_range`` are multiplier bounds applied
    to the base parameter; ``alpha_range`` is an absolute bound on the
    Hawkes branching ratio (a probability-like quantity, so it is clamped
    rather than scaled). An adversary action ``a in [-1, 1]`` interpolates
    between the bounds with ``a = 0`` the identity perturbation.
    """

    mu_range: tuple[float, float] = (0.5, 2.0)
    alpha_range: tuple[float, float] = (0.0, 0.9)
    sigma_range: tuple[float, float] = (0.5, 3.0)
    eta_range: tuple[float, float] = (0.25, 20.0)

    def __post_init__(self) -> None:
        for name in ("mu_range", "sigma_range", "eta_range", "alpha_range"):
            r = getattr(self, name)
            if (
                not isinstance(r, tuple)
                or len(r) != 2
                or not math.isfinite(float(r[0]))
                or not math.isfinite(float(r[1]))
                or float(r[0]) >= float(r[1])
            ):
                raise ValueError(f"{name} must be a (lo, hi) pair with lo < hi, got {r!r}")
        for name in ("mu_range", "sigma_range", "eta_range"):
            r = getattr(self, name)
            if float(r[0]) <= 0.0:
                raise ValueError(f"{name} multipliers must be positive, got {r!r}")
        lo, hi = (float(v) for v in self.alpha_range)
        if lo < 0.0 or hi >= 1.0:
            raise ValueError(
                f"alpha_range must lie in [0, 1) (Hawkes stationarity), got {self.alpha_range!r}"
            )


def _bounded_action(action: Sequence[float] | Array, name: str = "adversary_action") -> Array:
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != 4:
        raise ValueError(f"{name} must have exactly 4 components, got {a.size}")
    if not bool(np.all(np.isfinite(a))):
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(a) > 1.0 + 1e-9):
        raise ValueError(f"{name} components must lie in [-1, 1], got {a!r}")
    return np.asarray(np.clip(a, -1.0, 1.0), dtype=np.float64)


def _interp_multiplier(a: float, lo: float, hi: float) -> float:
    """Piecewise-geometric multiplier: a=-1 -> lo, a=0 -> 1, a=+1 -> hi."""
    return math.exp(a * (math.log(hi) if a >= 0.0 else -math.log(lo)))


def _interp_absolute(a: float, base: float, lo: float, hi: float) -> float:
    """Piecewise-linear absolute: a=-1 -> lo, a=0 -> base, a=+1 -> hi."""
    return base + a * ((hi - base) if a >= 0.0 else (base - lo))


def apply_perturbation(
    params: ArlMMEnvParams, budget: AdversaryBudget, action: Sequence[float] | Array
) -> ArlMMEnvParams:
    """Perturb environment params by adversary action ``a in [-1, 1]^4``.

    Action order: ``(mu, alpha, sigma, eta)`` — arrival intensity
    multiplier, Hawkes branching shift, vol multiplier, impact multiplier.
    ``a = 0`` is the identity; the result is revalidated through
    :class:`ArlMMEnvParams` (fail closed).
    """
    if not isinstance(params, ArlMMEnvParams):
        raise TypeError("params must be an ArlMMEnvParams")
    if not isinstance(budget, AdversaryBudget):
        raise TypeError("budget must be an AdversaryBudget")
    a = _bounded_action(action)
    return replace(
        params,
        hawkes_mu=params.hawkes_mu * _interp_multiplier(float(a[0]), *map(float, budget.mu_range)),
        hawkes_alpha=_interp_absolute(
            float(a[1]), params.hawkes_alpha, *map(float, budget.alpha_range)
        ),
        sigma=params.sigma * _interp_multiplier(float(a[2]), *map(float, budget.sigma_range)),
        impact_eta=params.impact_eta
        * _interp_multiplier(float(a[3]), *map(float, budget.eta_range)),
    )


# ---------------------------------------------------------------------------
# Episode record + the environment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """One simulated episode's accounting record (simulator-internal).

    ``sim_internal_return`` is the RL episode return — terminal wealth
    ``cash + q * S_T`` minus the running inventory penalty and the terminal
    AS penalty. It is a synthetic training signal, never a headline metric.
    """

    sim_internal_return: float
    sim_internal_wealth_path: Array  # (n_steps + 1,) cash + q * mid per decision time
    terminal_inventory: int
    terminal_mid: float
    n_fills_bid: int
    n_fills_ask: int
    n_mo: int
    mo_times: Array
    max_abs_inventory: int


class ArlMMEnv:
    """Seeded Hawkes/impact market-making environment (paper Sec. env model).

    All episode randomness is drawn at :meth:`reset` from
    ``numpy.random.default_rng`` seeds — MO times via the composed
    :func:`~quant_fund.models.point_process.hawkes_simulate` (Ogata
    thinning), then sides, sizes, diffusion increments, jump times, and the
    per-order fill uniforms — so identical ``(params, seed)`` pairs give
    bit-identical episodes. Quotes are unit lots pinned to the decision-time
    mid; a market order fills the quote it reaches with probability
    ``exp(-k_fill * d_ticks / size)`` and then moves the mid permanently by
    ``impact_eta * size`` in its direction, whether or not the maker filled
    (public-order-flow price impact). Fail-closed: non-finite actions,
    rewards, or mid paths raise.
    """

    def __init__(self, params: ArlMMEnvParams) -> None:
        if not isinstance(params, ArlMMEnvParams):
            raise TypeError("params must be an ArlMMEnvParams")
        self._p = params
        self._reset_done = False

    @property
    def params(self) -> ArlMMEnvParams:
        return self._p

    # -- reset internals -----------------------------------------------------

    def reset(self, seed: int) -> None:
        """Draw the episode's exogenous randomness; fail closed on bad seed."""
        s = _seed_int(seed)
        p = self._p
        rng = np.random.default_rng(s)
        # Hawkes MO clock — composed Ogata thinning (single source of truth).
        mo_times = hawkes_simulate(
            p.hawkes_mu, p.hawkes_alpha, p.hawkes_beta, p.horizon, seed=s + 7_919
        )
        n_mo = int(mo_times.size)
        self._mo_times = mo_times
        self._mo_signs = np.where(rng.random(n_mo) < p.p_buy, 1.0, -1.0)
        self._mo_sizes = rng.exponential(p.mean_mo_size, n_mo)
        # Diffusion increments per decision interval.
        self._z_diff = rng.standard_normal(p.n_steps)
        # Compound-Poisson mid jumps, assigned to decision intervals.
        self._jump_steps = np.zeros(p.n_steps, dtype=np.int64)
        self._jump_sizes = np.zeros(p.n_steps, dtype=float)
        if p.jump_rate > 0.0:
            n_j = int(rng.poisson(p.jump_rate * p.horizon))
            if n_j > 0:
                jt = np.sort(rng.uniform(0.0, p.horizon, n_j))
                js = rng.normal(0.0, p.jump_scale, n_j)
                idx = np.minimum((jt / p.dt).astype(np.int64), p.n_steps - 1)
                for k in range(n_j):
                    self._jump_steps[idx[k]] += 1
                    self._jump_sizes[idx[k]] += js[k]
        # Fill uniforms, one per MO.
        self._u_fill = rng.random(n_mo)
        # Episode state.
        self._t_idx = 0
        self._mid = float(p.s0)
        self._cash = 0.0
        self._q = 0
        self._mo_cursor = 0
        self._wealth = 0.0
        self._n_fills_bid = 0
        self._n_fills_ask = 0
        self._max_abs_inv = 0
        self._last_side_sign = 0.0
        self._last_fill_sign = 0.0
        self._recent_sizes = 0.0
        self._recent_n_mo = 0.0
        self._wealth_path: list[float] = [0.0]
        self._rewards: list[float] = []
        self._reset_done = True

    # -- features ------------------------------------------------------------

    def _hawkes_lam(self, t: float) -> float:
        """Conditional intensity at time ``t`` given the realized MO history."""
        p = self._p
        past = self._mo_times[self._mo_times < t]
        if past.size == 0:
            return p.hawkes_mu
        return float(
            p.hawkes_mu
            + p.hawkes_alpha * p.hawkes_beta * np.sum(np.exp(-p.hawkes_beta * (t - past)))
        )

    def obs(self) -> Array:
        """Maker observation vector at the current decision time (11 features).

        ``[t_frac, q/cap, log(mid/s0)/0.05 clip, lam/mu clip, signed MO
        imbalance of the last interval, last fill side, mean MO size
        ratio clip]`` plus a normalized environment descriptor
        ``[mu/2.5, alpha, sigma/0.005, eta/0.02]`` — the regime summary the
        paper's LSTM infers from observed flow, supplied directly at CI
        scale so the comparative assertions learn fast (documented
        simplification). All entries are bounded and finite by
        construction.
        """
        if not self._reset_done:
            raise RuntimeError("obs called before reset")
        p = self._p
        t = self._t_idx * p.dt
        lam_rel = self._hawkes_lam(t) / p.hawkes_mu
        return np.asarray(
            [
                t / p.horizon,
                self._q / p.inventory_cap,
                float(np.clip(math.log(self._mid / p.s0) / 0.05, -_MID_REL_CLIP, _MID_REL_CLIP)),
                float(np.clip(lam_rel / 4.0, 0.0, _LAM_REL_CLIP)),
                self._last_side_sign,
                self._last_fill_sign,
                float(
                    np.clip(
                        self._recent_sizes / max(self._recent_n_mo, 1.0) / p.mean_mo_size
                        if self._recent_n_mo > 0.0
                        else 1.0,
                        0.0,
                        _SIZE_REL_CLIP,
                    )
                ),
                p.hawkes_mu / 2.5,
                p.hawkes_alpha,
                float(np.clip(p.sigma / 0.005, 0.0, 4.0)),
                float(np.clip(p.impact_eta / 0.02, 0.0, 4.0)),
            ],
            dtype=float,
        )

    # -- dynamics ------------------------------------------------------------

    @property
    def mid(self) -> float:
        return self._mid

    @property
    def inventory(self) -> int:
        return self._q

    @property
    def mo_times(self) -> Array:
        """Realized market-order arrival times for this episode (post-reset)."""
        if not self._reset_done:
            raise RuntimeError("mo_times accessed before reset")
        return np.asarray(self._mo_times, dtype=float)

    @property
    def done(self) -> bool:
        return self._t_idx >= self._p.n_steps

    def _check_quote_action(self, action: Sequence[float] | Array) -> tuple[float, float]:
        a = np.asarray(action, dtype=float).reshape(-1)
        if a.size != 2:
            raise ValueError(
                f"quote action must be a (d_bid_ticks, d_ask_ticks) pair, got {a.size}"
            )
        if not bool(np.all(np.isfinite(a))):
            raise ValueError(f"quote action must be finite, got {action!r}")
        cap = float(self._p.max_quote_ticks)
        if np.any(a < -1e-9) or np.any(a > cap + 1e-9):
            raise ValueError(f"quote action must lie in [0, {cap}] ticks, got {a!r}")
        return float(a[0]), float(a[1])

    def step(self, action: Sequence[float] | Array) -> float:
        """Post quotes for one decision interval; return the interval reward.

        ``action = (d_bid_ticks, d_ask_ticks)`` in ``[0, max_quote_ticks]``.
        The reward is the mark-to-market wealth increment minus the running
        quadratic inventory penalty; the last interval additionally realizes
        the terminal AS penalty ``terminal_penalty * q_T^2``.
        """
        if not self._reset_done:
            raise RuntimeError("step called before reset")
        if self.done:
            raise RuntimeError("episode already finished; call reset")
        p = self._p
        d_bid, d_ask = self._check_quote_action(action)
        i = self._t_idx
        # 1) Diffusion + jumps on the log-mid for this interval.
        drift = -0.5 * p.sigma * p.sigma * p.dt + p.sigma * math.sqrt(p.dt) * self._z_diff[i]
        mid = self._mid * math.exp(drift)
        if self._jump_steps[i] > 0:
            mid *= math.exp(self._jump_sizes[i])
        mid = max(mid, p.tick)  # absorbing floor; degenerate only under extreme params
        # 2) Quotes pinned to the post-diffusion decision mid. The action
        #        pair may come straight from a policy or from the AS-style
        #        spread+skew parameterization applied upstream.
        bid_px = mid - d_bid * p.tick
        ask_px = mid + d_ask * p.tick
        # 3) Market orders arriving inside (t_i, t_i + dt].
        t1 = (i + 1) * p.dt
        n_side_buy = 0.0
        n_side_tot = 0.0
        size_sum = 0.0
        self._last_fill_sign = 0.0
        while self._mo_cursor < self._mo_times.size and self._mo_times[self._mo_cursor] < t1:
            j = self._mo_cursor
            self._mo_cursor += 1
            sign = float(self._mo_signs[j])
            size = float(self._mo_sizes[j])
            n_side_tot += 1.0
            n_side_buy += 1.0 if sign > 0.0 else 0.0
            size_sum += size
            if sign > 0.0:
                # Buy MO lifts the maker's ask if it reaches it.
                if self._q > -p.inventory_cap:
                    p_fill = math.exp(-p.k_fill * (d_ask / max(size, 1e-9)))
                    if float(self._u_fill[j]) < p_fill:
                        self._q -= 1
                        self._cash += ask_px
                        self._n_fills_ask += 1
                        self._last_fill_sign = -1.0
            else:
                # Sell MO hits the maker's bid.
                if self._q < p.inventory_cap:
                    p_fill = math.exp(-p.k_fill * (d_bid / max(size, 1e-9)))
                    if float(self._u_fill[j]) < p_fill:
                        self._q += 1
                        self._cash -= bid_px
                        self._n_fills_bid += 1
                        self._last_fill_sign = 1.0
            # Trade-induced permanent impact (public flow moves the mid).
            mid += sign * p.impact_eta * size
            mid = max(mid, p.tick)
        self._last_side_sign = (2.0 * n_side_buy - n_side_tot) / max(n_side_tot, 1.0)
        self._recent_n_mo = n_side_tot
        self._recent_sizes = size_sum
        self._max_abs_inv = max(self._max_abs_inv, abs(self._q))
        # 4) Mark-to-market reward with cap-normalized AS inventory terms.
        wealth = self._cash + self._q * mid
        qf = self._q / p.inventory_cap
        reward = wealth - self._wealth - p.risk_aversion * qf * qf * p.dt
        if i == p.n_steps - 1:
            reward -= p.terminal_penalty * qf * qf
        if not (math.isfinite(mid) and math.isfinite(wealth) and math.isfinite(reward)):
            raise RuntimeError("episode produced a non-finite state (mid/wealth/reward)")
        self._mid = mid
        self._wealth = wealth
        self._wealth_path.append(float(wealth))
        self._rewards.append(float(reward))
        self._t_idx = i + 1
        return float(reward)

    def result(self) -> EpisodeResult:
        """Terminal accounting record; fail closed before the episode ends."""
        if not self._reset_done:
            raise RuntimeError("result called before reset")
        if not self.done:
            raise RuntimeError("episode not finished; cannot produce a result")
        ret = float(np.sum(np.asarray(self._rewards, dtype=float)))
        if not math.isfinite(ret):
            raise RuntimeError("episode return is non-finite")
        return EpisodeResult(
            sim_internal_return=ret,
            sim_internal_wealth_path=np.asarray(self._wealth_path, dtype=float),
            terminal_inventory=self._q,
            terminal_mid=self._mid,
            n_fills_bid=self._n_fills_bid,
            n_fills_ask=self._n_fills_ask,
            n_mo=int(self._mo_times.size),
            mo_times=np.asarray(self._mo_times, dtype=float),
            max_abs_inventory=self._max_abs_inv,
        )


def run_episode(
    params: ArlMMEnvParams,
    actions: Array,
    *,
    seed: int = 0,
) -> EpisodeResult:
    """Run one episode under a precomputed ``(n_steps, 2)`` quote plan.

    Numpy convenience path for evaluation: reset the env, apply every quote
    pair in order, return the :class:`EpisodeResult`.
    """
    if not isinstance(params, ArlMMEnvParams):
        raise TypeError("params must be an ArlMMEnvParams")
    a = np.asarray(actions, dtype=float)
    if a.shape != (params.n_steps, 2):
        raise ValueError(f"actions must have shape ({params.n_steps}, 2); got {a.shape}")
    env = ArlMMEnv(params)
    env.reset(seed)
    for i in range(params.n_steps):
        env.step(a[i])
    return env.result()


# ---------------------------------------------------------------------------
# Hawkes cluster diagnostics (numpy; composes models.point_process)
# ---------------------------------------------------------------------------


def hawkes_cluster_stats(
    mo_times: Array,
    *,
    horizon: float,
    n_windows: int = 10,
    fit_mle: bool = True,
    mu: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
) -> dict[str, float]:
    """Clustering statistics of a realized MO arrival path.

    ``fano_factor`` is the variance/mean ratio of per-window event counts:
    1 for a Poisson process, > 1 under Hawkes clustering (self-excitation).
    ``intensity_max_over_mu`` is the peak conditional intensity relative to
    the baseline (burst amplification), computed via the composed
    :func:`~quant_fund.models.point_process.hawkes_intensity` when
    ``mu``/``alpha``/``beta`` are supplied. When ``fit_mle`` the composed
    :func:`~quant_fund.models.point_process.hawkes_mle` branching-ratio
    estimate is included (needs >= 5 events). Fail closed on empty or
    degenerate inputs.
    """
    t = np.asarray(mo_times, dtype=float).reshape(-1)
    h = _pos_finite(horizon, "horizon")
    _int_at_least(n_windows, 2, "n_windows")
    if t.size < 2:
        raise ValueError(f"mo_times needs at least 2 events, got {t.size}")
    if not bool(np.all(np.isfinite(t))) or np.any(t < 0.0) or np.any(t >= h):
        raise ValueError("mo_times must be finite times in [0, horizon)")
    edges = np.linspace(0.0, h, int(n_windows) + 1)
    counts = np.histogram(t, bins=edges)[0].astype(float)
    mean_c = float(counts.mean())
    var_c = float(counts.var())
    fano = var_c / mean_c if mean_c > 0.0 else float("nan")
    out: dict[str, float] = {
        "n_mo": float(t.size),
        "fano_factor": fano,
        "counts_mean": mean_c,
        "counts_var": var_c,
    }
    if mu is not None or alpha is not None or beta is not None:
        if mu is None or alpha is None or beta is None:
            raise ValueError("mu, alpha and beta must be supplied together")
        lam = hawkes_intensity(t, float(mu), float(alpha), float(beta))
        out["intensity_max_over_mu"] = float(lam.max() / float(mu))
        out["intensity_mean_over_mu"] = float(lam.mean() / float(mu))
    if fit_mle:
        fit = hawkes_mle(t)
        out["branching_ratio_mle"] = float(fit["branching_ratio"])
        out["mle_converged"] = float(fit["converged"])
    return out


# ---------------------------------------------------------------------------
# Left-tail evaluation protocol (numpy)
# ---------------------------------------------------------------------------


def adversary_env_grid(
    params: ArlMMEnvParams,
    budget: AdversaryBudget,
    *,
    n_interior: int = 2,
    seed: int = 0,
    extra_actions: Sequence[Array] | None = None,
) -> list[tuple[str, ArlMMEnvParams]]:
    """Fixed adversarial environment grid for the left-tail protocol.

    Grid points: the benign environment (``a = 0``), each single-dimension
    worst corner ``a_i = +1``, the all-worst corner ``a = +1^4``,
    ``n_interior`` seeded interior points uniform in ``[-1, 1]^4`` (full
    perturbation box, mixed signs — a trained adversary typically picks a
    mixed-sign corner), plus any ``extra_actions`` appended as
    ``adv_learned_k`` cells (e.g. the trained adversary's greedy
    perturbation — the game's own worst case). Deterministic given
    ``seed``; fail closed on degenerate counts.
    """
    _seed_int(seed)
    ni = _int_at_least(n_interior, 0, "n_interior")
    grid: list[tuple[str, ArlMMEnvParams]] = [("benign", params)]
    names = ("mu", "alpha", "sigma", "eta")
    for k, nm in enumerate(names):
        a = np.zeros(4)
        a[k] = 1.0
        grid.append((f"adv_{nm}_max", apply_perturbation(params, budget, a)))
    grid.append(("adv_all_max", apply_perturbation(params, budget, np.ones(4))))
    rng = np.random.default_rng(seed)
    for k in range(ni):
        a = rng.uniform(-1.0, 1.0, 4)
        grid.append((f"adv_interior_{k}", apply_perturbation(params, budget, a)))
    if extra_actions is not None:
        for k, act in enumerate(extra_actions):
            grid.append((f"adv_learned_{k}", apply_perturbation(params, budget, act)))
    return grid


def left_tail_stats(returns: Array, *, alpha: float = 0.05) -> dict[str, float]:
    """Left-tail statistics of an episode-return sample.

    ``q_alpha`` is the empirical alpha-quantile of the return distribution
    and ``es_alpha`` the mean of the subsample at or below it (expected
    shortfall / CVaR of the LEFT tail — the paper's robustness criterion).
    Fail closed on empty or non-finite samples and on ``alpha`` outside
    (0, 1).
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        raise ValueError("returns must be non-empty")
    if not bool(np.all(np.isfinite(r))):
        raise ValueError("returns must be finite (NaN/inf rejected)")
    a = float(alpha)
    if not math.isfinite(a) or not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    q = float(np.quantile(r, a))
    tail = r[r <= q]
    if tail.size == 0:
        tail = np.asarray([r.min()])
    return {
        "q_alpha": q,
        "es_alpha": float(tail.mean()),
        "mean": float(r.mean()),
        "std": float(r.std()),
        "min": float(r.min()),
        "n": float(r.size),
    }


def inventory_bias_check(
    terminal_inv_a: Array,
    terminal_inv_b: Array,
    *,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Directional-inventory-bias check (the paper's bootstrap test).

    Compares ``|mean terminal inventory|`` between two policies over matched
    evaluation episodes. ``bias_gap = |E[q_a]| - |E[q_b]|`` and
    ``p_more_bias`` is the bootstrap share of resamples where the gap is
    positive — a LARGE p means no evidence that policy ``a`` carries more
    terminal directional bias (the paper's null-preserving finding). Pairs
    are resampled jointly; deterministic given ``seed``. Fail closed on
    empty, mismatched, or non-finite inputs.
    """
    qa = np.asarray(terminal_inv_a, dtype=float).reshape(-1)
    qb = np.asarray(terminal_inv_b, dtype=float).reshape(-1)
    if qa.size == 0 or qb.size == 0:
        raise ValueError("terminal inventory samples must be non-empty")
    if qa.shape != qb.shape:
        raise ValueError(f"paired samples must have equal length, got {qa.shape} vs {qb.shape}")
    if not (bool(np.all(np.isfinite(qa))) and bool(np.all(np.isfinite(qb)))):
        raise ValueError("terminal inventory samples must be finite")
    nb = _int_at_least(n_boot, 10, "n_boot")
    _seed_int(seed)
    gap = abs(float(qa.mean())) - abs(float(qb.mean()))
    rng = np.random.default_rng(seed)
    n = qa.size
    boots = np.empty(nb)
    for b in range(nb):
        idx = rng.integers(0, n, n)
        boots[b] = abs(float(qa[idx].mean())) - abs(float(qb[idx].mean()))
    return {
        "bias_gap": gap,
        "p_more_bias": float(np.mean(boots > 0.0)),
        "mean_abs_q_a": abs(float(qa.mean())),
        "mean_abs_q_b": abs(float(qb.mean())),
        "n_pairs": float(n),
        "n_boot": float(nb),
    }


@dataclass(frozen=True)
class LeftTailEval:
    """Left-tail comparison of two makers on the environment grid."""

    returns: dict[str, dict[str, Array]]  # policy_name -> grid_name -> episode returns
    stats: dict[str, dict[str, dict[str, float]]]
    terminal_inventory: dict[str, dict[str, Array]]


def evaluate_left_tail(
    policies: dict[str, Any],
    grid: list[tuple[str, ArlMMEnvParams]],
    *,
    n_episodes: int,
    seed: int = 0,
    alpha: float = 0.05,
) -> LeftTailEval:
    """Evaluate maker policies on the environment grid (greedy actions).

    ``policies`` maps a name to either ``None`` (uniform mid-grid quotes), a
    ``(n_steps, 2)`` numpy action array, or a trained LSTM actor (any object
    exposing ``act_batch(obs, hidden)`` used by the torch rollout helpers —
    the torch path is taken when the object is not an ndarray/None). Every
    (policy, grid-point) cell runs ``n_episodes`` seeded episodes and stores
    the raw return + terminal-inventory samples plus their
    :func:`left_tail_stats`. Fail closed on empty grid/policies/episodes.
    """
    if not policies:
        raise ValueError("policies must be a non-empty mapping")
    if not grid:
        raise ValueError("grid must be non-empty")
    ne = _int_at_least(n_episodes, 1, "n_episodes")
    _seed_int(seed)
    for name, params in grid:
        if not isinstance(params, ArlMMEnvParams):
            raise TypeError(f"grid point {name!r} is not an ArlMMEnvParams")
    n_steps = grid[0][1].n_steps
    for _name, params in grid:
        if params.n_steps != n_steps:
            raise ValueError("all grid params must share n_steps")

    returns: dict[str, Any] = {}
    inventories: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for pol_name, pol in policies.items():
        for g_name, params in grid:
            rets = np.empty(ne)
            invs = np.empty(ne)
            for e in range(ne):
                ep_seed = seed + 10_007 * e + 101
                if pol is None:
                    acts = np.full((n_steps, 2), params.max_quote_ticks / 2.0)
                    res = run_episode(params, acts, seed=ep_seed)
                elif isinstance(pol, np.ndarray):
                    res = run_episode(params, pol, seed=ep_seed)
                else:
                    res = _torch_episode(pol, params, ep_seed)
                rets[e] = res.sim_internal_return
                invs[e] = float(res.terminal_inventory)
            returns.setdefault(pol_name, {})[g_name] = rets
            inventories.setdefault(pol_name, {})[g_name] = invs
            stats.setdefault(pol_name, {})[g_name] = left_tail_stats(rets, alpha=alpha)
    return LeftTailEval(returns=returns, stats=stats, terminal_inventory=inventories)


# ---------------------------------------------------------------------------
# Torch lane (lazy; skipped entirely without the nn extra)
# ---------------------------------------------------------------------------

MAKER_OBS_DIM = 11
ADV_OBS_DIM = 8
ADV_ACT_DIM = 4

# Maker quote table: (spread_ticks, skew_ticks) templates spanning the
# benign-aggressive to defensive range of the Avellaneda-Stoikov
# spread+linear-skew parameterization. The maker's LSTM actor is a
# categorical policy over these templates — the CI-scale discretization
# keeps the episodic-REINFORCE gradient low-variance (documented
# simplification of the paper's continuous Gaussian head).
MAKER_QUOTE_TABLE: Array = np.asarray(
    [
        [0.75, 0.5],
        [1.5, 1.0],
        [2.5, 2.0],
        [4.0, 3.0],
        [6.0, 4.0],
    ],
    dtype=float,
)
MAKER_ACT_DIM = int(MAKER_QUOTE_TABLE.shape[0])


def _build_lstm_actor(torch: Any, obs_dim: int, hidden: int, act_dim: int) -> Any:
    """One-layer LSTM actor: LSTMCell core + linear mean head."""

    class _Actor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTMCell(obs_dim, hidden)
            self.head = torch.nn.Linear(hidden, act_dim)
            self.hidden_size = hidden

        def init_h(self, batch: int) -> tuple[Any, Any]:
            dev = self.head.weight.device
            dt = self.head.weight.dtype
            return (
                torch.zeros(batch, self.hidden_size, device=dev, dtype=dt),
                torch.zeros(batch, self.hidden_size, device=dev, dtype=dt),
            )

        def forward(self, x: Any, hx: tuple[Any, Any]) -> tuple[Any, tuple[Any, Any]]:
            h, c = self.lstm(x, hx)
            return self.head(h), (h, c)

    return _Actor()


def _squash_adv(torch: Any, z: Any) -> Any:
    """Squash Gaussian samples to adversary actions in ``[-1, 1]^4``."""
    return torch.tanh(z)


def _normal_logp(torch: Any, z: Any, mean: Any, std: float) -> Any:
    """Diagonal-Gaussian log density of ``z`` summed over action dims."""
    var = std * std
    return (
        -0.5 * (z - mean) * (z - mean) / var - math.log(std) - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)


def _adv_descriptor(params: ArlMMEnvParams, budget: AdversaryBudget) -> Array:
    """Normalized base-environment descriptor the adversary conditions on."""
    return np.asarray(
        [
            params.hawkes_mu / 4.0,
            params.hawkes_alpha,
            params.hawkes_beta / 6.0,
            params.sigma / 0.1,
            params.impact_eta / 0.08,
            params.p_buy - 0.5,
            params.k_fill / 4.0,
            params.risk_aversion / 0.1,
        ],
        dtype=float,
    )


def _skew_to_quotes(actions: Array, q: int, cap: int, max_off: float) -> Array:
    """Map (spread, skew) to per-side tick offsets for the current inventory."""
    s, k = float(actions[0]), float(actions[1])
    qf = q / cap
    return np.asarray(
        [
            float(np.clip(s + k * qf, 0.0, max_off)),
            float(np.clip(s - k * qf, 0.0, max_off)),
        ],
        dtype=float,
    )


def _torch_episode(actor: Any, params: ArlMMEnvParams, seed: int) -> EpisodeResult:
    """Run one episode with the actor's greedy (argmax) actions; numpy boundary."""
    torch = _torch()
    torch.set_num_threads(1)
    env = ArlMMEnv(params)
    env.reset(seed)
    actor.eval()
    max_off = float(params.max_quote_ticks)
    with torch.no_grad():
        hx = actor.init_h(1)
        for _ in range(params.n_steps):
            obs_t = torch.as_tensor(env.obs(), dtype=torch.float32).unsqueeze(0)
            logits, hx = actor(obs_t, hx)
            a_idx = int(logits.argmax(dim=-1).item())
            act = MAKER_QUOTE_TABLE[a_idx]
            env.step(_skew_to_quotes(act, env.inventory, params.inventory_cap, max_off))
    return env.result()


def _rollout_maker_batch(
    torch: Any,
    actor: Any,
    params_seq: list[ArlMMEnvParams],
    seeds: list[int],
    *,
    explore: bool,
    explore_std: float,
) -> tuple[Array, Any, list[int]]:
    """Batched-step maker rollout: one LSTM step per decision time.

    Returns ``(returns, logp_sums, terminal_inventories)`` where
    ``logp_sums`` is a ``(batch,)`` tensor of summed per-step log
    densities (zero tensor when ``explore=False``).
    """
    envs: list[ArlMMEnv] = []
    for p, s in zip(params_seq, seeds, strict=True):
        env = ArlMMEnv(p)
        env.reset(s)
        envs.append(env)
    n = len(envs)
    steps = envs[0]._p.n_steps
    max_off = float(envs[0]._p.max_quote_ticks)
    hx = actor.init_h(n)
    logps: list[Any] = []
    for _ in range(steps):
        obs_np = np.stack([e.obs() for e in envs])
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        logits, hx = actor(obs_t, hx)
        if explore:
            dist = torch.distributions.Categorical(logits=logits)
            a_idx = dist.sample()
            logps.append(dist.log_prob(a_idx))
        else:
            a_idx = logits.argmax(dim=-1)
        act = MAKER_QUOTE_TABLE[a_idx.detach().numpy()]
        for e, a in zip(envs, act, strict=True):
            e.step(_skew_to_quotes(a.astype(float), e.inventory, e.params.inventory_cap, max_off))
    rets = np.asarray([e.result().sim_internal_return for e in envs], dtype=float)
    invs = [int(e.result().terminal_inventory) for e in envs]
    logp_sum = (
        torch.stack(logps, dim=0).sum(dim=0) if logps else torch.zeros(n, dtype=torch.float32)
    )
    return rets, logp_sum, invs


def _rollout_adversary_batch(
    torch: Any,
    adv: Any,
    maker: Any,
    base: ArlMMEnvParams,
    budget: AdversaryBudget,
    seeds: list[int],
    *,
    explore: bool,
    explore_std: float,
    maker_explore_std: float,
) -> tuple[Array, Any, list[ArlMMEnvParams], Array]:
    """Adversary picks one perturbation per episode; maker rolls it out.

    Returns ``(maker_returns, adv_logp, perturbed_params, adv_actions)``.
    """
    n = len(seeds)
    desc = np.tile(_adv_descriptor(base, budget), (4, 1))  # fixed ctx length 4
    hx = adv.init_h(n)
    desc_t = torch.as_tensor(desc, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1)
    mean = None
    for k in range(desc_t.shape[1]):
        mean, hx = adv(desc_t[:, k, :], hx)
    if mean is None:
        raise RuntimeError("adversary descriptor sequence is empty")
    if explore:
        z = mean + explore_std * torch.randn(mean.shape)
        logp = _normal_logp(torch, z.detach(), mean, explore_std)
    else:
        z = mean
        logp = torch.zeros(n, dtype=torch.float32)
    acts = _squash_adv(torch, z).detach().numpy()
    perturbed = [apply_perturbation(base, budget, a.astype(float)) for a in acts]
    rets, _, _ = _rollout_maker_batch(
        torch,
        maker,
        perturbed,
        seeds,
        explore=maker_explore_std > 0.0,
        explore_std=maker_explore_std,
    )
    return rets, logp, perturbed, acts.astype(float)


@dataclass(frozen=True)
class ARLTrainResult:
    """Training record for one alternating ARL run (or benign baseline)."""

    maker: Any  # torch actor (field kept off repr/compare)
    adversary: Any | None
    maker_loss_curve: Array
    adv_loss_curve: Array | None
    maker_return_curve: Array  # batch-mean maker return per maker update
    n_maker_updates: int
    n_adv_updates: int
    n_episodes: int
    seed: int


def _reinforce_step(
    torch: Any,
    opt: Any,
    logp: Any,
    advantages_np: Array,
) -> float:
    """One episodic-REINFORCE descent step; returns the loss value."""
    adv_t = torch.as_tensor(advantages_np, dtype=torch.float32)
    loss = -(logp * adv_t).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach().numpy())


def _adversary_greedy_action(
    torch: Any, adv: Any, base: ArlMMEnvParams, budget: AdversaryBudget
) -> Array:
    """The adversary's current worst-case perturbation (mean action)."""
    desc = np.tile(_adv_descriptor(base, budget), (4, 1))
    with torch.no_grad():
        desc_t = torch.as_tensor(desc, dtype=torch.float32).unsqueeze(0)
        hx = adv.init_h(1)
        mean = None
        for k in range(desc_t.shape[1]):
            mean, hx = adv(desc_t[:, k, :], hx)
        if mean is None:
            raise RuntimeError("adversary descriptor sequence is empty")
        return np.asarray(_squash_adv(torch, mean).squeeze(0).numpy(), dtype=np.float64)


def train_arl_pair(
    base: ArlMMEnvParams,
    budget: AdversaryBudget,
    *,
    n_rounds: int = 4,
    adv_epochs: int = 3,
    maker_epochs: int = 5,
    batch_size: int = 24,
    hidden: int = 16,
    lr: float = 0.05,
    adv_lr: float = 0.1,
    explore_std: float = 0.5,
    benign_frac: float = 0.4,
    seed: int = 0,
) -> ARLTrainResult:
    """Train maker + adversary by the alternating zero-sum ARL loop.

    Each round freezes the maker and runs ``adv_epochs`` adversary REINFORCE
    updates (reward = negated maker return) against the maker's greedy
    policy, then freezes the adversary at its current greedy perturbation
    and runs ``maker_epochs`` maker best-response updates against that
    perturbed environment — the alternating-descent scheme of the paper.
    A ``benign_frac`` share of each maker batch keeps the nominal
    environment so quoting stays profitable under benign conditions, and
    benign/adversarial updates take separate baselines so advantages are
    not confounded by environment identity. Deterministic given ``seed``:
    single-thread CPU torch, seeded ``torch.manual_seed``, and per-episode
    seeded environments.
    """
    torch = _torch()
    if not isinstance(base, ArlMMEnvParams):
        raise TypeError("base must be an ArlMMEnvParams")
    if not isinstance(budget, AdversaryBudget):
        raise TypeError("budget must be an AdversaryBudget")
    nr = _int_at_least(n_rounds, 1, "n_rounds")
    ae = _int_at_least(adv_epochs, 1, "adv_epochs")
    me = _int_at_least(maker_epochs, 1, "maker_epochs")
    bs = _int_at_least(batch_size, 2, "batch_size")
    _int_at_least(hidden, 1, "hidden")
    rate = _pos_finite(lr, "lr")
    a_rate = _pos_finite(adv_lr, "adv_lr")
    std = _pos_finite(explore_std, "explore_std")
    bf = float(benign_frac)
    if not math.isfinite(bf) or not 0.0 <= bf <= 1.0:
        raise ValueError(f"benign_frac must lie in [0, 1], got {benign_frac!r}")
    sd = _seed_int(seed)

    torch.manual_seed(sd)
    torch.set_num_threads(1)
    maker = _build_lstm_actor(torch, MAKER_OBS_DIM, hidden, MAKER_ACT_DIM)
    adversary = _build_lstm_actor(torch, ADV_OBS_DIM, hidden, ADV_ACT_DIM)
    opt_m = torch.optim.Adam(maker.parameters(), lr=rate)
    opt_a = torch.optim.Adam(adversary.parameters(), lr=a_rate)

    m_losses: list[float] = []
    a_losses: list[float] = []
    m_rets: list[float] = []
    rng_adv = np.random.default_rng(sd + 65_537)
    n_eps = 0
    n_mk = 0
    n_av = 0
    for rnd in range(nr):
        # Adversary phase: maker acts greedily; the adversary explores the
        # perturbation box and climbs -E[R_maker].
        maker.eval()
        adversary.train()
        for ep in range(ae):
            seeds = [sd + 500_009 * (rnd + 1) + 10_007 * ep + 101 * e for e in range(bs)]
            rets, logp, _, _ = _rollout_adversary_batch(
                torch,
                adversary,
                maker,
                base,
                budget,
                seeds,
                explore=True,
                explore_std=std,
                maker_explore_std=0.0,
            )
            n_eps += bs
            adv_r = -rets  # zero-sum: adversary maximizes -maker return
            adv = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-6)
            a_losses.append(_reinforce_step(torch, opt_a, logp, adv))
            n_av += 1
        # Maker phase: best-response to the adversary's greedy perturbation
        # plus a benign slice; each env family gets its own baseline.
        adversary.eval()
        maker.train()
        adv_action = _adversary_greedy_action(torch, adversary, base, budget)
        n_benign = int(round(bs * bf))
        n_adv_eps = bs - n_benign
        for ep in range(me):
            if n_adv_eps > 0:
                # Half the adversarial batch rides the adversary's greedy
                # worst case; the rest jitters around it so the best-response
                # covers a neighborhood of bad regimes, not a point.
                corners = [np.ones(4)] + [np.eye(4)[i] for i in range(4)]
                acts_seq: list[Array] = []
                for k in range(n_adv_eps):
                    if k < n_adv_eps // 2:
                        acts_seq.append(adv_action)
                    elif k < 3 * n_adv_eps // 4:
                        acts_seq.append(
                            np.clip(
                                adv_action + rng_adv.normal(0.0, 0.3, size=4),
                                -1.0,
                                1.0,
                            )
                        )
                    else:
                        acts_seq.append(corners[(rnd + ep + k) % len(corners)])
                adv_params_seq = [apply_perturbation(base, budget, a) for a in acts_seq]
                seeds = [sd + 700_001 * (rnd + 1) + 20_011 * ep + 211 * e for e in range(n_adv_eps)]
                rets, logp, _ = _rollout_maker_batch(
                    torch,
                    maker,
                    adv_params_seq,
                    seeds,
                    explore=True,
                    explore_std=std,
                )
                n_eps += n_adv_eps
                m_rets.append(float(rets.mean()))
                adv_m = (rets - rets.mean()) / (rets.std() + 1e-6)
                m_losses.append(_reinforce_step(torch, opt_m, logp, adv_m))
                n_mk += 1
            if n_benign > 0:
                seeds = [sd + 900_011 * (rnd + 1) + 30_011 * ep + 307 * e for e in range(n_benign)]
                rets_b, logp_b, _ = _rollout_maker_batch(
                    torch,
                    maker,
                    [base] * n_benign,
                    seeds,
                    explore=True,
                    explore_std=std,
                )
                n_eps += n_benign
                m_rets.append(float(rets_b.mean()))
                adv_b = (rets_b - rets_b.mean()) / (rets_b.std() + 1e-6)
                m_losses.append(_reinforce_step(torch, opt_m, logp_b, adv_b))
                n_mk += 1
    maker.eval()
    adversary.eval()
    return ARLTrainResult(
        maker=maker,
        adversary=adversary,
        maker_loss_curve=np.asarray(m_losses, dtype=float),
        adv_loss_curve=np.asarray(a_losses, dtype=float),
        maker_return_curve=np.asarray(m_rets, dtype=float),
        n_maker_updates=n_mk,
        n_adv_updates=n_av,
        n_episodes=n_eps,
        seed=sd,
    )


def train_baseline_maker(
    base: ArlMMEnvParams,
    *,
    n_updates: int,
    batch_size: int = 8,
    hidden: int = 16,
    lr: float = 0.05,
    explore_std: float = 0.5,
    seed: int = 0,
) -> ARLTrainResult:
    """Non-adversarial baseline: identical maker trained on the benign env.

    Same architecture, optimizer, episode budget per update, and total
    update count as the ARL maker — the paper's non-adversarial control arm.
    Deterministic given ``seed``.
    """
    torch = _torch()
    if not isinstance(base, ArlMMEnvParams):
        raise TypeError("base must be an ArlMMEnvParams")
    nu = _int_at_least(n_updates, 1, "n_updates")
    bs = _int_at_least(batch_size, 2, "batch_size")
    _int_at_least(hidden, 1, "hidden")
    rate = _pos_finite(lr, "lr")
    std = _pos_finite(explore_std, "explore_std")
    sd = _seed_int(seed)

    torch.manual_seed(sd)
    torch.set_num_threads(1)
    maker = _build_lstm_actor(torch, MAKER_OBS_DIM, hidden, MAKER_ACT_DIM)
    opt_m = torch.optim.Adam(maker.parameters(), lr=rate)
    m_losses: list[float] = []
    m_rets: list[float] = []
    for u in range(nu):
        seeds = [sd + 300_017 * (u + 1) + 40_009 * e for e in range(bs)]
        rets, logp, _ = _rollout_maker_batch(
            torch, maker, [base] * bs, seeds, explore=True, explore_std=std
        )
        m_rets.append(float(rets.mean()))
        adv_b = (rets - rets.mean()) / (rets.std() + 1e-6)
        m_losses.append(_reinforce_step(torch, opt_m, logp, adv_b))
    maker.eval()
    return ARLTrainResult(
        maker=maker,
        adversary=None,
        maker_loss_curve=np.asarray(m_losses, dtype=float),
        adv_loss_curve=None,
        maker_return_curve=np.asarray(m_rets, dtype=float),
        n_maker_updates=nu,
        n_adv_updates=0,
        n_episodes=nu * bs,
        seed=sd,
    )


# ---------------------------------------------------------------------------
# Bench entry point (flat dict[str, float|str]; ~seconds)
# ---------------------------------------------------------------------------


def arl_mm_bench(
    *,
    params: ArlMMEnvParams | None = None,
    budget: AdversaryBudget | None = None,
    n_rounds: int = 4,
    adv_epochs: int = 3,
    maker_epochs: int = 5,
    batch_size: int = 24,
    hidden: int = 16,
    lr: float = 0.05,
    adv_lr: float = 0.1,
    explore_std: float = 0.5,
    n_eval_episodes: int = 20,
    n_interior: int = 3,
    tail_alpha: float = 0.05,
    n_boot: int = 400,
    seed: int = 2,
) -> dict[str, float | str]:
    """Seeded CI-scale ARLMM bench: train both arms, compare left tails.

    Trains the ARL pair and the benign baseline, evaluates both on the
    adversarial environment grid, and returns a flat metric dict:
    ``synthetic_*`` left-tail and mechanism diagnostics plus ``str`` stamp
    keys (``label``/``dgp``/``claim``/``kind``/``data_source``). All values
    are SYNTHETIC correctness diagnostics on a synthetic environment —
    ``sim_internal_*`` accounting never enters this blob; no headline
    P&L-family metrics, no market evidence, no live-trading claim.
    """
    torch = _torch()  # fail closed with install guidance when nn extra absent
    _seed_int(seed)
    base = params if params is not None else ArlMMEnvParams()
    bud = budget if budget is not None else AdversaryBudget()
    if not isinstance(base, ArlMMEnvParams):
        raise TypeError("params must be an ArlMMEnvParams")
    if not isinstance(bud, AdversaryBudget):
        raise TypeError("budget must be an AdversaryBudget")
    ne = _int_at_least(n_eval_episodes, 2, "n_eval_episodes")
    a_tail = float(tail_alpha)
    if not math.isfinite(a_tail) or not 0.0 < a_tail < 1.0:
        raise ValueError(f"tail_alpha must lie in (0, 1), got {tail_alpha!r}")

    arl = train_arl_pair(
        base,
        bud,
        n_rounds=n_rounds,
        adv_epochs=adv_epochs,
        maker_epochs=maker_epochs,
        batch_size=batch_size,
        hidden=hidden,
        lr=lr,
        adv_lr=adv_lr,
        explore_std=explore_std,
        seed=seed,
    )
    base_maker = train_baseline_maker(
        base,
        n_updates=arl.n_maker_updates,
        batch_size=batch_size,
        hidden=hidden,
        lr=lr,
        explore_std=explore_std,
        seed=seed + 5_551,
    )
    adv_greedy = _adversary_greedy_action(torch, arl.adversary, base, bud)
    grid = adversary_env_grid(
        base, bud, n_interior=n_interior, seed=seed + 97, extra_actions=[adv_greedy]
    )
    ev = evaluate_left_tail(
        {"arl": arl.maker, "baseline": base_maker.maker},
        grid,
        n_episodes=ne,
        seed=seed + 20_003,
        alpha=a_tail,
    )

    adv_names = [n for n, _ in grid if n != "benign"]
    rets_a = {n: ev.returns["arl"][n] for n in adv_names}
    rets_b = {n: ev.returns["baseline"][n] for n in adv_names}
    q05_a = {n: ev.stats["arl"][n]["q_alpha"] for n in adv_names}
    q05_b = {n: ev.stats["baseline"][n]["q_alpha"] for n in adv_names}
    pooled_a = np.concatenate([rets_a[n] for n in adv_names])
    pooled_b = np.concatenate([rets_b[n] for n in adv_names])
    pooled_stats_a = left_tail_stats(pooled_a, alpha=a_tail)
    pooled_stats_b = left_tail_stats(pooled_b, alpha=a_tail)

    benign_mean = float(ev.stats["baseline"]["benign"]["mean"])
    adv_mean_b = float(np.mean([ev.stats["baseline"][n]["mean"] for n in adv_names]))
    learned = "adv_learned_0"
    learned_a = ev.stats["arl"][learned]
    learned_b = ev.stats["baseline"][learned]
    bias = inventory_bias_check(
        np.concatenate([ev.terminal_inventory["arl"][n] for n in adv_names]),
        np.concatenate([ev.terminal_inventory["baseline"][n] for n in adv_names]),
        n_boot=n_boot,
        seed=seed + 77_777,
    )

    # Hawkes cluster diagnostics on a fresh benign episode's realized path.
    probe = ArlMMEnv(base)
    probe.reset(seed + 88_001)
    cluster = hawkes_cluster_stats(
        probe.mo_times,
        horizon=base.horizon,
        mu=base.hawkes_mu,
        alpha=base.hawkes_alpha,
        beta=base.hawkes_beta,
    )
    poisson_params = replace(base, hawkes_alpha=0.0)
    probe_p = ArlMMEnv(poisson_params)
    probe_p.reset(seed + 88_001)
    cluster_p = hawkes_cluster_stats(probe_p.mo_times, horizon=base.horizon, fit_mle=False)

    return {
        "label": "SYNTHETIC",
        "data_source": ARL_MM_REVISION,
        "research_only": 1.0,
        "live_pnl_claim": 0.0,
        "claim": "simulator_internal_diagnostic_only",
        "kind": "arl_mm_bench",
        "dgp": "hawkes_gbm_impact_zero_sum",
        "synthetic_arl_left_tail_q05_improvement": float(
            pooled_stats_a["q_alpha"] - pooled_stats_b["q_alpha"]
        ),
        "synthetic_arl_left_tail_es05_improvement": float(
            pooled_stats_a["es_alpha"] - pooled_stats_b["es_alpha"]
        ),
        "synthetic_arl_left_tail_q05_arl": pooled_stats_a["q_alpha"],
        "synthetic_arl_left_tail_q05_baseline": pooled_stats_b["q_alpha"],
        "synthetic_arl_left_tail_env_improvement_frac": float(
            np.mean([1.0 if q05_a[n] > q05_b[n] else 0.0 for n in adv_names])
        ),
        "synthetic_adversary_effectiveness_baseline": float(benign_mean - learned_b["mean"]),
        "synthetic_adversary_effectiveness_pooled": float(benign_mean - adv_mean_b),
        "synthetic_arl_left_tail_q05_learned_env": float(learned_a["q_alpha"]),
        "synthetic_arl_left_tail_q05_learned_env_baseline": float(learned_b["q_alpha"]),
        "synthetic_arl_left_tail_q05_learned_improvement": float(
            learned_a["q_alpha"] - learned_b["q_alpha"]
        ),
        "synthetic_arl_left_tail_es05_learned_improvement": float(
            learned_a["es_alpha"] - learned_b["es_alpha"]
        ),
        "synthetic_adversary_greedy_mu": float(adv_greedy[0]),
        "synthetic_adversary_greedy_alpha": float(adv_greedy[1]),
        "synthetic_adversary_greedy_sigma": float(adv_greedy[2]),
        "synthetic_adversary_greedy_eta": float(adv_greedy[3]),
        "synthetic_arl_adversarial_mean_return": float(pooled_stats_a["mean"]),
        "synthetic_baseline_adversarial_mean_return": float(pooled_stats_b["mean"]),
        "synthetic_inventory_bias_gap": float(bias["bias_gap"]),
        "synthetic_inventory_bias_p_more": float(bias["p_more_bias"]),
        "synthetic_terminal_abs_inv_arl": float(bias["mean_abs_q_a"]),
        "synthetic_terminal_abs_inv_baseline": float(bias["mean_abs_q_b"]),
        "synthetic_hawkes_fano_factor": float(cluster["fano_factor"]),
        "synthetic_hawkes_fano_factor_poisson": float(cluster_p["fano_factor"]),
        "synthetic_hawkes_intensity_max_over_mu": float(cluster["intensity_max_over_mu"]),
        "synthetic_hawkes_branching_mle": float(cluster["branching_ratio_mle"]),
        "synthetic_hawkes_branching_true": float(base.hawkes_alpha),
        "synthetic_hawkes_n_mo": float(cluster["n_mo"]),
        "n_training_episodes_arl": float(arl.n_episodes),
        "n_eval_episodes": float(ne),
        "n_env_grid": float(len(grid)),
        "horizon": float(base.horizon),
        "n_steps": float(base.n_steps),
        "tail_alpha": a_tail,
        "seed": float(seed),
    }
