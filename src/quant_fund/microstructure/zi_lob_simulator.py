"""Zero-intelligence limit-order-book simulator + classical market-making baselines.

**Labeled SYNTHETIC** research infrastructure (lane B4-i): an event-driven ZI
("Santa Fe") LOB with a price-time-priority matching engine, closed-form
Avellaneda-Stoikov / GLFT quoting baselines, a session runner, and emergent
microstructure diagnostics (square-root impact, order-flow autocorrelation,
spread/depth phase metrics). Foundation for the Moret & Lillo (2026) RL
market-maker lane (RL layer lands later; this module is pure numpy).

Composition / differentiation from existing modules (nothing here modifies
them):

- ``microstructure.synthetic_lob`` builds **static L2 snapshot panels** from
  OHLCV bars (vectorized book metrics for Northset fusion). This module is the
  **dynamic** complement: the book *emerges* from Poisson order flow through a
  FIFO matching engine, so queue position, cancellations, and endogenous
  spread/depth exist as state, not as bar-derived proxies. No shared code path.
- ``models.market_making`` owns the Avellaneda-Stoikov (2008) closed form.
  ``avellaneda_stoikov_quotes`` **composes** it (single source of truth) and
  adds tick-grid snapping for the discrete simulator. ``glft_quotes`` adds the
  Gueant-Lehalle-Fernandez-Tapia asymptotic quotes (Eqs. 2-3 of Moret & Lillo
  2026), which are not in ``models.market_making``.
- ``execution/impact.py`` owns the parametric impact formulas
  (``sqrt_impact_bps``, ``pow_law_total_impact``). ``metaorder_impact_slope``
  is the *emergent* counterpart: it measures the log-log impact exponent of
  metaorders executed inside this simulator and checks it against the
  square-root law (exponent 0.5).

Honesty: every output is a SYNTHETIC correctness diagnostic, never market
evidence. Mark-to-market accounting is simulator-internal (``sim_internal_*``
keys), must never be headlined, and there is no broker connectivity or
live-trading claim anywhere in this module.

References:
- Cont, Stoikov, Talreja (2010). A stochastic model for order book dynamics.
  *Operations Research* 58(1):191-205 — ZI-LOB queueing model.
- Avellaneda, Stoikov (2008). High-frequency trading in a limit order book.
  *Quantitative Finance* 8(3):217-224.
- Gueant, Lehalle, Fernandez-Tapia (2012). Optimal portfolio liquidation with
  limit orders. *Operations Research* 60(1):1167-1187 — GLFT closed form.
- Moret, Lillo (2026). Deep learning of robust market making under
  regime-switching order flow. arXiv:2609.11614 — ZI (Santa Fe) LOB setup and
  calibration lambda=0.06, mu=0.10, theta_cxl=0.02 per second (Eq. 4, LOBSTER
  AMZN L3); GLFT quotes Eqs. 2-3; regime generation on the MO clock.
- Rosenzweig (2026). Agentic limit order books: phase transitions and market
  impact. arXiv:2609.31260 — spread/depth phase-transition diagnostics and
  impact-regime classification.
- Donier, Bonart, Mastromatteo, Bouchaud (2015). A fully non-linear,
  scale-invariant theory of market impact. arXiv:1504.06829 (*Physica A*
  449:70-81) — square-root impact emerging from zero-intelligence books.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Side = Literal["buy", "sell"]

#: Resting-order placement class at submit time, relative to the own-side
#: touch: ``"join"`` lands at the touch, ``"improve"`` lands strictly inside
#: the open spread (a new own-side best), ``"deep"`` lands outside it.
PlacementClass = Literal["join", "improve", "deep"]
PLACEMENT_CLASSES: tuple[PlacementClass, ...] = ("join", "improve", "deep")

MM_TAG = "mm_session"
ZI_LOB_REVISION = "SYNTHETIC_ZI_LOB_v1"

# ---------------------------------------------------------------------------
# Fail-closed validation helpers
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


def _check_side(side: str) -> Side:
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    return side  # type: ignore[return-value]


def _check_size_pmf(
    pmf: tuple[tuple[int, float], ...] | None, name: str
) -> tuple[tuple[int, float], ...] | None:
    """Validate a size pmf: ``((size, weight), ...)`` with int sizes >= 1.

    Weights need only be positive and finite — they are normalized at
    draw time. ``None`` (default) means unit-size events and consumes
    zero RNG draws, preserving the legacy bit-identical event stream.
    """
    if pmf is None:
        return None
    if not isinstance(pmf, (tuple, list)) or len(pmf) == 0:
        raise ValueError(f"{name} must be a non-empty (size, weight) table")
    out: list[tuple[int, float]] = []
    total = 0.0
    for entry in pmf:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"{name} entries must be (size, weight) pairs")
        size, w = entry
        if isinstance(size, bool) or int(size) < 1:
            raise ValueError(f"{name} sizes must be ints >= 1, got {size!r}")
        w = float(w)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"{name} weights must be positive and finite, got {w!r}")
        out.append((int(size), w))
        total += w
    if total <= 0.0:  # pragma: no cover - positive-weight guard above
        raise ValueError(f"{name} weights must sum > 0")
    return tuple(out)


# ---------------------------------------------------------------------------
# Self-exciting event clock (multivariate Hawkes modulation)
# ---------------------------------------------------------------------------

# Event-type order for the Hawkes kernel: limit, market, cancel.
HAWKES_TYPES: tuple[str, str, str] = ("limit", "market", "cancel")


@dataclass(frozen=True)
class HawkesClockSpec:
    """Excitation kernel for an optional Hawkes event clock.

    ``kernel[i][j]`` is the intensity jump (events/s) that one event of type
    ``i`` adds to the type-``j`` intensity; the jump decays as
    ``exp(-beta * dt)`` with a shared decay rate ``beta``. Event types are
    indexed by ``HAWKES_TYPES``: 0 = limit, 1 = market, 2 = cancel.

    A jump of ``alpha`` decaying at ``beta`` contributes branching ratio
    ``alpha / beta`` expected direct children, so the branching matrix is
    ``kernel / beta``. Fail-closed unless the matrix is finite,
    non-negative, 3x3 and strictly sub-critical (spectral radius < 1) —
    a super-critical kernel explodes and would silently fabricate a tape.
    """

    kernel: tuple[tuple[float, float, float], ...]
    beta: float
    # Optional multi-timescale excitation: ``rates`` gives R decay banks and
    # ``bank_weights`` the share of each kernel jump deposited in each bank
    # (must sum to 1). A bank mixture approximates a power-law kernel
    # (Omori-type clustering tail) while staying exactly Markovian for Ogata
    # thinning. ``None`` keeps the single shared decay ``beta``.
    rates: tuple[float, ...] | None = None
    bank_weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kernel, (tuple, list)) or len(self.kernel) != 3:
            raise ValueError("kernel must be a 3x3 matrix")
        k = np.asarray(self.kernel, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("kernel must be a 3x3 matrix")
        if not np.all(np.isfinite(k)) or bool((k < 0.0).any()):
            raise ValueError("kernel entries must be non-negative and finite")
        _pos_finite(self.beta, "beta")
        rates = self.rates if self.rates is not None else (self.beta,)
        weights = self.bank_weights
        if weights is None:
            weights = tuple(1.0 / len(rates) for _ in rates)
        if len(rates) != len(weights) or len(rates) < 1:
            raise ValueError("rates and bank_weights must share a non-empty length")
        for r in rates:
            _pos_finite(r, "rates")
        wsum = sum(_nonneg_finite(w, "bank_weights") for w in weights)
        if not math.isclose(wsum, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"bank_weights must sum to 1, got {wsum!r}")
        object.__setattr__(self, "rates", tuple(float(r) for r in rates))
        object.__setattr__(self, "bank_weights", tuple(float(w) for w in weights))
        # Children per edge = kernel_ij * H, H = sum_r w_r / beta_r.
        h = float(sum(w / r for w, r in zip(weights, rates, strict=True)))
        rho = float(np.max(np.abs(np.linalg.eigvals(k * h))))
        if not math.isfinite(rho) or rho >= 1.0:
            raise ValueError(
                f"Hawkes kernel must be sub-critical (spectral radius < 1), got {rho:.4f}"
            )
        object.__setattr__(self, "kernel", tuple(tuple(float(x) for x in row) for row in k))

    def branching_matrix(self) -> tuple[tuple[float, float, float], ...]:
        """Effective children-per-event matrix ``kernel * H``."""
        h = float(
            sum(
                w / r
                for w, r in zip(
                    self.bank_weights or (1.0,), self.rates or (self.beta,), strict=True
                )
            )
        )
        k = np.asarray(self.kernel, dtype=np.float64) * h
        return tuple((float(row[0]), float(row[1]), float(row[2])) for row in k)


@dataclass(frozen=True)
class RateRegimeSpec:
    """Markov-modulated baseline scaling (MMPP) for the event clock.

    ``scales[s]`` multiplies the per-type base rates ``(lo, mo, cxl)``
    while the chain sits in state ``s``; ``stay_probs[s]`` is the
    per-event probability of remaining in ``s`` (geometric dwell).
    Transitions move to a uniformly-chosen other state. Combined with
    ``hawkes`` this is a Cox-Hawkes hybrid: the modulated bases feed the
    thinning clock. ``None`` (or a single all-ones state) consumes zero
    RNG draws and is bit-identical to the unmodulated clock.
    """

    scales: tuple[tuple[float, float, float], ...]
    stay_probs: tuple[float, ...]
    start: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.scales, (tuple, list)) or len(self.scales) < 1:
            raise ValueError("scales must be a non-empty tuple of 3-vectors")
        if len(self.scales) != len(self.stay_probs):
            raise ValueError("scales and stay_probs must share a length")
        for s in self.scales:
            if not isinstance(s, (tuple, list)) or len(s) != 3:
                raise ValueError("each scale must be a 3-vector")
            for v in s:
                _nonneg_finite(float(v), "scales")
        for p in self.stay_probs:
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(f"stay_probs entries must be in [0, 1], got {p!r}")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ValueError(f"start must be an int, got {self.start!r}")
        if not 0 <= self.start < len(self.scales):
            raise ValueError(f"start out of range: {self.start!r}")
        object.__setattr__(
            self,
            "scales",
            tuple((float(s[0]), float(s[1]), float(s[2])) for s in self.scales),
        )
        object.__setattr__(self, "stay_probs", tuple(float(p) for p in self.stay_probs))


class RateRegimeFlow:
    """Markov-modulated baseline multiplier driven per event.

    ``ZILobSimulator.step`` multiplies the (lo, mo, cxl) bases by the
    current state's scale vector before feeding the clock. The transition
    check consumes one uniform per event when the chain can move; a
    single-state spec (or a stay_prob of exactly 1) consumes zero draws.
    """

    def __init__(self, spec: RateRegimeSpec, rng: np.random.Generator) -> None:
        self._scales = np.asarray(spec.scales, dtype=np.float64)
        self._stay = np.asarray(spec.stay_probs, dtype=np.float64)
        self._rng = rng
        self.state = int(spec.start)
        self.n_transitions = 0
        self._frozen = len(spec.stay_probs) == 1 or float(self._stay[self.state]) >= 1.0

    def scale(self) -> Array:
        """Current ``(lo, mo, cxl)`` multipliers."""
        s: Array = self._scales[self.state]
        return s

    def advance(self) -> None:
        """Per-event transition step (uniform-over-others move)."""
        if self._frozen:
            return
        if float(self._rng.random()) >= self._stay[self.state]:
            n = self._scales.shape[0]
            jump = 1 + int(float(self._rng.random()) * (n - 1))
            self.state = (self.state + jump) % n
            self.n_transitions += 1
            if float(self._stay[self.state]) >= 1.0:
                self._frozen = True


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZILobConfig:
    """Zero-intelligence LOB parameters (Moret & Lillo 2026, Sec. 2).

    Unit-size orders on a discrete tick grid. Three exogenous Poisson flows:
    limit orders at rate ``lam`` per side per level inside a ``band`` of levels
    around the opposite best quote; market orders of total intensity ``2*mu``
    (buy with probability ``p_buy``); cancellations at rate ``theta_cxl`` per
    resting order. All rates are per simulated second.

    ``density_exponent`` shapes the limit-order placement distance ``d`` (in
    ticks from the anchor) via ``P(d) ∝ d**density_exponent`` over ``d ∈
    [1, band]``. ``0`` = uniform placement → flat stationary book density →
    *linear* metaorder impact (the Moret & Lillo market-making setting). ``1``
    = triangular placement → book density rising linearly with distance →
    cumulative liquidity ∝ δ² → *square-root* metaorder impact (the
    Donier et al. 2015 / square-root-law regime; impact exponent = 1/(1+β)).
    This is an exogenous zero-intelligence placement law, not agent learning.

    ``anchor`` selects the placement reference. ``"touch"`` (default) anchors
    LOs to the current best opposite quote — the Moret & Lillo market-making
    setting, where the band follows the price. ``"ref"`` anchors LOs to a slow
    reference level (an EMA of the mid with half-life ``ref_halflife`` seconds;
    ``0`` freezes it at the seed mid), so liquidity is deposited in *absolute*
    price space and diffuses there. Only ``"ref"`` anchoring reproduces the
    emergent square-root metaorder impact of the volume-diffusion theory; with
    ``"touch"`` the band re-forms ahead of a walking metaorder and impact is
    linear. Crossing placements under ``"ref"`` are dropped (that aggressiveness
    is already represented by the market-order flow).
    """

    s0: float = 100.0
    tick: float = 0.01
    lam: float = 0.06
    mu: float = 0.10
    theta_cxl: float = 0.02
    p_buy: float = 0.5
    band: int = 5
    init_levels: int = 3
    init_depth: int = 5
    density_exponent: float = 0.0
    anchor: str = "touch"
    ref_halflife: float = 0.0
    # ``ref_fill_gain`` >= 0: informed-flow anchoring — each market-order
    # fill shifts the slow reference level by ``ref_fill_gain`` ticks per
    # unit consumed, in the fill direction. The book's latent-value
    # estimate moves on trade prints (Glosten–Milgrom), so a fill
    # produces *persistent* post-fill drift even when the aggressor flow
    # itself has no memory. Bites only with ``anchor="ref"`` (the
    # reference is what LOs deposit around); 0 is bit-identical legacy
    # and consumes zero extra RNG draws.
    ref_fill_gain: float = 0.0
    # ``refill_cooldown`` >= 0: per-price vacancy memory — when a book
    # level empties (fill, touch pull, or last-cancel), ZI placements
    # that would land on that exact (side, level) are suppressed for this
    # many events: the vacated price stays empty on a timescale, matching
    # the tape's measured refill hazard (~median 86 events, ~51% never
    # re-occupied within 400 on AMZN). 0 = disabled (bit-identical: no
    # extra RNG draws or bookkeeping).
    refill_cooldown: int = 0
    # ``lo_tilt_gain`` >= 0: post-fill LO side bias (accommodation) — each
    # fill adds its aggressor sign times this gain to a tilt state in
    # [-1, 1], and the LO side draw becomes P(buy) = (1 + tilt) / 2. On
    # the tape the book leans *into* the flow direction for ~100 events
    # after a fill (depth_tilt.v1: signed tilt +0.16 -> +0.07); flat
    # flow cannot express that. ``lo_tilt_decay`` in [0, 1] is the
    # per-event geometric decay of the tilt state. gain == 0 disables
    # the mechanism bit-identically (tilt pinned at 0; the side draw
    # consumes the same RNG).
    lo_tilt_gain: float = 0.0
    lo_tilt_decay: float = 0.0
    # ``hit_narrow_dist`` >= 0: for ``hit_narrow_window`` events after a
    # fill, LO placements on the *unhit* side clamp their placement
    # distance to ``<= hit_narrow_dist`` — the tape's accommodation
    # (depth_tilt.v1: after a buy fill the bid stays heavier ~100
    # events) is the unhit side stacking near-touch, not a side-rate
    # shift: global side bias lands mostly deep under the band law and
    # barely registers at the touch. 0 disables bit-identically.
    hit_narrow_dist: int = 0
    hit_narrow_window: int = 0
    # ``place_mode_frac`` in [0, 1]: when > 0 the LO placement distance
    # law switches from the monotone P(d) ~ d**density_exponent family
    # to a shifted Binomial, ``d ~ 1 + Bin(band - 1, place_mode_frac)``
    # — a hump-shaped law with mode at ``1 + frac * (band - 1)`` ticks.
    # The tape's placement law peaks at 8-13 ticks with thin mass below
    # 5 (place_law.v1); monotone families cannot express that shape. 0
    # keeps the legacy law bit-identically (same weights, same draws).
    place_mode_frac: float = 0.0
    seed: int = 0
    # ``lo_offset`` shifts the touch-anchored LO anchor back by this many
    # ticks: a buy deposits at ``best_ask - lo_offset - dist`` instead of
    # ``best_ask - dist``. Deep anchoring floors the spread near
    # ``lo_offset + 1`` — the zero-intelligence version of
    # adverse-selection-aware quoting (makers refuse the touch). 0 is
    # bit-identical to the legacy placement.
    lo_offset: int = 0
    # ``iceberg_reload`` ∈ [0, 1]: probability that consuming the front
    # order of a level immediately re-rests one unit at the SAME level
    # tagged ``iceberg`` — hidden reserve liquidity that refills after
    # each fill (the synthetic iceberg approximation). A fill whose
    # maker tag is ``iceberg`` counts in ``n_hidden_fills``: fills on
    # liquidity that was not displayed before execution. 0 is
    # bit-identical to the legacy matcher (zero RNG draws consumed).
    iceberg_reload: float = 0.0
    # ``iceberg_reload_mode``: ``per_unit`` (default — the draw refills
    # the level whether or not any visible depth survives) vs
    # ``residual`` — the refill only fires when the level still holds
    # visible units after the fill, i.e. hidden persistence *stacks on*
    # a living level but never resurrects a cleared one. Per-unit reload
    # makes the touch nearly unkillable under iceberg liquidity (a
    # mid-burst reload keeps the level alive however deep the burst
    # cuts), which suppresses the emptied-touch channel; residual mode
    # is the tape's structure — a level that fully empties stays empty
    # until visible flow re-sites there. Consumes no extra RNG either
    # way (the draw precedes the gate), so ``per_unit`` is
    # bit-identical.
    iceberg_reload_mode: str = "per_unit"
    # ``iceberg_budget``: maximum number of hidden refills each
    # (side, level) may spend over the whole run — the real iceberg is
    # a finite reserve consumed once and gone. 0 (default) is
    # unlimited, i.e. bit-identical to the legacy mechanism.
    iceberg_budget: int = 0
    # ``lo_offset_gain`` couples the LO anchor offset to MO excitation:
    # the effective offset is ``lo_offset + round(gain * e_MO)`` where
    # ``e_MO`` is the Hawkes excitation state summed over banks — makers
    # retreat while fills cluster, so the spread widens exactly when
    # toxicity is high and relaxes as excitation decays (the tape's
    # post-fill spread kernel). Requires ``hawkes``; 0 ignores it.
    lo_offset_gain: float = 0.0
    # ``lo_improve_frac`` ∈ [0, 1]: probability an LO uses improve
    # anchoring instead of the deep anchor — placed uniformly on the
    # open spread [own_touch, opp_touch - 1], so dist 0 joins the touch
    # and anything deeper sits strictly inside. The deep-anchor kernel
    # can never emit inside-spread flow (``off + dist < spread`` is
    # unreachable once the spread floors near ``lo_offset + 1``); the
    # tape's 10.3% inside-spread share and join flow need this second
    # placement component. Under ``anchor="ref"`` the same share of flow
    # lands strictly inside the spread (uniform on (own_touch,
    # opp_touch)) when ``place_join_frac``/``lo_improve_frac`` are on.
    # 0 is bit-identical to the legacy placement.
    lo_improve_frac: float = 0.0
    # ``place_join_frac`` ∈ [0, 1]: under ``anchor="ref"``, probability an
    # LO joins the own-side touch (level = own best) instead of drawing
    # from the distance law. With ``lo_improve_frac`` it forms the
    # join/improve/stack mixture the tape shows (place_law.v1: ~12% of
    # submissions at the touch, ~10% inside the spread). When either
    # knob is on the ref path consumes ONE extra uniform for the mixture
    # pick; both at 0 stays bit-identical (no extra draw).
    place_join_frac: float = 0.0
    # ``crown_stack_frac`` ∈ [0, 1]: probability an LO arrival stacks
    # into the near-touch crown — buys land on ``[bb - crown_stack_span,
    # bb]``, sells on ``[ba, ba + crown_stack_span]`` — instead of the
    # default anchor draw. The tape carries ~21% of visible top-10
    # depth within 3 ticks of each touch (crown_density.v1); the sim's
    # arms reach only 3-11%, and the wide book's emptied-touch reveal
    # overshoots (6.9 vs 3.8 ticks) because nothing rests in the crown.
    # 0 is bit-identical (no extra draws). On the ``anchor="ref"`` path
    # the crown slice joins the ``u_mix`` partition; on the touch-anchor
    # path it is drawn (one extra uniform) only when nonzero, between
    # the chase marker and the improve slice.
    crown_stack_frac: float = 0.0
    # ``crown_stack_span`` >= 0: depth (ticks behind the own touch) of
    # the crown band. Level is uniform on the closed span, so span 0
    # degenerates to join-the-touch.
    crown_stack_span: int = 3
    # ``crown_offset`` >= 0: distance (ticks) the crown band starts
    # behind the own touch — buys land on ``[bb - crown_offset -
    # crown_stack_span, bb - crown_offset]``. The tape's crown is dense
    # *behind* a thin touch (depth_consumption.v1: median fill eats 90%
    # of the touch, 47% full sweeps), so offset > 0 keeps the touch
    # empty-able while the band stacks. 0 preserves the original band
    # (touch included); only live when ``crown_stack_frac`` is nonzero.
    crown_offset: int = 0
    # ``crown_cap`` >= 0: maximum resting depth (orders) at a level for a
    # new crown stack to land there; a capped-out draw falls through to
    # the default anchor placement. The tape's crown is dense but
    # bounded (levels 1-3 sweepable — 47% of fills empty the touch);
    # an unbounded crown accumulates until nothing can empty it. 0 is
    # unbounded (original semantics); only live when ``crown_stack_frac``
    # is nonzero.
    crown_cap: int = 0
    # ``crown_size_pmf``: optional ``((size, weight), ...)`` table drawn
    # per crown placement instead of ``lo_size_pmf``. The tape's crown
    # is a few LARGE orders (60-200 shares each), not many unit orders
    # — a voluminous-but-shallow band that stays sweepable. ``None``
    # means crown placements use the shared LO size draw bit-identically.
    crown_size_pmf: tuple[tuple[int, float], ...] | None = None
    # ``near_level_cap`` >= 0: maximum resting units at any level within
    # ``near_level_span`` ticks behind a side's own touch — an LO arrival
    # landing in that band on a full level is refused (makers decline to
    # join a full queue: queue_fate.v1 shows join fill rate collapsing
    # with queue depth). The cap is on the BAND, not just the touch —
    # deep queues that promote to touch must arrive already thin;
    # ice_budget.v1 diagnosed bounded total depth at the touch as the
    # missing ingredient for the tape's 47% emptied-touch share. 0 is
    # unbounded and bit-identical (pure gate, no extra draws). Requotes
    # and chase re-sites respect the cap (they are visible flow); only
    # iceberg re-rests bypass it — the hidden reserve is not part of the
    # visible queue the cap bounds.
    near_level_cap: int = 0
    # ``near_level_span`` >= 0: how far behind the own touch (in ticks)
    # the ``near_level_cap`` band reaches. Levels beyond the band are
    # uncapped; span 0 caps the touch only.
    near_level_span: int = 3
    # ``touch_pull`` ∈ [0, 1]: after a fill, probability the NEW front
    # order on the hit side is pulled — the tape's instant re-quote
    # retreat (spread widens the moment liquidity is consumed, before
    # any new deposit arrives). 0 is bit-identical legacy (zero draws).
    touch_pull: float = 0.0
    # ``cxl_touch_bias`` ∈ [0, 1]: probability a cancellation event picks
    # the front order at a touch instead of a uniform outstanding order.
    # On the real tape cancels concentrate at the touch (propensity ~1.4x
    # uniform, falling to ~0.7 deep — the re-quote cycle churns the
    # front, not the back).
    cxl_touch_bias: float = 0.0
    # ``cxl_dist_decay`` > 0 changes the biased branch into a
    # distance-decaying propensity kernel: the biased cancel draws a
    # resting order with weight ``exp(-d / L)`` where ``d`` is its
    # distance from its own side's touch. The real profile is a
    # near-touch RING (d1-3 propensity 1.44 > touch 1.37), which the
    # pure front-pick cannot express; ``L`` ~ 3 reproduces the ring.
    # 0 keeps the front-order pick bit-identical.
    cxl_dist_decay: float = 0.0
    # ``cxl_requote`` ∈ [0, 1]: probability a *biased* cancel is a
    # re-quote — the removed order is immediately replaced by a fresh
    # order on the same side at the same level (back of that level's
    # FIFO). The tape's cancel churn is cancel+replace, so the book
    # keeps its depth while resting lifetimes collapse toward the
    # real ~0.8s deleted median. Applies only inside the biased
    # branches; 0 keeps every path bit-identical.
    cxl_requote: float = 0.0
    # ``cxl_unhit_relief`` ∈ [0, 1]: for ``cxl_unhit_window`` events
    # after a fill, a canceled event on the UNHIT side is rerouted to
    # the hit side with this probability — the sampled order is drawn
    # uniformly from the hit side's resting depth. The tape's post-fill
    # accommodation is add-driven, not cancel-driven: unhit cancels per
    # fill stay ~equal to hit-side ones (aftermath_flow.v1) while the
    # sim's depth-proportional kernel over-cancels the side that just
    # stacked. Both knobs at 0 keep every path bit-identical.
    cxl_unhit_relief: float = 0.0
    # ``cxl_unhit_damp`` ∈ [0, 1]: for ``cxl_unhit_window`` events after
    # a fill, a cancel event whose depth-proportional pick lands on the
    # UNHIT side is suppressed with this probability — the sampled order
    # survives. On the tape the unhit side's *count* of cancels stays
    # equal to the hit side's despite holding ~1.6x the adds — a lower
    # per-depth cancel hazard, i.e. protection, not relocation. Unlike
    # ``cxl_unhit_relief`` (which reroutes the mass onto the hit side),
    # damping removes it. Marker shared with the relief/narrow channels;
    # 0 keeps every path bit-identical.
    cxl_unhit_damp: float = 0.0
    # ``cxl_unhit_damp_decay`` > 0: the suppression probability decays
    # exponentially with events-since-fill, ``damp * exp(-elapsed/decay)``.
    # The tape's unhit-side protection is heavy in the first ~10 events and
    # gone by ~50 (unhit cxl/add 0.61 -> 0.90 -> 0.93 across the bench
    # windows), so a flat window over-shields late. 0 keeps the flat kernel
    # bit-identical; the marker's fill time is already carried in
    # ``_hit_retreat``.
    cxl_unhit_damp_decay: float = 0.0
    cxl_unhit_window: int = 0
    # ``hit_refill_damp`` > 0: while the marker's starve window is live, an
    # LO arrival landing on the HIT side within ``hit_refill_band`` ticks
    # of that side's touch is dropped. The tape's fill-channel continuation
    # (+2.5 ticks of the k200 drift) needs the consumed side to stay thin
    # so the next aggressor walks a level; refilling the touch before the
    # child fires is exactly what the calibrated sim does. Window is
    # independent of the unhit-side relief window. 0 defaults keep every
    # path bit-identical; the marker gains a ``starve_until`` field.
    hit_refill_damp: float = 0.0
    hit_refill_band: int = 0
    hit_refill_window: int = 0
    # ``hit_flee_frac`` ∈ [0, 1]: for ``hit_flee_window`` events after a
    # fill, EVERY event fires one extra cancel with this probability on a
    # resting HIT-side order within ``hit_flee_band`` levels of that
    # side's touch. The tape's post-exec retreat is a real cancel surge —
    # 5-7x baseline for ~0.5s (cancel_cluster.v1) — and it is what lets
    # an emptied touch reveal a multi-tick gap (touch_follow.v1): when
    # the crown behind the touch stays put the reveal is capped at one
    # tick (sim instant_given_empty 1.03 vs tape 1.88). 0 keeps every
    # path bit-identical; the marker gains a ``flee_until`` field.
    hit_flee_frac: float = 0.0
    hit_flee_band: int = 0
    hit_flee_window: int = 0
    # ``repost_frac`` in [0, 1]: an LO arrival on a side lands at the
    # freshest still-vacant level emptied within ``repost_window``
    # events — the tape's per-price re-post memory (reseed_hazard.v1:
    # 54% of emptied levels re-seed within 500 events, ~75% back at
    # the touch). A bounded per-side vacancy ledger (256 entries)
    # records every vacated level; candidates are scanned
    # freshest-first and skipped when stale, refilled, or illegal
    # (crossing). A reposted arrival bypasses the cooldown/starve
    # suppression gates — it IS the refill those knobs suppress. 0
    # keeps every path bit-identical.
    repost_frac: float = 0.0
    repost_window: int = 0
    # ``repost_band`` > 0 restricts repost candidates to vacancies within this
    # many ticks of the same-side best (the tape's re-seeds concentrate at the
    # touch); 0 = any vacated level.
    repost_band: int = 0
    # ``repost_cause`` selects which vacancies the ledger offers for
    # re-posting: "any" pools every emptied level; "fill" restricts to
    # levels emptied by a fill — the tape's reseed stat counts only
    # fill-emptied levels, and hit-side cancel surges (hit_flee) flood
    # the ledger with vacancies the measure never sees.
    repost_cause: str = "any"
    # ``repost_depth`` > 1: a repost restores ``repost_depth`` units at the
    # vacancy, not one — the tape's re-seeds bring real size back (median
    # ~90 shares vs ~60-share MO prints), while a unit re-post dies to the
    # very next MO and re-empties, so unit reposts can't lift the measured
    # reseed rate. Rests carry no extra RNG draws. 1 = baseline.
    repost_depth: int = 1
    # ``fill_repost_frac`` in [0, 1]: when a fill empties a level, schedule
    # a re-post of ``repost_depth`` units at that level, due after a
    # geometric-ish delay with mean ``fill_repost_delay`` events — the
    # tape's maker re-quote after being lifted (reseed_hazard.v1: reseed
    # latency p50 ~110 events). Arrival-driven reposts (``repost_frac``)
    # cannot reproduce this — the fill-vacancy pool is nearly empty at
    # arrival times and cancel-vacancies dominate the ledger. A delayed
    # post preserves the emptied-touch reading (the level is empty at
    # the next snapshot) while restoring it within the measure window.
    # 0 disables; no RNG draws while off.
    fill_repost_frac: float = 0.0
    fill_repost_delay: int = 0
    # ``unhit_step_ticks`` > 0: while the marker's step window is live, an
    # LO arrival landing on the UNHIT side is shifted toward the touch by
    # up to ``unhit_step_ticks`` ticks, capped one tick inside the
    # opposite quote (never marketable). This is the repricing channel the
    # tape's post-fill LO flow carries (+2.81 ticks of k200): the unhit
    # side steps up after a fill instead of sitting at its pre-fill
    # distance. Deterministic reroute — no extra RNG draws; 0 defaults
    # keep every path bit-identical; the marker gains a ``step_until``
    # field.
    unhit_step_ticks: int = 0
    unhit_step_window: int = 0
    # ``unhit_imp_frac`` > 0: while the marker's improve window is live, an
    # LO arrival on the UNHIT side is rerouted with probability
    # ``unhit_imp_frac`` to one tick inside the spread (bb+1 / ba-1,
    # falling back to the touch when the spread is one tick). This is the
    # tape's post-fill chase: the unhit side keeps repricing the touch
    # WITH the drift for ~200 events (continuation_attr windows show
    # unhit-lo +8.0 ticks in (50,200] alone). The reroute draws one extra
    # uniform per live unhit-side arrival only when the knob is on; 0
    # keeps every path bit-identical; the marker gains an ``imp_until``
    # field.
    unhit_imp_frac: float = 0.0
    unhit_imp_window: int = 0
    # ``vac_chase_frac`` > 0: the vacancy-COUPLED chase — an LO arrival on
    # the side opposite a recently-emptied level is rerouted with
    # probability ``vac_chase_frac`` to one tick inside the spread, but
    # ONLY while that emptied level stays vacant and within
    # ``vac_chase_window`` events of its vacancy. Unlike ``unhit_imp``
    # (fixed post-fill window) the chase dies the moment the hit side
    # repairs — it cannot fight the refill mechanism. Draws one uniform
    # per eligible arrival only when the knob is on; 0 keeps every path
    # bit-identical.
    vac_chase_frac: float = 0.0
    vac_chase_window: int = 0
    # ``chase_release`` ∈ [0, 1]: a cancel event picks a chase-tagged
    # order (one rerouted by ``_unhit_chase``/``_vac_chase``) with this
    # probability — the re-quote churn that RELEASES the touch press so
    # the chase reprices without pinning the spread. Draws one uniform
    # per cancel event only when the knob is on and chase orders exist;
    # 0 keeps every path bit-identical.
    chase_release: float = 0.0
    # ``chase_reprice`` ∈ [0, 1]: the share of release picks that RE-SITE
    # the chase order at the current chase level (one tick inside the
    # spread, or the touch when spread == 1) instead of deleting it —
    # the re-quote cycle: chased depth follows the recovering spread
    # instead of pinning the touch or exiting. Draws one extra uniform
    # per release pick only when nonzero; 0 = pure delete.
    chase_reprice: float = 0.0
    # ``mid_dark_frac`` ∈ [0, 1]: share of LO events that rest at the
    # midpoint as HIDDEN pegged depth — dark orders absorb marketable
    # flow at mid without entering the visible book, so they carry flow
    # without pressing the touch. The placement class the wave-23
    # mechanism map diagnosed as missing (iceberg/hidden share ~21% on
    # the tape). A dark peg lapses when the visible mid moves or after
    # ``mid_dark_ttl`` events (re-quoting, not resting). Draws one extra
    # uniform per LO event only when the knob is on; 0 bit-identical.
    mid_dark_frac: float = 0.0
    mid_dark_ttl: int = 0
    # Optional event-size tables ``((size, weight), ...)``. When set, each
    # market-order event consumes ``size`` resting units in one burst
    # (sweeping levels when the touch is thin, so multi-level sweeps
    # emerge) and each limit-order event deposits ``size`` units at its
    # level. ``None`` keeps the unit-size default with zero extra RNG
    # draws — an unset table is bit-identical to the legacy stream.
    mo_size_pmf: tuple[tuple[int, float], ...] | None = None
    lo_size_pmf: tuple[tuple[int, float], ...] | None = None
    # Optional self-exciting event clock. When set, the homogeneous
    # Poisson superposition is replaced by a 3-type multivariate Hawkes
    # process over (limit, market, cancel) whose baselines are the same
    # rates (``2*lam*band``, ``2*mu``, ``theta_cxl*depth``) plus a decaying
    # excitation state — the tape's submit/cancel storms and post-exec
    # cancel retreat become expressible. ``None`` keeps the Poisson clock
    # bit-identical (zero change to the draw stream).
    hawkes: HawkesClockSpec | None = None
    # Optional Markov-modulated baseline scaling (MMPP): the (lo, mo, cxl)
    # base rates are multiplied by the current regime state's scale before
    # the clock draws — produces session-level activity regimes on top of
    # (or instead of) self-excitation. ``None`` keeps the constant-base
    # clock bit-identical.
    rate_regimes: RateRegimeSpec | None = None
    # ``p_buy_drift`` is a per-second linear drift applied to the effective
    # buy probability wherever it is derived (flat config or regime arm):
    # ``p_eff = clip(p_buy + p_buy_drift * t, 0, 1)`` evaluated at each MO
    # side draw — the tape's intraday initiative fade becomes expressible
    # without disturbing regime composition. 0 skips the clip entirely
    # (bit-identical to the legacy side draw).
    p_buy_drift: float = 0.0

    def __post_init__(self) -> None:
        _pos_finite(self.s0, "s0")
        _pos_finite(self.tick, "tick")
        _pos_finite(self.lam, "lam")
        _pos_finite(self.mu, "mu")
        _pos_finite(self.theta_cxl, "theta_cxl")
        _prob(self.p_buy, "p_buy")
        _nonneg_finite(self.density_exponent, "density_exponent")
        _nonneg_finite(self.ref_halflife, "ref_halflife")
        _nonneg_finite(self.ref_fill_gain, "ref_fill_gain")
        if isinstance(self.refill_cooldown, bool) or int(self.refill_cooldown) < 0:
            raise ValueError(
                f"refill_cooldown must be a non-negative int, got {self.refill_cooldown!r}"
            )
        _nonneg_finite(self.lo_tilt_gain, "lo_tilt_gain")
        d = self.lo_tilt_decay
        if isinstance(d, bool) or not math.isfinite(d) or d < 0.0 or d > 1.0:
            raise ValueError(f"lo_tilt_decay must be in [0, 1], got {d!r}")
        for _name in ("hit_narrow_dist", "hit_narrow_window"):
            _v = getattr(self, _name)
            if isinstance(_v, bool) or int(_v) < 0:
                raise ValueError(f"{_name} must be a non-negative int, got {_v!r}")
        _prob(self.place_mode_frac, "place_mode_frac")
        if self.anchor not in ("touch", "ref"):
            raise ValueError(f"anchor must be 'touch' or 'ref', got {self.anchor!r}")
        if isinstance(self.band, bool) or int(self.band) < 1:
            raise ValueError(f"band must be an int >= 1, got {self.band!r}")
        if isinstance(self.init_levels, bool) or int(self.init_levels) < 0:
            raise ValueError(f"init_levels must be an int >= 0, got {self.init_levels!r}")
        if isinstance(self.init_depth, bool) or int(self.init_depth) < 0:
            raise ValueError(f"init_depth must be an int >= 0, got {self.init_depth!r}")
        if int(self.init_levels) > 0 and int(self.init_depth) < 1:
            raise ValueError("init_depth must be >= 1 when init_levels > 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        _check_size_pmf(self.mo_size_pmf, "mo_size_pmf")
        _check_size_pmf(self.lo_size_pmf, "lo_size_pmf")
        if isinstance(self.lo_offset, bool) or int(self.lo_offset) < 0:
            raise ValueError(f"lo_offset must be an int >= 0, got {self.lo_offset!r}")
        _prob(self.iceberg_reload, "iceberg_reload")
        if self.iceberg_reload_mode not in ("per_unit", "residual"):
            raise ValueError(
                "iceberg_reload_mode must be 'per_unit' or 'residual', "
                f"got {self.iceberg_reload_mode!r}"
            )
        if isinstance(self.iceberg_budget, bool) or int(self.iceberg_budget) < 0:
            raise ValueError(f"iceberg_budget must be an int >= 0, got {self.iceberg_budget!r}")
        _nonneg_finite(self.lo_offset_gain, "lo_offset_gain")
        _prob(self.lo_improve_frac, "lo_improve_frac")
        _prob(self.place_join_frac, "place_join_frac")
        _prob(self.crown_stack_frac, "crown_stack_frac")
        if isinstance(self.crown_stack_span, bool) or int(self.crown_stack_span) < 0:
            raise ValueError(f"crown_stack_span must be an int >= 0, got {self.crown_stack_span!r}")
        if isinstance(self.crown_offset, bool) or int(self.crown_offset) < 0:
            raise ValueError(f"crown_offset must be an int >= 0, got {self.crown_offset!r}")
        if isinstance(self.crown_cap, bool) or int(self.crown_cap) < 0:
            raise ValueError(f"crown_cap must be an int >= 0, got {self.crown_cap!r}")
        if isinstance(self.near_level_cap, bool) or int(self.near_level_cap) < 0:
            raise ValueError(f"near_level_cap must be an int >= 0, got {self.near_level_cap!r}")
        if isinstance(self.near_level_span, bool) or int(self.near_level_span) < 0:
            raise ValueError(f"near_level_span must be an int >= 0, got {self.near_level_span!r}")
        _check_size_pmf(self.crown_size_pmf, "crown_size_pmf")
        _prob(self.touch_pull, "touch_pull")
        _prob(self.cxl_touch_bias, "cxl_touch_bias")
        _nonneg_finite(self.cxl_dist_decay, "cxl_dist_decay")
        _prob(self.cxl_requote, "cxl_requote")
        _prob(self.cxl_unhit_relief, "cxl_unhit_relief")
        _prob(self.cxl_unhit_damp, "cxl_unhit_damp")
        if isinstance(self.cxl_unhit_damp_decay, bool) or self.cxl_unhit_damp_decay < 0.0:
            raise ValueError(
                f"cxl_unhit_damp_decay must be >= 0, got {self.cxl_unhit_damp_decay!r}"
            )
        if isinstance(self.cxl_unhit_window, bool) or int(self.cxl_unhit_window) < 0:
            raise ValueError(f"cxl_unhit_window must be an int >= 0, got {self.cxl_unhit_window!r}")
        _prob(self.hit_refill_damp, "hit_refill_damp")
        _prob(self.hit_flee_frac, "hit_flee_frac")
        _prob(self.unhit_imp_frac, "unhit_imp_frac")
        _prob(self.repost_frac, "repost_frac")
        if isinstance(self.repost_window, bool) or int(self.repost_window) < 0:
            raise ValueError(f"repost_window must be an int >= 0, got {self.repost_window!r}")
        if isinstance(self.repost_band, bool) or int(self.repost_band) < 0:
            raise ValueError(f"repost_band must be an int >= 0, got {self.repost_band!r}")
        if self.repost_cause not in ("any", "fill"):
            raise ValueError(f"repost_cause must be 'any' or 'fill', got {self.repost_cause!r}")
        if isinstance(self.repost_depth, bool) or int(self.repost_depth) < 1:
            raise ValueError(f"repost_depth must be an int >= 1, got {self.repost_depth!r}")
        _prob(self.fill_repost_frac, "fill_repost_frac")
        if isinstance(self.fill_repost_delay, bool) or int(self.fill_repost_delay) < 0:
            raise ValueError(
                f"fill_repost_delay must be an int >= 0, got {self.fill_repost_delay!r}"
            )
        _prob(self.vac_chase_frac, "vac_chase_frac")
        _prob(self.chase_release, "chase_release")
        _prob(self.chase_reprice, "chase_reprice")
        _prob(self.mid_dark_frac, "mid_dark_frac")
        for _name in (
            "hit_refill_band",
            "hit_refill_window",
            "hit_flee_band",
            "hit_flee_window",
            "unhit_step_ticks",
            "unhit_step_window",
            "unhit_imp_window",
            "vac_chase_window",
            "mid_dark_ttl",
        ):
            _v = getattr(self, _name)
            if isinstance(_v, bool) or int(_v) < 0:
                raise ValueError(f"{_name} must be a non-negative int, got {_v!r}")
        if self.lo_offset_gain > 0.0 and self.hawkes is None:
            raise ValueError("lo_offset_gain requires a HawkesClockSpec (hawkes=)")
        if self.hawkes is not None and not isinstance(self.hawkes, HawkesClockSpec):
            raise TypeError(f"hawkes must be a HawkesClockSpec, got {self.hawkes!r}")
        if self.rate_regimes is not None and not isinstance(self.rate_regimes, RateRegimeSpec):
            raise TypeError(f"rate_regimes must be a RateRegimeSpec, got {self.rate_regimes!r}")
        if not math.isfinite(float(self.p_buy_drift)):
            raise ValueError(f"p_buy_drift must be finite, got {self.p_buy_drift!r}")


def santa_fe_config(
    *,
    seed: int = 0,
    p_buy: float = 0.5,
    band: int = 5,
    s0: float = 100.0,
    tick: float = 0.01,
) -> ZILobConfig:
    """ZI-LOB config with the Moret & Lillo (2026) AMZN calibration.

    lambda=0.06, mu=0.10, theta_cxl=0.02 per simulated second (arXiv:
    2609.11614 Eq. 4, calibrated from LOBSTER Level-3 data). SYNTHETIC
    research default only — not a claim about any live market.
    """
    return ZILobConfig(
        s0=s0, tick=tick, lam=0.06, mu=0.10, theta_cxl=0.02, p_buy=p_buy, band=band, seed=seed
    )


# ---------------------------------------------------------------------------
# Regime-switching flow (two-state Markov modulation on the MO clock)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeState:
    """One flow regime: MO intensity multiplier and buy probability."""

    name: str
    intensity_mult: float
    p_buy: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("RegimeState.name must be a non-empty string")
        _pos_finite(self.intensity_mult, "intensity_mult")
        _prob(self.p_buy, "p_buy")


class MOFlow(Protocol):
    """MO-clock flow driver consumed by ZILobSimulator.

    ``current()`` returns the effective ``RegimeState`` (intensity
    multiplier + buy probability); ``advance()`` is called once per
    market-order event. Structural — SplitFlow and MarkovRegimeFlow
    both satisfy it.
    """

    def current(self) -> RegimeState: ...

    def advance(self) -> None: ...


class MarkovRegimeFlow:
    """Two-state Markov modulation of MO intensity and direction.

    The chain advances on the **MO clock** (one transition draw per market
    order), matching the regime generation of Moret & Lillo (2026, Sec. 5):
    regimes are indexed on MO events, so physical regime duration scales
    inversely with the MO rate. ``stay_probs[i]`` is the probability the chain
    remains in state ``i`` at the next MO event. Seeded and deterministic.
    """

    def __init__(
        self,
        states: Sequence[RegimeState],
        stay_probs: Sequence[float],
        *,
        seed: int = 0,
    ) -> None:
        if len(states) != 2:
            raise ValueError(f"exactly 2 regime states required, got {len(states)}")
        if len(stay_probs) != 2:
            raise ValueError(f"exactly 2 stay probabilities required, got {len(stay_probs)}")
        for p in stay_probs:
            v = float(p)
            if not math.isfinite(v) or v <= 0.0 or v > 1.0:
                raise ValueError(f"stay probabilities must lie in (0, 1], got {p!r}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        self._states: tuple[RegimeState, RegimeState] = (states[0], states[1])
        self._stay: tuple[float, float] = (float(stay_probs[0]), float(stay_probs[1]))
        self._rng = np.random.default_rng(seed)
        self._state_idx = 0
        self.n_mo = 0
        self.state_mo_counts: list[int] = [0, 0]
        self.transitions: list[tuple[int, int]] = []

    @property
    def state_index(self) -> int:
        return self._state_idx

    def current(self) -> RegimeState:
        return self._states[self._state_idx]

    def advance(self) -> None:
        """One MO-clock tick: count the visit, then draw the transition."""
        self.state_mo_counts[self._state_idx] += 1
        self.n_mo += 1
        if float(self._rng.random()) > self._stay[self._state_idx]:
            self._state_idx = 1 - self._state_idx
            self.transitions.append((self.n_mo, self._state_idx))

    def expected_p_buy(self) -> float:
        """Visit-weighted mean buy probability realized so far (fail-closed)."""
        if self.n_mo == 0:
            raise ValueError("expected_p_buy undefined before any MO event")
        total = sum(self.state_mo_counts[i] * self._states[i].p_buy for i in range(2))
        return float(total / self.n_mo)


class HawkesClock:
    """Ogata-thinning clock for a 3-type mutually-exciting Hawkes process.

    Drives ``ZILobSimulator.step`` when ``config.hawkes`` is set. The
    per-type intensity is ``bases[k] + e[k]`` where ``bases`` are the
    state's current homogeneous rates (the depth-proportional cancel
    baseline is exact: book depth is constant between events) and ``e``
    is the decaying excitation. Between events each intensity is
    non-increasing, so thinning with the start-of-gap intensity as the
    upper bound is exact Ogata sampling. The accepted proposal's uniform
    is reused for the type draw — it is uniform on the accepted range —
    so a zero kernel consumes the same draws as the Poisson clock and is
    bit-identical to it.
    """

    _MAX_PROPOSALS = 100_000

    def __init__(self, spec: HawkesClockSpec, rng: np.random.Generator) -> None:
        self._k = np.asarray(spec.kernel, dtype=np.float64)
        self._rates = np.asarray(spec.rates or (spec.beta,), dtype=np.float64)
        self._w = np.asarray(spec.bank_weights or (1.0,), dtype=np.float64)
        self._rng = rng
        # Excitation state per (target type, decay bank).
        self._e = np.zeros((3, self._rates.size), dtype=np.float64)
        self.n_proposals = 0
        self.n_rejected = 0

    def excitation(self, kind: int) -> float:
        """Current excitation state for ``kind`` summed over decay banks."""
        return float(self._e[int(kind)].sum())

    def _intensity(self, base: Array, decay_row: Array | None) -> Array:
        """``base + e``; ``decay_row`` applies per-bank ``exp(-beta_r s)``."""
        e = self._e if decay_row is None else self._e * decay_row[None, :]
        return base + e.sum(axis=1)

    def step(self, bases: tuple[float, float, float]) -> tuple[float, int]:
        """Draw ``(dt, kind)`` for the next event; kind indexes HAWKES_TYPES."""
        base = np.asarray(bases, dtype=np.float64)
        lam = self._intensity(base, None)
        for _ in range(self._MAX_PROPOSALS):
            total = float(lam.sum())
            if not math.isfinite(total) or total <= 0.0:
                raise RuntimeError(f"degenerate Hawkes intensity {lam!r}")
            dt = float(self._rng.exponential(1.0 / total))
            decay_row = np.exp(-self._rates * dt)
            lam_s = self._intensity(base, decay_row)
            u = float(self._rng.random()) * total
            self.n_proposals += 1
            if u <= float(lam_s.sum()):
                kind = int(np.searchsorted(np.cumsum(lam_s), u, side="right"))
                if kind > 2:  # pragma: no cover - u <= sum(lam_s) by acceptance
                    kind = 2
                # Advance excitation to the event, then deposit the kernel
                # jump across decay banks by share weight.
                self._e *= decay_row[None, :]
                self._e += self._k[kind][:, None] * self._w[None, :]
                return dt, kind
            self.n_rejected += 1
            lam = lam_s
        raise RuntimeError(f"Hawkes thinning exceeded {self._MAX_PROPOSALS} proposals")


class ScenarioRegimeFlow:
    """Deterministic exogenous regime plan replayed on the MO clock.

    The scenario ``ξ`` of Moret & Lillo (2026, Sec. 9, Algorithm C): a stored
    schedule of ``(leg, length_in_MO_events)`` pairs rather than a stochastic
    Markov chain. ``p_buy(m) = legs[k].p_buy`` whenever
    ``C[k-1] <= m < C[k]`` where ``C`` is the cumulative MO-count boundary;
    once the plan is exhausted the final leg's parameters persist (plans are
    sized at generation time to cover the episode's MO horizon, so exhaustion
    is a bounded edge case, not a silent wrap). Same consumption contract as
    :class:`MarkovRegimeFlow` — ``current()`` / ``advance()`` on the MO clock.
    Fully deterministic: no RNG is drawn after construction.
    """

    def __init__(self, legs: Sequence[tuple[RegimeState, int]]) -> None:
        if not legs:
            raise ValueError("ScenarioRegimeFlow requires at least one leg")
        self._legs: tuple[tuple[RegimeState, int], ...] = ()
        for i, item in enumerate(legs):
            st, n = item
            if not isinstance(st, RegimeState):
                raise TypeError(f"legs[{i}][0] must be a RegimeState")
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"legs[{i}][1] must be an int >= 1, got {n!r}")
            self._legs += ((st, int(n)),)
        self._bounds: tuple[int, ...] = tuple(
            sum(n for _, n in self._legs[: k + 1]) for k in range(len(self._legs))
        )
        self._leg_idx = 0
        self.n_mo = 0
        self.state_mo_counts: list[int] = [0] * len(self._legs)
        self.transitions: list[tuple[int, int]] = []

    @property
    def n_legs(self) -> int:
        return len(self._legs)

    @property
    def total_mo(self) -> int:
        """MO horizon the plan was sized for (sum of leg lengths)."""
        return self._bounds[-1]

    @property
    def state_index(self) -> int:
        """Index of the active leg (saturates at the last leg when exhausted)."""
        return self._leg_idx

    def current(self) -> RegimeState:
        return self._legs[self._leg_idx][0]

    def advance(self) -> None:
        """One MO-clock tick: count the visit, then move past any boundary."""
        self.state_mo_counts[self._leg_idx] += 1
        self.n_mo += 1
        while self._leg_idx < len(self._legs) - 1 and self.n_mo >= self._bounds[self._leg_idx]:
            self._leg_idx += 1
            self.transitions.append((self.n_mo, self._leg_idx))

    def expected_p_buy(self) -> float:
        """Visit-weighted mean buy probability realized so far (fail-closed)."""
        if self.n_mo == 0:
            raise ValueError("expected_p_buy undefined before any MO event")
        total = sum(
            self.state_mo_counts[k] * self._legs[k][0].p_buy for k in range(len(self._legs))
        )
        return float(total / self.n_mo)


class AdversarialFlow:
    """Learned-adversary regime flow — the adversary picks each leg's
    ``p_buy`` at the boundary, in feedback with the defender's state.

    The Glielmo-style semi-MDP adversary of Moret & Lillo (2026): rather
    than replaying a stored plan, the flow *extends itself* at regime
    boundaries. When a leg's MO budget is exhausted, ``advance()`` marks a
    boundary pending; the next ``current()`` call resolves it by invoking
    ``picker(obs)`` — lazily, so the observation reflects the fills that the
    triggering market order just produced (``advance`` runs inside
    ``sim.step()`` before the session's fill handler). Leg durations come
    from ``duration_sampler(rng)`` — the adversary controls direction, not
    duration, matching the paper's Pareto leg lengths.

    ``note_inventory(inventory)`` is the feedback channel: the session
    calls it after draining fills so the picker sees the defender's raw
    inventory at boundary time (normalisation is the picker's job — the
    flow stays policy-agnostic). ``boundary_log`` records every resolution
    as ``(n_mo, inventory, prev_p_buy, chosen_p_buy)`` — the adversary's
    transition tuples plus enough context to audit what was chosen and why.

    Consumption contract identical to the other flows: ``current()`` /
    ``advance()`` on the MO clock, ``expected_p_buy()`` fail-closed before
    the first MO. Deterministic given ``seed`` and the inventory feedback —
    ``rng`` only draws leg durations.
    """

    def __init__(
        self,
        *,
        picker: Callable[[float, float], float],
        duration_sampler: Callable[[np.random.Generator], int],
        seed: int,
        first_p_buy: float = 0.5,
        intensity_mult: float = 1.0,
        max_legs: int = 10_000,
    ) -> None:
        if not callable(picker):
            raise TypeError("picker must be callable")
        if not callable(duration_sampler):
            raise TypeError("duration_sampler must be callable")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        _prob(first_p_buy, "first_p_buy")
        _pos_finite(intensity_mult, "intensity_mult")
        if isinstance(max_legs, bool) or int(max_legs) < 1:
            raise ValueError(f"max_legs must be an int >= 1, got {max_legs!r}")
        self._picker = picker
        self._duration_sampler = duration_sampler
        self._rng = np.random.default_rng(seed)
        self._max_legs = int(max_legs)
        self._intensity_mult = float(intensity_mult)
        self._inv_norm = 0.0
        self._pending = False
        first = float(duration_sampler(self._rng))
        if not math.isfinite(first) or first < 1:
            raise ValueError(f"duration_sampler returned {first!r}, expected an int >= 1")
        self._legs: list[tuple[RegimeState, int]] = [
            (RegimeState("adversary:leg0", intensity_mult, first_p_buy), int(first))
        ]
        self._bound = int(first)
        self._leg_idx = 0
        self.n_mo = 0
        self.state_mo_counts: list[int] = [0]
        self.transitions: list[tuple[int, int]] = []
        self.boundary_log: list[tuple[int, float, float, float]] = []

    def note_inventory(self, inventory: float) -> None:
        """Record the defender's raw inventory for the next pick."""
        v = float(inventory)
        if not math.isfinite(v):
            raise ValueError(f"inventory must be finite, got {inventory!r}")
        self._inv_norm = v

    def _resolve_leg(self) -> None:
        if not self._pending:
            return
        self._pending = False
        if len(self._legs) >= self._max_legs:
            # fail closed: unbounded self-extension must not run forever
            raise RuntimeError(
                f"AdversarialFlow exceeded max_legs={self._max_legs} — "
                "the episode horizon should exhaust before this many legs"
            )
        prev_p_buy = self._legs[self._leg_idx][0].p_buy
        p_buy = float(self._picker(self._inv_norm, prev_p_buy))
        if not math.isfinite(p_buy) or not (0.0 < p_buy < 1.0):
            raise RuntimeError(f"adversary picker returned p_buy={p_buy!r} — must lie in (0,1)")
        n = int(self._duration_sampler(self._rng))
        if n < 1:
            raise RuntimeError(f"duration_sampler returned {n!r}, expected >= 1")
        self._legs.append(
            (RegimeState(f"adversary:leg{len(self._legs)}", self._intensity_mult, p_buy), n)
        )
        self.boundary_log.append((self.n_mo, self._inv_norm, prev_p_buy, p_buy))
        self.transitions.append((self.n_mo, len(self._legs) - 1))
        self.state_mo_counts.append(0)
        self._bound += n
        self._leg_idx += 1

    @property
    def n_legs(self) -> int:
        return len(self._legs)

    @property
    def state_index(self) -> int:
        return self._leg_idx

    def current(self) -> RegimeState:
        self._resolve_leg()
        return self._legs[self._leg_idx][0]

    def advance(self) -> None:
        """One MO-clock tick; a boundary is resolved lazily on next current()."""
        self.state_mo_counts[self._leg_idx] += 1
        self.n_mo += 1
        if self.n_mo >= self._bound:
            self._pending = True

    def expected_p_buy(self) -> float:
        """Visit-weighted mean buy probability realized so far (fail-closed)."""
        if self.n_mo == 0:
            raise ValueError("expected_p_buy undefined before any MO event")
        total = sum(
            self.state_mo_counts[k] * self._legs[k][0].p_buy for k in range(len(self._legs))
        )
        return float(total / self.n_mo)


# ---------------------------------------------------------------------------
# Events / records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeEvent:
    """One unit-lot execution: an aggressor MO consumed a resting maker order."""

    t: float
    aggressor: Side
    price: float
    level: int
    qty: int
    maker_order_id: int
    maker_side: Side
    maker_tag: str
    maker_t_submit: float
    maker_queue_ahead_at_submit: int
    maker_placement_class: PlacementClass


@dataclass(frozen=True)
class BookSample:
    """Point-in-time book snapshot for phase diagnostics."""

    t: float
    mid: float | None
    spread_ticks: int | None
    bid_depth: int
    ask_depth: int


@dataclass
class _Order:
    order_id: int
    side: Side
    level: int
    tag: str
    t_submit: float
    queue_ahead: int
    placement_class: PlacementClass


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------


class ZILobSimulator:
    """Event-driven zero-intelligence LOB with price-time priority.

    State-dependent Poisson clocks (Cont-Stoikov-Talreja 2010 queueing; Santa
    Fe parameterization per Moret & Lillo 2026): at each step the total rate
    is ``lo_rate + 2*mu_eff + theta_cxl*depth``; the next event time is
    exponential in that rate and the event type is drawn proportionally. All
    randomness comes from one seeded ``numpy`` Generator, so identical
    ``(config.seed, flow seed)`` pairs give bit-identical event paths.

    Fail-closed: off-grid prices, marketable limit submits, unknown order ids
    (on cancel-by-id paths used by sessions), and non-finite horizons raise.
    """

    def __init__(self, config: ZILobConfig, flow: MOFlow | None = None) -> None:
        if not isinstance(config, ZILobConfig):
            raise TypeError("config must be a ZILobConfig")
        self._cfg = config
        self._flow = flow
        self._rng = np.random.default_rng(config.seed)
        self._t = 0.0
        self._bids: dict[int, deque[int]] = {}
        self._asks: dict[int, deque[int]] = {}
        self._orders: dict[int, _Order] = {}
        self._next_id = 0
        self.trades: list[TradeEvent] = []
        self.n_events = 0
        self.n_lo_arrivals = 0
        self.n_mo_arrivals = 0
        self.n_mo_noop = 0
        self.n_fills = 0
        self.n_cancellations = 0
        self.n_submitted = 0
        self.n_lo_improve = 0
        self.n_lo_join = 0
        self.n_hidden_fills = 0
        self.n_lo_crown = 0
        self.n_touch_pulls = 0
        self.n_hit_flees = 0
        self.n_cxl_touch = 0
        self.n_lo_capped = 0
        self.n_lo_reposts = 0
        self.n_requotes = 0
        # Cancel-distance histogram: bucket d counts cancels d ticks
        # from that side's touch; index 20 collects the tail.
        self.cxl_dist = [0] * 21
        # Age (seconds) of each canceled order at removal.
        self.cxl_ages: list[float] = []
        self._n_orders_created = 0
        # Per-placement-class fate tallies: every resting order is classified
        # at submit (join/improve/deep vs the own-side touch) and its
        # resolution (fill or cancel) is counted against that class.
        self._fate_placed: dict[PlacementClass, int] = {c: 0 for c in PLACEMENT_CLASSES}
        self._fate_fills: dict[PlacementClass, int] = {c: 0 for c in PLACEMENT_CLASSES}
        self._fate_cancels: dict[PlacementClass, int] = {c: 0 for c in PLACEMENT_CLASSES}
        # Per-order resolution log: (placement_class, queue_ahead_at_submit,
        # outcome) for every fill and cancel — the bench derives class-level
        # and queue-conditional fill rates from it.
        self.fate_log: list[tuple[PlacementClass, int, str]] = []
        # Reference level for LO bands when the opposite side is empty
        # (keeps book recovery possible; falls back to the seeded mid level).
        self._ref_level = 0
        # Slow absolute-price reference (EMA of the mid level) for anchor="ref".
        # Frozen at the seed mid (level 0) when ref_halflife == 0.
        self._ref_ema = 0.0
        # Limit-order placement distance law over d ∈ [1, band]: the
        # monotone P(d) ∝ d**density_exponent family, or — when
        # place_mode_frac > 0 — a shifted Binomial hump d ~ 1 +
        # Bin(band - 1, frac) built by the p_{k+1} recurrence.
        band = int(config.band)
        if config.place_mode_frac > 0.0:
            frac = float(config.place_mode_frac)
            n_bin = band - 1
            weights = np.zeros(band, dtype=np.float64)
            if frac >= 1.0:
                weights[-1] = 1.0
            else:
                weights[0] = (1.0 - frac) ** n_bin
                for k in range(n_bin):
                    weights[k + 1] = weights[k] * (n_bin - k) / (k + 1) * frac / (1.0 - frac)
        else:
            weights = np.arange(1, band + 1, dtype=np.float64) ** float(config.density_exponent)
        self._dist_cdf = np.cumsum(weights / weights.sum())
        self._dist_cdf[-1] = 1.0
        # Event-size tables (None → unit-size, zero extra RNG draws).
        self._mo_size_cdf = self._size_cdf(config.mo_size_pmf)
        self._lo_size_cdf = self._size_cdf(config.lo_size_pmf)
        self._crown_size_cdf = self._size_cdf(config.crown_size_pmf)
        self._hawkes = HawkesClock(config.hawkes, self._rng) if config.hawkes is not None else None
        self._rate_flow = (
            RateRegimeFlow(config.rate_regimes, self._rng)
            if config.rate_regimes is not None
            else None
        )
        self.n_mo_units = 0
        self.n_lo_units = 0
        self.n_dark_placed = 0
        self.n_dark_fills = 0
        self.n_dark_lapses = 0
        # Vacancy memory: (side, level) -> event index the level emptied.
        # Populated only when ``refill_cooldown`` or ``vac_chase_window`` > 0.
        self._vacancy: dict[tuple[str, int], int] = {}
        # Most recently emptied (side -> (level, event)) — the re-post
        # memory consumed by repost_frac.
        self._last_empty: dict[str, dict[int, tuple[int, str]]] = {"buy": {}, "sell": {}}
        # Scheduled fill-triggered re-posts: (due_event, side, level).
        self._fill_repost_q: list[tuple[int, Side, int]] = []
        # Remaining hidden refills per (side, level); used only when
        # ``iceberg_budget`` > 0.
        self._ice_budget: dict[tuple[Side, int], int] = {}
        self._chase_oids: set[int] = set()
        # Hidden midpoint-pegged depth: side -> deque of (order, expiry
        # event index). Never enters the visible book; pegs lapse when
        # the visible mid moves or TTL passes (re-quote, not rest).
        self._dark: dict[Side, deque[tuple[_Order, int]]] = {
            "buy": deque(),
            "sell": deque(),
        }
        # Post-fill accommodation state; pinned at 0 when lo_tilt_gain == 0.
        self._tilt = 0.0
        # Post-fill marker: (hit_side, narrow_deadline, relief_deadline)
        # of the last fill; a deadline of n_events means that channel off.
        # (hit_side, narrow_until, relief_until, fill_event) — the fill's
        # n_events lets damp_decay compute elapsed time.
        self._hit_retreat: tuple[str, int, int, int, int, int, int, int] | None = None
        self.n_lo_suppressed = 0
        for k in range(1, config.init_levels + 1):
            for _ in range(config.init_depth):
                self._rest("buy", -k, "zi_seed")
                self._rest("sell", k, "zi_seed")

    @staticmethod
    def _size_cdf(
        pmf: tuple[tuple[int, float], ...] | None,
    ) -> tuple[tuple[int, ...], Array] | None:
        if pmf is None:
            return None
        sizes = tuple(int(s) for s, _ in pmf)
        w = np.asarray([w for _, w in pmf], dtype=np.float64)
        cdf = np.cumsum(w / w.sum())
        cdf[-1] = 1.0
        return sizes, cdf

    def _draw_size(self, table: tuple[tuple[int, ...], Array] | None) -> int:
        """Draw an event size; 1 with no RNG draw when the table is unset."""
        if table is None:
            return 1
        sizes, cdf = table
        return sizes[int(np.searchsorted(cdf, float(self._rng.random()), side="left"))]

    # -- grid helpers -------------------------------------------------------

    @property
    def cfg(self) -> ZILobConfig:
        return self._cfg

    @property
    def t(self) -> float:
        return self._t

    def level_to_price(self, level: int) -> float:
        return self._cfg.s0 + int(level) * self._cfg.tick

    def price_to_level(self, price: float) -> int:
        p = float(price)
        if not math.isfinite(p) or p <= 0.0:
            raise ValueError(f"price must be positive and finite, got {price!r}")
        raw = (p - self._cfg.s0) / self._cfg.tick
        level = int(round(raw))
        if abs(raw - level) > 1e-6:
            raise ValueError(f"price {p!r} is off the tick grid (tick={self._cfg.tick})")
        return level

    # -- book state ---------------------------------------------------------

    @property
    def best_bid_level(self) -> int | None:
        return max(self._bids) if self._bids else None

    @property
    def best_ask_level(self) -> int | None:
        return min(self._asks) if self._asks else None

    @property
    def best_bid(self) -> float | None:
        lvl = self.best_bid_level
        return self.level_to_price(lvl) if lvl is not None else None

    @property
    def best_ask(self) -> float | None:
        lvl = self.best_ask_level
        return self.level_to_price(lvl) if lvl is not None else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid_level, self.best_ask_level
        if bb is None or ba is None:
            return None
        return self.level_to_price(0) + 0.5 * (bb + ba) * self._cfg.tick

    @property
    def spread_ticks(self) -> int | None:
        bb, ba = self.best_bid_level, self.best_ask_level
        if bb is None or ba is None:
            return None
        return ba - bb

    @staticmethod
    def _side_depth(book: dict[int, deque[int]]) -> int:
        return sum(len(dq) for dq in book.values())

    @property
    def bid_depth(self) -> int:
        return self._side_depth(self._bids)

    @property
    def ask_depth(self) -> int:
        return self._side_depth(self._asks)

    @property
    def total_depth(self) -> int:
        return self.bid_depth + self.ask_depth

    def depth_at(self, side: Side, level: int) -> int:
        _check_side(side)
        book = self._bids if side == "buy" else self._asks
        return len(book.get(int(level), ()))

    def queue_position(self, order_id: int) -> int | None:
        """Number of unit orders ahead of ``order_id`` at its level (FIFO).

        ``None`` when the order is unknown (already filled or canceled).
        """
        order = self._orders.get(int(order_id))
        if order is None:
            return None
        dq = (self._bids if order.side == "buy" else self._asks).get(order.level)
        if dq is None:
            return None
        for i, oid in enumerate(dq):
            if oid == order.order_id:
                return i
        return None

    def order_alive(self, order_id: int) -> bool:
        return int(order_id) in self._orders

    def queue_ahead_at_submit(self, order_id: int) -> int | None:
        order = self._orders.get(int(order_id))
        return order.queue_ahead if order is not None else None

    def sample(self) -> BookSample:
        return BookSample(
            t=self._t,
            mid=self.mid,
            spread_ticks=self.spread_ticks,
            bid_depth=self.bid_depth,
            ask_depth=self.ask_depth,
        )

    def event_counts(self) -> dict[str, int]:
        """Conservation accounting (unit orders): created = fills + cxl + resting."""
        return {
            "n_events": self.n_events,
            "n_lo_arrivals": self.n_lo_arrivals,
            "n_mo_arrivals": self.n_mo_arrivals,
            "n_mo_noop": self.n_mo_noop,
            "n_fills": self.n_fills,
            "n_cancellations": self.n_cancellations,
            "n_submitted": self.n_submitted,
            "n_orders_created": self._n_orders_created,
            "resting": self.total_depth,
            "n_mo_units": self.n_mo_units,
            "n_lo_units": self.n_lo_units,
            "n_hawkes_proposals": self._hawkes.n_proposals if self._hawkes else 0,
            "n_hawkes_rejected": self._hawkes.n_rejected if self._hawkes else 0,
            "n_regime_transitions": (
                self._rate_flow.n_transitions if self._rate_flow is not None else 0
            ),
            "n_lo_improve": self.n_lo_improve,
            "n_lo_join": self.n_lo_join,
            "n_lo_crown": self.n_lo_crown,
            "n_hidden_fills": self.n_hidden_fills,
            "n_touch_pulls": self.n_touch_pulls,
            "n_hit_flees": self.n_hit_flees,
            "n_cxl_touch": self.n_cxl_touch,
            "n_lo_capped": self.n_lo_capped,
            "n_lo_reposts": self.n_lo_reposts,
            "placed_join": self._fate_placed["join"],
            "placed_improve": self._fate_placed["improve"],
            "placed_deep": self._fate_placed["deep"],
            "fills_join": self._fate_fills["join"],
            "fills_improve": self._fate_fills["improve"],
            "fills_deep": self._fate_fills["deep"],
            "cancels_join": self._fate_cancels["join"],
            "cancels_improve": self._fate_cancels["improve"],
            "cancels_deep": self._fate_cancels["deep"],
            "n_requotes": self.n_requotes,
            "n_dark_placed": self.n_dark_placed,
            "n_dark_fills": self.n_dark_fills,
            "n_dark_lapses": self.n_dark_lapses,
        }

    def fate_by_class(self) -> dict[PlacementClass, dict[str, int]]:
        """Per-placement-class fate accounting; placed = fills + cxl + resting."""
        out: dict[PlacementClass, dict[str, int]] = {}
        for c in PLACEMENT_CLASSES:
            placed = self._fate_placed[c]
            fills = self._fate_fills[c]
            cancels = self._fate_cancels[c]
            out[c] = {
                "placed": placed,
                "fills": fills,
                "cancels": cancels,
                "resting": placed - fills - cancels,
            }
        return out

    # -- order lifecycle ----------------------------------------------------

    def _new_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def _placement_class(self, side: Side, level: int) -> PlacementClass:
        """Classify a resting placement against the own-side touch.

        ``join`` at the own best (or when that side is empty — the order forms
        the touch), ``improve`` strictly inside the open spread (a new own
        best), ``deep`` behind the own best.
        """
        best = self.best_bid_level if side == "buy" else self.best_ask_level
        if best is None or int(level) == best:
            return "join"
        if side == "buy":
            return "improve" if level > best else "deep"
        return "improve" if level < best else "deep"

    def _rest(self, side: Side, level: int, tag: str) -> int:
        # Classify before the level key exists in the book — an improving
        # order must be measured against the pre-placement touch.
        cls = self._placement_class(side, int(level))
        book = self._bids if side == "buy" else self._asks
        dq = book.setdefault(int(level), deque())
        oid = self._new_id()
        order = _Order(
            order_id=oid,
            side=side,
            level=int(level),
            tag=tag,
            t_submit=self._t,
            queue_ahead=len(dq),
            placement_class=cls,
        )
        dq.append(oid)
        self._orders[oid] = order
        if tag == "chase":
            self._chase_oids.add(oid)
        self._n_orders_created += 1
        self._fate_placed[order.placement_class] += 1
        return oid

    def submit_limit_order(self, side: Side, price: float, tag: str = "zi") -> int:
        """Rest a unit limit order on the grid; fail-closed if marketable."""
        s = _check_side(side)
        level = self.price_to_level(price)
        if s == "buy":
            ba = self.best_ask_level
            if ba is not None and level >= ba:
                raise ValueError(
                    "marketable limit buys are not supported in the ZI simulator; "
                    "use inject_market_order for aggressive flow"
                )
        else:
            bb = self.best_bid_level
            if bb is not None and level <= bb:
                raise ValueError(
                    "marketable limit sells are not supported in the ZI simulator; "
                    "use inject_market_order for aggressive flow"
                )
        oid = self._rest(s, level, tag)
        self.n_submitted += 1
        return oid

    def cancel_order(self, order_id: int) -> bool:
        """Cancel a resting order by id. Returns False when already gone."""
        order = self._orders.pop(int(order_id), None)
        if order is None:
            return False
        book = self._bids if order.side == "buy" else self._asks
        dq = book.get(order.level)
        if dq is None:  # pragma: no cover - registry/book invariant
            raise RuntimeError("order registry and book diverged")
        dq.remove(order.order_id)
        if not dq:
            del book[order.level]
            self._level_vacated(order.side, order.level)
        self.n_cancellations += 1
        self._fate_cancels[order.placement_class] += 1
        self.fate_log.append((order.placement_class, order.queue_ahead, "cancel"))
        return True

    def _maybe_requote(self, order: _Order) -> None:
        """Cancel+replace half of the biased path: re-rest the removed
        order at the same level with a fresh submit time (back of the
        level's queue). Only active when ``cxl_requote > 0``."""
        if (
            self._cfg.cxl_requote > 0.0
            and self._rng.random() < self._cfg.cxl_requote
            and not self._touch_capped(order.side, order.level)
        ):
            self._rest(order.side, order.level, "requote")
            self.n_requotes += 1

    def _level_vacated(self, side: Side, level: int, cause: str = "cancel") -> None:
        """Record that ``level`` on ``side`` just emptied (sticky vacancy).

        ``cause`` is "fill" when a market order emptied the level, else
        "cancel" — ``repost_cause`` filters the re-post pool on it.
        """
        if self._cfg.repost_frac > 0.0:
            vacs = self._last_empty[side]
            vacs[int(level)] = (self.n_events, cause)
            if len(vacs) > 256:
                cutoff = self.n_events - max(self._cfg.repost_window, 1)
                old_lv = [lv for lv, ev0 in vacs.items() if ev0[0] < cutoff]
                for lv in old_lv:
                    del vacs[lv]
                if len(vacs) > 256:
                    oldest = min(vacs, key=lambda lv: vacs[lv][0])
                    del vacs[oldest]
        if (
            cause == "fill"
            and self._cfg.fill_repost_frac > 0.0
            and (self._rng.random() < self._cfg.fill_repost_frac)
        ):
            mean = max(self._cfg.fill_repost_delay, 1)
            due = self.n_events + max(1, int(round(float(self._rng.exponential(mean)))))
            self._fill_repost_q.append((due, side, int(level)))
        horizon = max(self._cfg.refill_cooldown, self._cfg.vac_chase_window)
        if horizon <= 0:
            return
        self._vacancy[(side, int(level))] = self.n_events
        stale = [k for k, t0 in self._vacancy.items() if self.n_events - t0 > horizon]
        for k in stale:
            del self._vacancy[k]

    def _is_cooled(self, side: Side, level: int) -> bool:
        """True when (side, level) emptied within ``refill_cooldown`` events."""
        t0 = self._vacancy.get((side, int(level)))
        return t0 is not None and self.n_events - t0 < self._cfg.refill_cooldown

    def _hit_starved(self, side: Side, level: int) -> bool:
        """True when a hit-side arrival inside the starve window is dropped.

        Draws the survival uniform only when the damp knob is on, the
        marker's starve window is live, and the placement lands on the
        hit side within ``hit_refill_band`` ticks of that side's touch —
        at damp 0 (or an inactive marker) the RNG stream is untouched.
        """
        damp = self._cfg.hit_refill_damp
        if damp <= 0.0 or self._hit_retreat is None:
            return False
        hit_side, _n_u, _r_u, _fe, s_until, _t_u, _i_u, _f_u = self._hit_retreat
        if side != hit_side or self.n_events >= s_until:
            return False
        book = self._asks if side == "sell" else self._bids
        if not book:
            return False
        touch = min(book) if side == "sell" else max(book)
        if abs(int(level) - touch) > self._cfg.hit_refill_band:
            return False
        return float(self._rng.random()) < damp

    def _drain_fill_reposts(self) -> None:
        """Fire due fill-triggered re-posts (``fill_repost_frac``).

        A due repost rests ``repost_depth`` units at the emptied level
        only while the level is still absent and still legal (not
        marketable) — a refilled or walked-past vacancy is dropped.
        """
        keep: list[tuple[int, Side, int]] = []
        for due, side, level in self._fill_repost_q:
            if due > self.n_events:
                keep.append((due, side, level))
                continue
            book = self._bids if side == "buy" else self._asks
            opp = self.best_ask_level if side == "buy" else self.best_bid_level
            if level in book:
                continue  # natural refill already reseeded it
            if opp is not None and (level >= opp if side == "buy" else level <= opp):
                continue  # the price grid walked past the vacancy
            self.n_lo_reposts += 1
            for _ in range(self._cfg.repost_depth):
                self._rest(side, level, "repost")
        self._fill_repost_q = keep

    def _repost_level(self, side: Side) -> int | None:
        """Price-level re-posting memory (``repost_frac``).

        With probability ``repost_frac`` the arriving LO is sited at the
        freshest still-vacant level emptied on ``side`` within
        ``repost_window`` events, preferring vacancies within
        ``repost_band`` ticks of the same-side best when set. Legal
        candidates only — a reposted bid must sit below the current ask
        (and vice versa). The draw consumes RNG only when the knob is on,
        so ``repost_frac == 0`` is bit-identical to the baseline path.
        """
        if self._cfg.repost_frac <= 0.0:
            return None
        if self._rng.random() >= self._cfg.repost_frac:
            return None
        vacs = self._last_empty[side]
        if not vacs:
            return None
        book = self._bids if side == "buy" else self._asks
        opp = self.best_ask_level if side == "buy" else self.best_bid_level
        now = self.n_events
        own = self.best_bid_level if side == "buy" else self.best_ask_level
        band = self._cfg.repost_band
        cause_filter = self._cfg.repost_cause
        for cand_l, (ev0, cause) in sorted(vacs.items(), key=lambda kv: kv[1][0], reverse=True):
            if now - ev0 > self._cfg.repost_window:
                break  # sorted freshest-first; rest are staler
            if cause_filter == "fill" and cause != "fill":
                continue
            if cand_l in book:
                continue
            if opp is not None and (cand_l >= opp if side == "buy" else cand_l <= opp):
                continue
            if (
                band > 0
                and own is not None
                and (cand_l < own - band if side == "buy" else cand_l > own + band)
            ):
                continue
            self.n_lo_reposts += 1
            for _ in range(self._cfg.repost_depth - 1):
                self._rest(side, cand_l, "repost")
            return cand_l
        return None

    def _step_unhit(self, side: Side, level: int) -> int:
        """Shift an unhit-side arrival toward the touch while the marker's
        step window is live. Buys move up toward (at most) one tick below
        the ask; sells move down toward one tick above the bid — never
        marketable, so the shifted order still rests. No RNG draws."""
        step = self._cfg.unhit_step_ticks
        if step <= 0 or self._hit_retreat is None:
            return level
        hit_side, _n_u, _r_u, _fe, _s_u, t_until, _i_u, _f_u = self._hit_retreat
        if side == hit_side or self.n_events >= t_until:
            return level
        if side == "buy":
            ba = self.best_ask_level
            return level if ba is None else min(level + step, ba - 1)
        bb = self.best_bid_level
        return level if bb is None else max(level - step, bb + 1)

    def _hit_flee(self) -> None:
        """Post-fill cancel surge on the HIT side's near-touch depth.

        While the marker's flee window is live, each event fires one
        extra cancel with probability ``hit_flee_frac`` on a resting
        hit-side order within ``hit_flee_band`` levels of that side's
        touch — the tape's measured post-exec retreat (5-7x baseline
        cancel rate for ~0.5s). Fleeing near-touch depth is what leaves
        an emptied touch revealing a multi-tick gap instead of an
        adjacent successor. Draws its trigger uniform only when the knob
        is on and the window is live; the pick uniform only when the
        band is non-empty — at 0 the RNG stream is untouched."""
        frac = self._cfg.hit_flee_frac
        if frac <= 0.0 or self._hit_retreat is None:
            return
        hit_side, _n_u, _r_u, _fe, _s_u, _t_u, _i_u, f_until = self._hit_retreat
        if self.n_events >= f_until:
            return
        book = self._asks if hit_side == "sell" else self._bids
        if not book:
            return
        if float(self._rng.random()) >= frac:
            return
        touch = min(book) if book is self._asks else max(book)
        band = self._cfg.hit_flee_band
        cands: list[tuple[int, int]] = []
        for lvl, dq in book.items():
            if abs(lvl - touch) <= band:
                for idx in range(len(dq)):
                    cands.append((lvl, idx))
        if not cands:  # pragma: no cover - banded zone nonempty when book nonempty
            return
        lvl, idx = cands[int(self._rng.integers(len(cands)))]
        order = self._remove_resting_at(book, lvl, idx)
        self.cxl_ages.append(self.t - order.t_submit)
        self.n_cancellations += 1
        self._fate_cancels[order.placement_class] += 1
        self.fate_log.append((order.placement_class, order.queue_ahead, "cancel"))
        self.n_hit_flees += 1
        d_hit = abs(lvl - touch)
        self.cxl_dist[min(d_hit, 20)] += 1
        if d_hit == 0:
            self.n_cxl_touch += 1

    def _unhit_chase(self, side: Side) -> int | None:
        """Reroute an unhit-side arrival to the chase level (one tick
        inside the spread, touch when spread is 1) with probability
        ``unhit_imp_frac`` while the marker's improve window is live.
        Draws its uniform only for in-window unhit-side arrivals with the
        knob on — at 0 the RNG stream is untouched."""
        frac = self._cfg.unhit_imp_frac
        if frac <= 0.0 or self._hit_retreat is None:
            return None
        hit_side, _n_u, _r_u, _fe, _s_u, _t_u, i_until, _f_u = self._hit_retreat
        if side == hit_side or self.n_events >= i_until:
            return None
        if float(self._rng.random()) >= frac:
            return None
        ba, bb = self.best_ask_level, self.best_bid_level
        if ba is None or bb is None:
            return None
        if ba - bb <= 1:
            return bb if side == "buy" else ba
        return bb + 1 if side == "buy" else ba - 1

    def _vac_chase(self, side: Side) -> int | None:
        """Vacancy-coupled chase: reroute ``side``'s arrival to the chase
        level (one tick inside the spread, touch when the spread is 1)
        with probability ``vac_chase_frac`` while a level emptied on the
        OPPOSITE side is still vacant within ``vac_chase_window`` events.
        Unlike ``_unhit_chase`` the trigger is the vacancy itself — the
        chase expires the moment the hit side repairs. Draws its uniform
        only for eligible arrivals with the knob on — at 0 the RNG
        stream is untouched."""
        frac = self._cfg.vac_chase_frac
        window = self._cfg.vac_chase_window
        if frac <= 0.0 or window <= 0:
            return None
        opp = "sell" if side == "buy" else "buy"
        book = self._asks if opp == "sell" else self._bids
        now = self.n_events
        for (vac_side, vac_level), t0 in self._vacancy.items():
            if vac_side == opp and now - t0 < window and vac_level not in book:
                break
        else:
            return None
        if float(self._rng.random()) >= frac:
            return None
        ba, bb = self.best_ask_level, self.best_bid_level
        if ba is None or bb is None:
            return None
        if ba - bb <= 1:
            return bb if side == "buy" else ba
        return bb + 1 if side == "buy" else ba - 1

    def _remove_resting_at(
        self, book: dict[int, deque[int]], level: int, idx: int, cause: str = "cancel"
    ) -> _Order:
        dq = book[level]
        oid = dq[idx]
        del dq[idx]
        if not dq:
            del book[level]
            self._level_vacated("buy" if book is self._bids else "sell", level, cause)
        order = self._orders.pop(oid)
        self._chase_oids.discard(oid)
        return order

    def _post_fill_markers(self, aggressor: Side) -> None:
        """Arm the post-fill state shared by visible and dark fills."""
        gain = self._cfg.ref_fill_gain
        if gain > 0.0:
            self._ref_ema += (1.0 if aggressor == "buy" else -1.0) * gain
        tg = self._cfg.lo_tilt_gain
        if tg > 0.0:
            sign = 1.0 if aggressor == "buy" else -1.0
            self._tilt = max(-1.0, min(1.0, self._tilt + sign * tg))
        nw = self._cfg.hit_narrow_window if self._cfg.hit_narrow_dist > 0 else 0
        rw = (
            self._cfg.cxl_unhit_window
            if (self._cfg.cxl_unhit_relief > 0.0 or self._cfg.cxl_unhit_damp > 0.0)
            else 0
        )
        sw = self._cfg.hit_refill_window if self._cfg.hit_refill_damp > 0.0 else 0
        tw = self._cfg.unhit_step_window if self._cfg.unhit_step_ticks > 0 else 0
        iw = self._cfg.unhit_imp_window if self._cfg.unhit_imp_frac > 0.0 else 0
        fw = self._cfg.hit_flee_window if self._cfg.hit_flee_frac > 0.0 else 0
        if nw or rw or sw or tw or iw or fw:
            # Hit side = the side the aggressor consumed (resting side).
            hit = "sell" if aggressor == "buy" else "buy"
            self._hit_retreat = (
                hit,
                self.n_events + nw,
                self.n_events + rw,
                self.n_events,
                self.n_events + sw,
                self.n_events + tw,
                self.n_events + iw,
                self.n_events + fw,
            )

    def _touch_capped(self, side: Side, level: int) -> bool:
        """True when an LO arrival is refused: the near-touch band level
        it lands on is already at ``near_level_cap`` units."""
        cap = self._cfg.near_level_cap
        if cap <= 0:
            return False
        book = self._bids if side == "buy" else self._asks
        if not book:
            return False
        best = max(book) if side == "buy" else min(book)
        span = self._cfg.near_level_span
        if side == "buy":
            in_band = best - span <= level <= best
        else:
            in_band = best <= level <= best + span
        dq = book.get(level)
        return in_band and dq is not None and len(dq) >= cap

    def _spend_ice(self, side: Side, level: int) -> bool:
        """Spend one unit of the level's hidden-refill budget."""
        b = self._cfg.iceberg_budget
        if b <= 0:
            return True
        key = (side, level)
        left = self._ice_budget.get(key, b)
        if left <= 0:
            return False
        self._ice_budget[key] = left - 1
        return True

    def _consume_best(self, aggressor: Side) -> TradeEvent | None:
        """Match one unit MO against the opposite best (price-time priority)."""
        # Midpoint dark liquidity matches first: a resting dark peg fills
        # the aggressor at mid before the visible touch is consumed.
        dark_side: Side = "sell" if aggressor == "buy" else "buy"
        dark_dq = self._dark[dark_side]
        while dark_dq:
            order, exp = dark_dq[0]
            bb_d, ba_d = self.best_bid_level, self.best_ask_level
            if (
                bb_d is None
                or ba_d is None
                or order.level != bb_d + ba_d
                or (exp > 0 and self.n_events > exp)
            ):
                dark_dq.popleft()
                self.n_dark_lapses += 1
                continue
            dark_dq.popleft()
            trade = TradeEvent(
                t=self._t,
                aggressor=aggressor,
                price=0.5 * (self.level_to_price(bb_d) + self.level_to_price(ba_d)),
                level=order.level,
                qty=1,
                maker_order_id=order.order_id,
                maker_side=order.side,
                maker_tag=order.tag,
                maker_t_submit=order.t_submit,
                maker_queue_ahead_at_submit=order.queue_ahead,
                maker_placement_class=order.placement_class,
            )
            self.trades.append(trade)
            self.n_fills += 1
            self.n_dark_fills += 1
            self._post_fill_markers(aggressor)
            return trade
        book = self._asks if aggressor == "buy" else self._bids
        if not book:
            self.n_mo_noop += 1
            return None
        level = min(book) if aggressor == "buy" else max(book)
        order = self._remove_resting_at(book, level, 0, "fill")
        if order.tag == "iceberg":
            self.n_hidden_fills += 1
        trade = TradeEvent(
            t=self._t,
            aggressor=aggressor,
            price=self.level_to_price(level),
            level=level,
            qty=1,
            maker_order_id=order.order_id,
            maker_side=order.side,
            maker_tag=order.tag,
            maker_t_submit=order.t_submit,
            maker_queue_ahead_at_submit=order.queue_ahead,
            maker_placement_class=order.placement_class,
        )
        self.trades.append(trade)
        self.n_fills += 1
        self._fate_fills[order.placement_class] += 1
        self.fate_log.append((order.placement_class, order.queue_ahead, "fill"))
        self._post_fill_markers(aggressor)
        # Iceberg reload: the consumed level immediately re-rests one
        # hidden unit with probability ``iceberg_reload`` — the display
        # refill that makes a level absorb more than its visible depth.
        p = self._cfg.iceberg_reload
        if (
            p > 0.0
            and float(self._rng.random()) < p
            and (self._cfg.iceberg_reload_mode == "per_unit" or bool(book.get(level)))
            and self._spend_ice(order.side, level)
        ):
            self._rest(order.side, level, "iceberg")
        # Touch pull: the front order on the hit side is withdrawn with
        # probability ``touch_pull`` — instant quote defense, the kernel's
        # t~0 component.
        if self._cfg.touch_pull > 0.0 and book and float(self._rng.random()) < self._cfg.touch_pull:
            next_level = min(book) if aggressor == "buy" else max(book)
            pulled = self._remove_resting_at(book, next_level, 0)
            self.n_touch_pulls += 1
            self.n_cancellations += 1
            self._fate_cancels[pulled.placement_class] += 1
            self.fate_log.append((pulled.placement_class, pulled.queue_ahead, "cancel"))
        return trade

    def inject_market_order(self, side: Side, qty: int = 1) -> list[TradeEvent]:
        """Exogenous aggressive flow at the current clock (metaorder building block)."""
        s = _check_side(side)
        q = int(qty)
        if q < 1:
            raise ValueError(f"qty must be >= 1, got {qty!r}")
        out: list[TradeEvent] = []
        for _ in range(q):
            trade = self._consume_best(s)
            self.n_mo_arrivals += 1
            if trade is not None:
                out.append(trade)
            if self._flow is not None:
                self._flow.advance()
        return out

    # -- event clock --------------------------------------------------------

    def _flow_params(self) -> tuple[float, float]:
        if self._flow is None:
            mu, p_buy = self._cfg.mu, self._cfg.p_buy
        else:
            st = self._flow.current()
            mu, p_buy = self._cfg.mu * st.intensity_mult, st.p_buy
        if self._cfg.p_buy_drift != 0.0:
            p_buy = min(1.0, max(0.0, p_buy + self._cfg.p_buy_drift * self._t))
        return mu, p_buy

    def _limit_order_event(self) -> None:
        lam, band = self._cfg.lam, self._cfg.band
        bid_rate = lam * band
        ask_rate = lam * band
        total = bid_rate + ask_rate
        u = float(self._rng.random()) * total
        # Placement distance d ∈ [1, band] drawn from P(d) ∝ d**density_exponent
        # via the precomputed CDF (uniform when density_exponent == 0).
        dist = int(np.searchsorted(self._dist_cdf, float(self._rng.random()), side="left")) + 1
        if dist > band:
            dist = band
        ba, bb = self.best_ask_level, self.best_bid_level
        want_buy = u < bid_rate * (1.0 + self._tilt) if self._tilt != 0.0 else u < bid_rate
        # Post-fill narrowing: while the marker is live, placements on the
        # unhit side clamp to near-touch distance (accommodation channel).
        if self._hit_retreat is not None:
            (
                hit_side,
                n_until,
                r_until,
                _fill_ev,
                s_until,
                t_until,
                i_until,
                f_until,
            ) = self._hit_retreat
            if self.n_events >= max(n_until, r_until, s_until, t_until, i_until, f_until):
                self._hit_retreat = None
            elif self.n_events < n_until and (
                (want_buy and hit_side == "sell") or (not want_buy and hit_side == "buy")
            ):
                dist = min(dist, self._cfg.hit_narrow_dist)
        # Midpoint dark peg: the event's units rest hidden at the mid
        # instead of entering the visible book (only when a mid exists —
        # spread >= 2 ticks). Consumes one extra uniform only when on.
        md = self._cfg.mid_dark_frac
        if (
            md > 0.0
            and ba is not None
            and bb is not None
            and ba - bb >= 2
            and float(self._rng.random()) < md
        ):
            k = self._draw_size(self._lo_size_cdf)
            dark_side: Side = "buy" if want_buy else "sell"
            ttl = self._cfg.mid_dark_ttl
            exp = self.n_events + ttl if ttl > 0 else 0
            for _ in range(k):
                order = _Order(
                    order_id=self._new_id(),
                    side=dark_side,
                    level=bb + ba,  # doubled-lattice index of the pegged mid
                    tag="mid_dark",
                    t_submit=self._t,
                    queue_ahead=0,
                    placement_class="improve",
                )
                self._dark[dark_side].append((order, exp))
                self._n_orders_created += 1
                self.n_dark_placed += 1
            self.n_lo_arrivals += 1
            self.n_lo_units += k
            return
        if self._cfg.anchor == "ref":
            # Absolute-space anchoring: LOs deposit around a slow reference level
            # so cumulative liquidity grows with distance from the reference and
            # a metaorder climbing away from it meets a deepening static profile
            # (volume-diffusion / square-root regime). Crossing placements are
            # dropped — that aggressiveness is already in the market-order flow.
            ref = int(round(self._ref_ema))
            k = self._draw_size(self._lo_size_cdf)
            # Mixture head (join/improve/stack): the pick consumes ONE
            # extra uniform and only when a mixture knob is on — both at
            # 0 keeps the ref path bit-identical.
            mix_on = (
                self._cfg.place_join_frac > 0.0
                or self._cfg.lo_improve_frac > 0.0
                or self._cfg.crown_stack_frac > 0.0
            )
            u_mix = self._rng.random() if mix_on else 1.0
            want_join = u_mix < self._cfg.place_join_frac
            want_imp = (
                not want_join and u_mix < self._cfg.place_join_frac + self._cfg.lo_improve_frac
            )
            want_crown = (
                not want_join
                and not want_imp
                and u_mix
                < self._cfg.place_join_frac + self._cfg.lo_improve_frac + self._cfg.crown_stack_frac
            )
            if want_buy:
                chase = self._unhit_chase("buy")
                if chase is None:
                    chase = self._vac_chase("buy")
                chased = chase is not None
                repost_l = self._repost_level("buy")
                if repost_l is not None:
                    level = repost_l
                elif chase is not None:
                    level = chase
                elif want_join and bb is not None:
                    level = bb
                elif want_crown and bb is not None:
                    cand = (
                        bb
                        - self._cfg.crown_offset
                        - int(self._rng.random() * (self._cfg.crown_stack_span + 1))
                    )
                    if (
                        self._cfg.crown_cap <= 0
                        or len(self._bids.get(cand, ())) < self._cfg.crown_cap
                    ):
                        level = cand
                        if self._cfg.crown_size_pmf is not None:
                            k = self._draw_size(self._crown_size_cdf)
                        self.n_lo_crown += 1
                    else:
                        level = ref - dist
                elif want_imp and ba is not None and bb is not None and ba - bb > 1:
                    level = bb + 1 + int(self._rng.random() * (ba - bb - 1))
                else:
                    level = ref - dist
                if repost_l is None:
                    level = self._step_unhit("buy", level)
                if repost_l is None and self._is_cooled("buy", level):
                    self.n_lo_suppressed += 1
                    return
                if repost_l is None and self._hit_starved("buy", level):
                    self.n_lo_suppressed += 1
                    return
                if ba is None or level < ba:
                    if bb is not None and level == bb:
                        self.n_lo_join += 1
                    elif bb is not None and level > bb:
                        self.n_lo_improve += 1
                    for _ in range(k):
                        if self._touch_capped("buy", level):
                            self.n_lo_capped += 1
                            break
                        self._rest(
                            "buy",
                            level,
                            "repost" if repost_l is not None else ("chase" if chased else "zi"),
                        )
                    self.n_lo_arrivals += 1
                    self.n_lo_units += k
                return
            chase = self._unhit_chase("sell")
            if chase is None:
                chase = self._vac_chase("sell")
            chased = chase is not None
            repost_l = self._repost_level("sell")
            if repost_l is not None:
                level = repost_l
            elif chase is not None:
                level = chase
            elif want_join and ba is not None:
                level = ba
            elif want_crown and ba is not None:
                cand = (
                    ba
                    + self._cfg.crown_offset
                    + int(self._rng.random() * (self._cfg.crown_stack_span + 1))
                )
                if self._cfg.crown_cap <= 0 or len(self._asks.get(cand, ())) < self._cfg.crown_cap:
                    level = cand
                    if self._cfg.crown_size_pmf is not None:
                        k = self._draw_size(self._crown_size_cdf)
                    self.n_lo_crown += 1
                else:
                    level = ref + dist
            elif want_imp and ba is not None and bb is not None and ba - bb > 1:
                level = ba - 1 - int(self._rng.random() * (ba - bb - 1))
            else:
                level = ref + dist
            if repost_l is None:
                level = self._step_unhit("sell", level)
            if repost_l is None and self._is_cooled("sell", level):
                self.n_lo_suppressed += 1
                return
            if repost_l is None and self._hit_starved("sell", level):
                self.n_lo_suppressed += 1
                return
            if bb is None or level > bb:
                if ba is not None and level == ba:
                    self.n_lo_join += 1
                elif ba is not None and level < ba:
                    self.n_lo_improve += 1
                for _ in range(k):
                    if self._touch_capped("sell", level):
                        self.n_lo_capped += 1
                        break
                    self._rest(
                        "sell",
                        level,
                        "repost" if repost_l is not None else ("chase" if chased else "zi"),
                    )
                self.n_lo_arrivals += 1
                self.n_lo_units += k
            return
        # Touch anchoring (Moret & Lillo market-making setting): band follows the
        # best opposite quote; fall back to the reference level when a side is
        # empty so the book can always recover.
        k = self._draw_size(self._lo_size_cdf)
        off = int(self._cfg.lo_offset)
        if self._hawkes is not None and self._cfg.lo_offset_gain > 0.0:
            off += int(round(self._cfg.lo_offset_gain * self._hawkes.excitation(1)))
        imp = self._cfg.lo_improve_frac > 0.0 and self._rng.random() < self._cfg.lo_improve_frac
        want_crown = (
            self._cfg.crown_stack_frac > 0.0 and self._rng.random() < self._cfg.crown_stack_frac
        )
        if want_buy:
            chase = self._unhit_chase("buy")
            if chase is None:
                chase = self._vac_chase("buy")
            chased = chase is not None
            repost_l = self._repost_level("buy")
            if repost_l is not None:
                level = repost_l
            elif chase is not None:
                level = chase
            elif want_crown and bb is not None:
                cand = (
                    bb
                    - self._cfg.crown_offset
                    - int(self._rng.random() * (self._cfg.crown_stack_span + 1))
                )
                if self._cfg.crown_cap <= 0 or len(self._bids.get(cand, ())) < self._cfg.crown_cap:
                    level = cand
                    if self._cfg.crown_size_pmf is not None:
                        k = self._draw_size(self._crown_size_cdf)
                    self.n_lo_crown += 1
                else:
                    anchor = (ba if ba is not None else self._ref_level + 1) - off
                    level = anchor - dist
            elif imp and ba is not None and bb is not None and ba > bb:
                level = bb + int(self._rng.random() * (ba - bb))
            else:
                anchor = (ba if ba is not None else self._ref_level + 1) - off
                level = anchor - dist
            if repost_l is None:
                level = self._step_unhit("buy", level)
            if repost_l is None and self._is_cooled("buy", level):
                self.n_lo_suppressed += 1
                self.n_lo_arrivals += 1
                return
            if repost_l is None and self._hit_starved("buy", level):
                self.n_lo_suppressed += 1
                self.n_lo_arrivals += 1
                return
            if bb is not None and level > bb:
                self.n_lo_improve += 1  # deposit strictly inside the spread
            elif bb is not None and level == bb:
                self.n_lo_join += 1
            for _ in range(k):
                if self._touch_capped("buy", level):
                    self.n_lo_capped += 1
                    break
                self._rest(
                    "buy",
                    level,
                    "repost" if repost_l is not None else ("chase" if chased else "zi"),
                )
        else:
            chase = self._unhit_chase("sell")
            if chase is None:
                chase = self._vac_chase("sell")
            chased = chase is not None
            repost_l = self._repost_level("sell")
            if repost_l is not None:
                level = repost_l
            elif chase is not None:
                level = chase
            elif want_crown and ba is not None:
                cand = (
                    ba
                    + self._cfg.crown_offset
                    + int(self._rng.random() * (self._cfg.crown_stack_span + 1))
                )
                if self._cfg.crown_cap <= 0 or len(self._asks.get(cand, ())) < self._cfg.crown_cap:
                    level = cand
                    if self._cfg.crown_size_pmf is not None:
                        k = self._draw_size(self._crown_size_cdf)
                    self.n_lo_crown += 1
                else:
                    anchor = (bb if bb is not None else self._ref_level - 1) + off
                    level = anchor + dist
            elif imp and ba is not None and bb is not None and ba > bb:
                level = ba - int(self._rng.random() * (ba - bb))
            else:
                anchor = (bb if bb is not None else self._ref_level - 1) + off
                level = anchor + dist
            if repost_l is None:
                level = self._step_unhit("sell", level)
            if repost_l is None and self._is_cooled("sell", level):
                self.n_lo_suppressed += 1
                self.n_lo_arrivals += 1
                return
            if repost_l is None and self._hit_starved("sell", level):
                self.n_lo_suppressed += 1
                self.n_lo_arrivals += 1
                return
            if ba is not None and level < ba:
                self.n_lo_improve += 1
            elif ba is not None and level == ba:
                self.n_lo_join += 1
            for _ in range(k):
                if self._touch_capped("sell", level):
                    self.n_lo_capped += 1
                    break
                self._rest(
                    "sell",
                    level,
                    "repost" if repost_l is not None else ("chase" if chased else "zi"),
                )
        self.n_lo_arrivals += 1
        self.n_lo_units += k

    def _cancel_event(self) -> None:
        bid_d, ask_d = self.bid_depth, self.ask_depth
        total = bid_d + ask_d
        if total == 0:
            return
        release = self._cfg.chase_release
        if release > 0.0 and self._chase_oids and float(self._rng.random()) < release:
            oid = int(self._rng.choice(tuple(self._chase_oids)))
            chase_order = self._orders.get(oid)
            if chase_order is None:  # pragma: no cover - set/registry invariant
                self._chase_oids.discard(oid)
                return
            chase_book = self._asks if chase_order.side == "sell" else self._bids
            chase_touch = min(chase_book) if chase_book is self._asks else max(chase_book)
            reprice = self._cfg.chase_reprice
            reprice_tgt: int | None = None
            if reprice > 0.0 and float(self._rng.random()) < reprice:
                bb_r, ba_r = self.best_bid_level, self.best_ask_level
                if chase_order.side == "buy":
                    if ba_r is not None and ba_r - 1 > chase_order.level:
                        reprice_tgt = ba_r - 1
                elif bb_r is not None and bb_r + 1 < chase_order.level:
                    reprice_tgt = bb_r + 1
            chase_dq = chase_book[chase_order.level]
            self._remove_resting_at(chase_book, chase_order.level, chase_dq.index(oid))
            self.cxl_ages.append(self.t - chase_order.t_submit)
            self.n_cancellations += 1
            self._fate_cancels[chase_order.placement_class] += 1
            self.fate_log.append((chase_order.placement_class, chase_order.queue_ahead, "cancel"))
            d_hit = abs(chase_order.level - chase_touch)
            self.cxl_dist[min(d_hit, 20)] += 1
            if d_hit == 0:
                self.n_cxl_touch += 1
            if reprice_tgt is not None and not self._touch_capped(chase_order.side, reprice_tgt):
                self._rest(chase_order.side, reprice_tgt, "chase")
            else:
                self._maybe_requote(chase_order)
            return
        relief = self._cfg.cxl_unhit_relief
        if relief > 0.0 and self._hit_retreat is not None:
            hit_side, _n_until, r_until, _fill_ev, _s_u, _t_u, _i_u, _f_u = self._hit_retreat
            if self.n_events >= r_until:
                pass  # marker stays for hit_narrow; relief expired
            elif float(self._rng.random()) < relief:
                # Post-fill cancel relief: route this cancel to the
                # HIT side, uniform over its resting depth.
                book = self._asks if hit_side == "sell" else self._bids
                n = len(book) and sum(len(dq) for dq in book.values())
                if n:
                    touch = min(book) if book is self._asks else max(book)
                    k = int(self._rng.integers(n))
                    rel_lvl = 0
                    rel_idx = 0
                    for lvl in sorted(book):
                        cnt = len(book[lvl])
                        if k < cnt:
                            rel_lvl, rel_idx = lvl, k
                            break
                        k -= cnt
                    rel_order = self._remove_resting_at(book, rel_lvl, rel_idx)
                    self.cxl_ages.append(self.t - rel_order.t_submit)
                    self.n_cancellations += 1
                    self._fate_cancels[rel_order.placement_class] += 1
                    self.fate_log.append(
                        (rel_order.placement_class, rel_order.queue_ahead, "cancel")
                    )
                    d_hit = abs(rel_lvl - touch)
                    self.cxl_dist[min(d_hit, 20)] += 1
                    if d_hit == 0:
                        self.n_cxl_touch += 1
                    self._maybe_requote(rel_order)
                    return
        bias = self._cfg.cxl_touch_bias
        if bias > 0.0 and float(self._rng.random()) < bias:
            bb, ba = self.best_bid_level, self.best_ask_level
            decay = self._cfg.cxl_dist_decay
            if decay > 0.0:
                # Distance-decaying propensity: pick the side
                # proportional to its resting depth, then an order on
                # that side weighted exp(-dist/L) — the tape's
                # near-touch ring instead of a touch-only spike.
                side_bid = float(self._rng.random()) * total < bid_d
                book = self._bids if side_bid else self._asks
                touch_lvl = max(book) if side_bid else min(book)
                r = float(self._rng.random())
                wsum = 0.0
                wts: list[tuple[int, int, float]] = []
                for lvl, dq in book.items():
                    w = math.exp(-abs(lvl - touch_lvl) / decay)
                    for idx in range(len(dq)):
                        wsum += w
                        wts.append((lvl, idx, wsum))
                tgt = r * wsum
                order = None
                lvl_hit = None
                for lvl, idx, cs in wts:
                    if tgt <= cs:
                        order = self._remove_resting_at(book, lvl, idx)
                        lvl_hit = lvl
                        break
                if order is None or lvl_hit is None:  # pragma: no cover
                    raise RuntimeError("weighted cancel missed the book")
                self.cxl_ages.append(self.t - order.t_submit)
                self.n_cancellations += 1
                self._fate_cancels[order.placement_class] += 1
                self.fate_log.append((order.placement_class, order.queue_ahead, "cancel"))
                d_hit = abs(lvl_hit - touch_lvl)
                self.cxl_dist[min(d_hit, 20)] += 1
                if d_hit == 0:
                    self.n_cxl_touch += 1
                self._maybe_requote(order)
                return
            tb = len(self._bids[bb]) if bb is not None else 0
            ta = len(self._asks[ba]) if ba is not None else 0
            draw = float(self._rng.random()) * (tb + ta)
            if draw < tb and bb is not None:
                order = self._remove_resting_at(self._bids, bb, 0)
            elif ba is not None:
                order = self._remove_resting_at(self._asks, ba, 0)
            else:  # pragma: no cover - touch depth bookkeeping invariant
                raise RuntimeError("touch cancel on an empty book")
            self.cxl_ages.append(self.t - order.t_submit)
            self.n_cancellations += 1
            self._fate_cancels[order.placement_class] += 1
            self.fate_log.append((order.placement_class, order.queue_ahead, "cancel"))
            self.n_cxl_touch += 1
            self.cxl_dist[0] += 1
            self._maybe_requote(order)
            return
        k = int(self._rng.integers(total))
        if k < bid_d:
            book, idx = self._bids, k
        else:
            book, idx = self._asks, k - bid_d
        damp = self._cfg.cxl_unhit_damp
        if damp > 0.0 and self._hit_retreat is not None:
            hit_side, _n_u, r_until, fill_ev, _s_u, _t_u, _i_u, _f_u = self._hit_retreat
            unhit_book = self._bids if hit_side == "sell" else self._asks
            if self.n_events < r_until and book is unhit_book:
                decay = self._cfg.cxl_unhit_damp_decay
                eff = damp * math.exp(-(self.n_events - fill_ev) / decay) if decay > 0.0 else damp
                if float(self._rng.random()) < eff:
                    return  # protected: the unhit-side order survives
        level = None
        for lvl in sorted(book):
            n = len(book[lvl])
            if idx < n:
                level = lvl
                break
            idx -= n
        if level is None:  # pragma: no cover - depth accounting invariant
            raise RuntimeError("cancellation sampling missed the book")
        touch = max(book) if book is self._bids else min(book)
        order = self._remove_resting_at(book, level, idx)
        self.cxl_ages.append(self.t - order.t_submit)
        self.n_cancellations += 1
        self._fate_cancels[order.placement_class] += 1
        self.fate_log.append((order.placement_class, order.queue_ahead, "cancel"))
        dist = abs(level - touch)
        self.cxl_dist[min(dist, 20)] += 1
        if dist == 0:
            self.n_cxl_touch += 1

    def step(self) -> str:
        """Advance to the next event; returns the event type drawn."""
        mu_eff, p_buy_eff = self._flow_params()
        lo_rate = 2.0 * self._cfg.lam * self._cfg.band
        mo_rate = 2.0 * mu_eff
        cxl_rate = self._cfg.theta_cxl * float(self.total_depth)
        if self._rate_flow is not None:
            s = self._rate_flow.scale()
            lo_rate *= float(s[0])
            mo_rate *= float(s[1])
            cxl_rate *= float(s[2])
            self._rate_flow.advance()
        total = lo_rate + mo_rate + cxl_rate
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"degenerate event rate {total!r}")
        if self._hawkes is not None:
            dt, kind = self._hawkes.step((lo_rate, mo_rate, cxl_rate))
        else:
            dt = float(self._rng.exponential(1.0 / total))
            u = float(self._rng.random()) * total
            kind = 0 if u < lo_rate else (1 if u < lo_rate + mo_rate else 2)
        self._t += dt
        self.n_events += 1
        if self._cfg.lo_tilt_decay > 0.0:
            self._tilt *= 1.0 - self._cfg.lo_tilt_decay
        bb, ba = self.best_bid_level, self.best_ask_level
        if bb is not None and ba is not None:
            mid_level = 0.5 * (bb + ba)
            self._ref_level = int(math.floor(mid_level))
            hl = self._cfg.ref_halflife
            if hl > 0.0:
                # EMA of the mid level; frozen (hl == 0) keeps the seed mid.
                alpha = min(1.0, dt / hl)
                self._ref_ema += alpha * (mid_level - self._ref_ema)
        if self._fill_repost_q:
            self._drain_fill_reposts()
        self._hit_flee()
        if kind == 0:
            self._limit_order_event()
            return "limit"
        if kind == 1:
            side: Side = "buy" if float(self._rng.random()) < p_buy_eff else "sell"
            self.n_mo_arrivals += 1
            # A size-k MO is a burst of unit fills; each consumes the current
            # opposite best, so a burst that exhausts the touch sweeps deeper
            # levels (the tape's multi-level sweep footprint).
            k = self._draw_size(self._mo_size_cdf)
            self.n_mo_units += k
            for _ in range(k):
                self._consume_best(side)
            if self._flow is not None:
                self._flow.advance()
            return "market"
        self._cancel_event()
        return "cancel"

    def run(self, until_t: float) -> None:
        """Step events until the clock reaches ``until_t`` (fail-closed)."""
        horizon = float(until_t)
        if not math.isfinite(horizon):
            raise ValueError(f"until_t must be finite, got {until_t!r}")
        while self._t < horizon:
            self.step()


# ---------------------------------------------------------------------------
# Closed-form quoting baselines
# ---------------------------------------------------------------------------


def _snap_bid(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-9) * tick


def _snap_ask(price: float, tick: float) -> float:
    return math.ceil(price / tick - 1e-9) * tick


def _check_tick(tick: float | None) -> float | None:
    if tick is None:
        return None
    return _pos_finite(tick, "tick")


def avellaneda_stoikov_quotes(
    mid: float,
    inventory: float,
    *,
    gamma: float,
    sigma: float,
    tau: float,
    kappa: float,
    tick: float | None = None,
) -> dict[str, Any]:
    """Avellaneda-Stoikov (2008) reservation price + optimal spread quotes.

    **Composes** ``models.market_making.as_optimal_quotes`` (single source of
    truth for the closed form: ``r = s - q*gamma*sigma^2*tau``, total spread
    ``gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/kappa)``) and adds optional
    tick-grid snapping for the discrete simulator: the bid is floored and the
    ask is ceiled to the grid, which can only widen the raw spread. ``kappa``
    is the fill-intensity decay of ``lambda(delta) = A exp(-kappa*delta)`` in
    inverse *price* units (``kappa_price = kappa_tick / tick``).
    """
    # Lazy: models.market_making owns the A-S closed form (analytics layer) —
    # a deferred import is the sanctioned layer-order cycle-breaker
    # (docs/ARCHITECTURE_GUARDS.md); microstructure must not depend on it at
    # import time.
    from quant_fund.models.market_making import as_optimal_quotes

    tk = _check_tick(tick)
    raw = as_optimal_quotes(mid, inventory, gamma, sigma, tau, kappa)
    out: dict[str, Any] = {
        "reservation_price": float(raw["reservation_price"]),
        "bid": float(raw["bid"]),
        "ask": float(raw["ask"]),
        "half_spread": float(raw["half_spread"]),
        "skew": float(raw["skew"]),
        "bid_raw": float(raw["bid"]),
        "ask_raw": float(raw["ask"]),
        "snapped": False,
    }
    if tk is not None:
        bid_s = _snap_bid(out["bid"], tk)
        ask_s = _snap_ask(out["ask"], tk)
        if ask_s <= bid_s:  # pragma: no cover - raw spread > 0 and snapping widens
            ask_s = bid_s + tk
        out["bid"] = bid_s
        out["ask"] = ask_s
        out["snapped"] = True
    return out


def glft_quotes(
    mid: float,
    inventory: float,
    *,
    gamma: float,
    sigma: float,
    kappa: float,
    a_fill: float,
    tick: float | None = None,
) -> dict[str, Any]:
    """Gueant-Lehalle-Fernandez-Tapia asymptotic closed-form quotes.

    Equations (2)-(3) of Moret & Lillo (2026, arXiv:2609.11614), from Gueant,
    Lehalle & Fernandez-Tapia (2012): with fill intensity ``A*exp(-kappa*delta)``,
    CARA risk aversion ``gamma``, and mid volatility ``sigma``,

        base = (1/gamma) * log(1 + gamma/kappa)
        root = sqrt( sigma^2*gamma / (2*kappa*A) * (1 + gamma/kappa)^(1 + kappa/gamma) )
        ask  = mid + base - ((2q - 1)/2) * root
        bid  = mid - base - ((2q + 1)/2) * root

    The ``base`` term is the inventory-independent half-spread; the ``root``
    term scaled by ``(2q -/+ 1)/2`` is the inventory skew. Like the paper's
    implementation, optional tick snapping floors the bid and ceils the ask.
    ``kappa`` is in inverse price units; ``sigma`` in price/sqrt(time).
    """
    g = _pos_finite(gamma, "gamma")
    s = _pos_finite(sigma, "sigma")
    k = _pos_finite(kappa, "kappa")
    a = _pos_finite(a_fill, "a_fill")
    m = _pos_finite(mid, "mid")
    q = float(inventory)
    if not math.isfinite(q):
        raise ValueError(f"inventory must be finite, got {inventory!r}")
    tk = _check_tick(tick)
    ratio = 1.0 + g / k
    base = math.log(ratio) / g
    root = math.sqrt((s * s * g / (2.0 * k * a)) * ratio ** (1.0 + k / g))
    bid = m - base - ((2.0 * q + 1.0) / 2.0) * root
    ask = m + base - ((2.0 * q - 1.0) / 2.0) * root
    if not (math.isfinite(bid) and math.isfinite(ask)):
        raise ValueError("GLFT quotes are non-finite for these parameters")
    if bid <= 0.0 or ask <= 0.0:
        raise ValueError(
            "GLFT quotes are non-positive (inventory skew too large for this mid); "
            "reduce |inventory|*root or raise mid"
        )
    out: dict[str, Any] = {
        "bid": bid,
        "ask": ask,
        "bid_raw": bid,
        "ask_raw": ask,
        "base_half_spread": base,
        "inventory_skew_unit": root,
        "spread": ask - bid,
        "mid": m,
        "inventory": q,
        "snapped": False,
    }
    if tk is not None:
        bid_s = _snap_bid(bid, tk)
        ask_s = _snap_ask(ask, tk)
        if bid_s <= 0.0:
            raise ValueError("snapped GLFT bid is non-positive")
        if ask_s <= bid_s:  # pragma: no cover - raw spread = 2*base + root > 0
            ask_s = bid_s + tk
        out["bid"] = bid_s
        out["ask"] = ask_s
        out["spread"] = ask_s - bid_s
        out["snapped"] = True
    return out


# ---------------------------------------------------------------------------
# Session runner: plug a quoting policy into the simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MMState:
    """Decision state handed to a quoting policy."""

    t: float
    mid: float | None
    best_bid: float | None
    best_ask: float | None
    inventory: int
    tau: float


QuotePolicy = Callable[[MMState], tuple[float | None, float | None]]


def as_policy(
    *, gamma: float, sigma: float, kappa: float, tick: float | None = None
) -> QuotePolicy:
    """QuotePolicy adapter for ``avellaneda_stoikov_quotes`` (fail-closed)."""
    avellaneda_stoikov_quotes(100.0, 0.0, gamma=gamma, sigma=sigma, tau=1.0, kappa=kappa, tick=tick)

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.mid is None or state.tau <= 0.0:
            return (None, None)
        q = avellaneda_stoikov_quotes(
            state.mid,
            float(state.inventory),
            gamma=gamma,
            sigma=sigma,
            tau=state.tau,
            kappa=kappa,
            tick=tick,
        )
        return (float(q["bid"]), float(q["ask"]))

    return policy


def glft_policy(
    *, gamma: float, sigma: float, kappa: float, a_fill: float, tick: float | None = None
) -> QuotePolicy:
    """QuotePolicy adapter for ``glft_quotes`` (fail-closed)."""
    glft_quotes(100.0, 0.0, gamma=gamma, sigma=sigma, kappa=kappa, a_fill=a_fill, tick=tick)

    def policy(state: MMState) -> tuple[float | None, float | None]:
        if state.mid is None or state.tau <= 0.0:
            return (None, None)
        try:
            q = glft_quotes(
                state.mid,
                float(state.inventory),
                gamma=gamma,
                sigma=sigma,
                kappa=kappa,
                a_fill=a_fill,
                tick=tick,
            )
        except ValueError:
            return (None, None)  # non-positive quotes under extreme skew: stand down
        return (float(q["bid"]), float(q["ask"]))

    return policy


def run_mm_session(
    *,
    config: ZILobConfig,
    policy: QuotePolicy,
    horizon: float,
    decision_interval: float = 1.0,
    flow: MarkovRegimeFlow | ScenarioRegimeFlow | AdversarialFlow | None = None,
    inventory_cap: int | None = None,
    sample_interval: float = 25.0,
) -> dict[str, Any]:
    """Run a market-making session: quotes into the simulator, fills out.

    The policy is consulted every ``decision_interval`` simulated seconds (a
    throttled decision clock, cf. Moret & Lillo 2026 Sec. 3); old MM orders
    are canceled and re-quoted. Quotes are clipped inside the touch (the ZI
    engine does not accept marketable limit orders), the side that would
    breach ``inventory_cap`` is suppressed, and fills are attributed per order
    with queue-position accounting.

    Returns a **SYNTHETIC** diagnostic bundle. ``sim_internal_mtm_pnl_path``
    is the simulator-internal mark-to-market path (cash + inventory*mid); it
    is a correctness diagnostic for this synthetic engine, NEVER a headline
    metric and never market evidence. No live-trading claim.
    """
    if not callable(policy):
        raise TypeError("policy must be callable")
    h = _pos_finite(horizon, "horizon")
    di = _pos_finite(decision_interval, "decision_interval")
    si = _pos_finite(sample_interval, "sample_interval")
    cap: int | None = None
    if inventory_cap is not None:
        if isinstance(inventory_cap, bool) or int(inventory_cap) < 1:
            raise ValueError(f"inventory_cap must be an int >= 1, got {inventory_cap!r}")
        cap = int(inventory_cap)

    sim = ZILobSimulator(config, flow=flow)
    inventory = 0
    cash = 0.0
    bid_oid: int | None = None
    ask_oid: int | None = None
    n_fills = n_fills_bid = n_fills_ask = 0
    n_cancels = n_clips = n_skipped = n_decisions = 0
    queue_ahead_fills: list[int] = []
    fill_waits: list[float] = []
    inv_path: list[int] = []
    inv_times: list[float] = []
    mtm_path: list[float] = []
    max_abs_inv = 0
    last_mid: float | None = sim.mid
    samples: list[BookSample] = [sim.sample()]
    signs: list[float] = []
    trade_cursor = 0

    def _cancel_outstanding() -> None:
        nonlocal bid_oid, ask_oid, n_cancels
        for oid in (bid_oid, ask_oid):
            if oid is not None and sim.cancel_order(oid):
                n_cancels += 1
        bid_oid = None
        ask_oid = None

    def _requote() -> None:
        nonlocal bid_oid, ask_oid, n_clips, n_skipped, n_decisions
        n_decisions += 1
        _cancel_outstanding()
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
        bid_px, ask_px = policy(state)
        if cap is not None:
            if inventory >= cap:
                bid_px = None
            if inventory <= -cap:
                ask_px = None
        tick = config.tick
        ba, bb = sim.best_ask, sim.best_bid
        if bid_px is not None and ba is not None and bid_px >= ba - 1e-12:
            bid_px = ba - tick
            n_clips += 1
            if bid_px <= 0.0:
                bid_px = None
        if ask_px is not None and bb is not None and ask_px <= bb + 1e-12:
            ask_px = bb + tick
            n_clips += 1
        if bid_px is not None and ask_px is not None and bid_px >= ask_px:
            n_skipped += 1
            return
        if bid_px is not None:
            bid_oid = sim.submit_limit_order("buy", bid_px, MM_TAG)
        if ask_px is not None:
            ask_oid = sim.submit_limit_order("sell", ask_px, MM_TAG)

    def _record_path() -> None:
        nonlocal last_mid
        mid = sim.mid
        if mid is not None:
            last_mid = mid
        inv_path.append(inventory)
        inv_times.append(sim.t)
        if mid is not None:
            mtm_path.append(cash + inventory * mid)
        else:
            mtm_path.append(float("nan"))

    def _drain_trades() -> None:
        nonlocal trade_cursor, inventory, cash, bid_oid, ask_oid
        nonlocal n_fills, n_fills_bid, n_fills_ask, max_abs_inv
        while trade_cursor < len(sim.trades):
            tr = sim.trades[trade_cursor]
            trade_cursor += 1
            signs.append(1.0 if tr.aggressor == "buy" else -1.0)
            if tr.maker_tag != MM_TAG:
                continue
            if tr.maker_side == "buy":
                inventory += tr.qty
                cash -= tr.price * tr.qty
                n_fills_bid += 1
            else:
                inventory -= tr.qty
                cash += tr.price * tr.qty
                n_fills_ask += 1
            n_fills += 1
            queue_ahead_fills.append(tr.maker_queue_ahead_at_submit)
            fill_waits.append(tr.t - tr.maker_t_submit)
            if tr.maker_order_id == bid_oid:
                bid_oid = None
            elif tr.maker_order_id == ask_oid:
                ask_oid = None
            max_abs_inv = max(max_abs_inv, abs(inventory))
        if isinstance(flow, AdversarialFlow):
            # Feedback channel: the adversary's picker reads the defender's
            # post-fill inventory when a pending boundary resolves.
            flow.note_inventory(float(inventory))

    _requote()
    _record_path()
    next_decision = di
    next_sample = si
    while sim.t < h:
        sim.step()
        _drain_trades()
        if sim.t >= next_decision:
            _requote()
            _record_path()
            while next_decision <= sim.t:
                next_decision += di
        if sim.t >= next_sample:
            samples.append(sim.sample())
            while next_sample <= sim.t:
                next_sample += si

    final_mid = sim.mid if sim.mid is not None else last_mid
    final_mtm = cash + inventory * final_mid if final_mid is not None else float("nan")
    flow_diag: dict[str, Any] | None = None
    if len(signs) >= 52:
        flow_diag = regime_flow_diagnostics(signs)
    return {
        "label": "SYNTHETIC",
        "data_source": ZI_LOB_REVISION,
        "research_only": True,
        "live_pnl_claim": False,
        "claim": "simulator_internal_diagnostic_only",
        "seed": config.seed,
        "horizon": h,
        "decision_interval": di,
        "inventory_cap": cap,
        "n_events": sim.n_events,
        "n_decisions": n_decisions,
        "n_fills": n_fills,
        "n_fills_bid": n_fills_bid,
        "n_fills_ask": n_fills_ask,
        "n_mm_cancels": n_cancels,
        "n_quote_clips": n_clips,
        "n_skipped_decisions": n_skipped,
        "inventory_final": inventory,
        "max_abs_inventory": max_abs_inv,
        "mean_abs_inventory": float(np.mean(np.abs(np.asarray(inv_path, dtype=np.float64))))
        if inv_path
        else float("nan"),
        "inventory_path": inv_path,
        "inventory_path_times": inv_times,
        # Simulator-internal mark-to-market accounting. Diagnostic only:
        # never a headline metric, never market evidence, no live-trading claim.
        "sim_internal_mtm_pnl_path": mtm_path,
        "sim_internal_mtm_pnl_final": float(final_mtm),
        "mean_queue_ahead_at_fill": float(np.mean(queue_ahead_fills))
        if queue_ahead_fills
        else float("nan"),
        "mean_fill_wait_seconds": float(np.mean(fill_waits)) if fill_waits else float("nan"),
        "phase_metrics": book_phase_metrics(samples),
        "flow_diagnostics": flow_diag,
        "n_signs": len(signs),
        "event_counts": sim.event_counts(),
    }


# ---------------------------------------------------------------------------
# Emergent-microstructure diagnostics
# ---------------------------------------------------------------------------


def order_flow_autocorrelation(signs: Sequence[float] | Array, lag: int = 1) -> float:
    """Lag-``lag`` autocorrelation of the signed aggressor flow (fail-closed).

    ``~0`` for i.i.d. zero-intelligence flow; strictly positive under sticky
    directional regimes (Moret & Lillo 2026 motivation).
    """
    x = np.asarray(signs, dtype=np.float64).ravel()
    lg = int(lag)
    if lg < 1:
        raise ValueError(f"lag must be >= 1, got {lag!r}")
    if x.size < lg + 2:
        raise ValueError(f"need at least {lg + 2} signs, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("signs must be finite")
    a, b = x[: x.size - lg], x[lg:]
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        raise ValueError("constant sign series has undefined autocorrelation")
    return float(np.corrcoef(a, b)[0, 1])


def regime_flow_diagnostics(
    signs: Sequence[float] | Array,
    *,
    window: int = 50,
    autocorr_threshold: float = 0.10,
    bias_std_threshold: float = 0.15,
    flow_bias_threshold: float = 0.30,
) -> dict[str, Any]:
    """Detect sticky directional flow from the signed MO stream.

    Three deterministic detectors (any one fires => ``regime_detected``):
    lag-1 sign autocorrelation, dispersion of the rolling-window buy fraction,
    and the overall flow bias ``|2*mean(buy) - 1|``. Under i.i.d. ZI flow all
    three stay near zero; under two-state Markov modulation they separate the
    regimes. SYNTHETIC diagnostic, not a market-regime claim.
    """
    x = np.asarray(signs, dtype=np.float64).ravel()
    w = int(window)
    if w < 2:
        raise ValueError(f"window must be >= 2, got {window!r}")
    if x.size < w + 2:
        raise ValueError(f"need at least {w + 2} signs, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("signs must be finite")
    for name, thr in (
        ("autocorr_threshold", autocorr_threshold),
        ("bias_std_threshold", bias_std_threshold),
        ("flow_bias_threshold", flow_bias_threshold),
    ):
        if not math.isfinite(float(thr)):
            raise ValueError(f"{name} must be finite")
    buys = (x > 0.0).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(buys)))
    roll = (cs[w:] - cs[:-w]) / float(w)
    if float(x.std()) > 0.0:
        autocorr = order_flow_autocorrelation(x, lag=1)
    else:
        autocorr = float("nan")  # constant flow: bias detector covers it
    buy_frac_mean = float(buys.mean())
    bias = abs(2.0 * buy_frac_mean - 1.0)
    roll_std = float(roll.std())
    roll_range = float(roll.max() - roll.min())
    detected = bool(
        (math.isfinite(autocorr) and autocorr > autocorr_threshold)
        or roll_std > bias_std_threshold
        or bias > flow_bias_threshold
    )
    return {
        "sign_autocorr_lag1": autocorr,
        "buy_fraction_mean": buy_frac_mean,
        "buy_fraction_std_window": roll_std,
        "buy_fraction_range_window": roll_range,
        "flow_bias_abs": bias,
        "regime_detected": detected,
        "n_signs": int(x.size),
        "window": w,
        "autocorr_threshold": float(autocorr_threshold),
        "bias_std_threshold": float(bias_std_threshold),
        "flow_bias_threshold": float(flow_bias_threshold),
        "label": "SYNTHETIC",
    }


def book_phase_metrics(samples: Sequence[BookSample]) -> dict[str, Any]:
    """Spread/depth phase metrics over sampled book states.

    Rosenzweig (2026, arXiv:2609.31260) diagnostics: orderly price discovery
    keeps the spread pinned near one tick with deep queues; disordered/cascade
    states widen the spread and thin the book. The coarse ``phase`` label uses
    fixed deterministic thresholds on SYNTHETIC data — a correctness
    diagnostic, not an empirical market claim.
    """
    if len(samples) < 2:
        raise ValueError(f"need >= 2 book samples, got {len(samples)}")
    spreads = np.array(
        [np.nan if s.spread_ticks is None else float(s.spread_ticks) for s in samples],
        dtype=np.float64,
    )
    depths = np.array([float(s.bid_depth + s.ask_depth) for s in samples], dtype=np.float64)
    bid_d = np.array([float(s.bid_depth) for s in samples], dtype=np.float64)
    ask_d = np.array([float(s.ask_depth) for s in samples], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(depths > 0.0, (bid_d - ask_d) / np.maximum(depths, 1e-12), np.nan)
    defined = np.isfinite(spreads)
    n_def = int(defined.sum())
    mean_spread = float(np.nanmean(spreads)) if n_def else float("nan")
    frac_one = float(np.mean(spreads[defined] == 1.0)) if n_def else float("nan")
    mean_depth = float(depths.mean())
    if (not math.isfinite(mean_spread)) or mean_depth < 2.0 or mean_spread >= 3.0:
        phase = "disordered_wide"
    elif frac_one >= 0.5 and mean_spread < 1.5:
        phase = "orderly_tight"
    else:
        phase = "intermediate"
    return {
        "mean_spread_ticks": mean_spread,
        "spread_std_ticks": float(np.nanstd(spreads)) if n_def else float("nan"),
        "frac_spread_one_tick": frac_one,
        "mean_total_depth": mean_depth,
        "min_total_depth": float(depths.min()),
        "depth_imbalance_mean": float(np.nanmean(imb)),
        "depth_imbalance_std": float(np.nanstd(imb)),
        "phase": phase,
        "n_samples": len(samples),
        "n_spread_defined": n_def,
        "label": "SYNTHETIC",
    }


def _await_two_sided_book(sim: ZILobSimulator, *, why: str, max_events: int = 5_000) -> float:
    """Step until both book sides are populated again (fail-closed on budget).

    A ZI book transiently thins — a run of market orders or cancellations can
    empty one side, making the mid undefined until limit flow re-seeds it.
    """
    mid = sim.mid
    start = sim.n_events
    while mid is None:
        if sim.n_events - start >= max_events:
            raise RuntimeError(
                f"book did not regain two sides within {max_events} events after {why}"
            )
        sim.step()
        mid = sim.mid
    return float(mid)


def metaorder_impact_slope(
    *,
    config: ZILobConfig,
    sizes: Sequence[int] = (16, 32, 64, 128, 256),
    n_seeds: int = 4,
    inject_rate: float | None = None,
    warmup: float = 300.0,
    relax: float = 200.0,
    measure: str = "transient",
) -> dict[str, Any]:
    """Emergent square-root impact check (connects to ``execution.impact``).

    Executes calibrated metaorders inside the ZI-LOB: for each size ``Q`` and
    sign, warm up the book, inject ``Q`` unit market orders at ``inject_rate``
    (interleaved with the natural flow, per Donier et al. 2015), and measure
    the signed mid displacement in return units, averaging over seeds and both
    signs so the mean-zero natural diffusion cancels and the systematic impact
    remains. Regresses ``log|impact|`` on ``log Q``; the square-root law
    predicts slope 0.5 — the same exponent hard-coded in
    ``execution.impact.sqrt_impact_bps`` / ``pow_law_total_impact``.

    ``measure``:
    - ``"transient"`` (default): displacement at the *end of execution*. This
      is the quantity that shows the √Q law in the volume-diffusion regime
      (use ``anchor="ref"`` + ``density_exponent=1`` so cumulative liquidity
      grows with distance from a slow absolute reference).
    - ``"permanent"``: displacement after a further ``relax`` window, once the
      book has re-equilibrated around the (touch-following) flow.

    SYNTHETIC correctness diagnostic, never market evidence.
    """
    if measure not in ("transient", "permanent"):
        raise ValueError(f"measure must be 'transient' or 'permanent', got {measure!r}")
    szs = [int(q) for q in sizes]
    if len(szs) < 2 or any(q < 2 for q in szs):
        raise ValueError(f"sizes must be >= 2 entries, each >= 2, got {sizes!r}")
    if any(b <= a for a, b in zip(szs, szs[1:], strict=False)):
        raise ValueError(f"sizes must be strictly increasing, got {sizes!r}")
    ns = int(n_seeds)
    if ns < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds!r}")
    wu = _pos_finite(warmup, "warmup")
    rx = _pos_finite(relax, "relax")
    rate = _pos_finite(inject_rate, "inject_rate") if inject_rate is not None else 8.0 * config.mu
    interval = 1.0 / rate
    mean_impacts: list[float] = []
    n_runs = 0
    for si, q in enumerate(szs):
        per_run: list[float] = []
        for k in range(ns):
            for sign_idx, side in enumerate(("buy", "sell")):
                seed = config.seed + (si * ns + k) * 2 + sign_idx
                sim = ZILobSimulator(replace(config, seed=seed))
                sim.run(wu)
                mid_pre = _await_two_sided_book(sim, why="warmup")
                t_next = sim.t + interval
                for _ in range(q):
                    sim.run(t_next)
                    sim.inject_market_order(side)  # type: ignore[arg-type]
                    t_next += interval
                if measure == "permanent":
                    sim.run(sim.t + rx)
                mid_post = _await_two_sided_book(sim, why="metaorder execution")
                direction = 1.0 if side == "buy" else -1.0
                per_run.append(direction * (mid_post - mid_pre) / config.s0)
                n_runs += 1
        mean_impact = float(np.mean(per_run))
        if abs(mean_impact) < 1e-12:
            raise ValueError(f"mean impact vanished at size {q}; cannot fit log-log slope")
        mean_impacts.append(abs(mean_impact))
    log_q = np.log(np.asarray(szs, dtype=np.float64))
    log_i = np.log(np.asarray(mean_impacts, dtype=np.float64))
    slope, intercept = np.polyfit(log_q, log_i, 1)
    resid = log_i - (slope * log_q + intercept)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((log_i - log_i.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        "impact_slope": float(slope),
        "impact_intercept": float(intercept),
        "impact_r2": float(r2),
        "sizes": szs,
        "mean_abs_impact_return": mean_impacts,
        "n_runs": n_runs,
        "inject_rate": rate,
        "warmup": wu,
        "relax": rx,
        "measure": measure,
        "label": "SYNTHETIC",
        "note": (
            f"log-log slope of {measure} metaorder impact vs size; 0.5 = square-root law "
            "(cf. execution.impact.sqrt_impact_bps, Donier et al. 2015). "
            "SYNTHETIC correctness diagnostic, not market evidence."
        ),
    }
