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
    seed: int = 0
    # Optional event-size tables ``((size, weight), ...)``. When set, each
    # market-order event consumes ``size`` resting units in one burst
    # (sweeping levels when the touch is thin, so multi-level sweeps
    # emerge) and each limit-order event deposits ``size`` units at its
    # level. ``None`` keeps the unit-size default with zero extra RNG
    # draws — an unset table is bit-identical to the legacy stream.
    mo_size_pmf: tuple[tuple[int, float], ...] | None = None
    lo_size_pmf: tuple[tuple[int, float], ...] | None = None

    def __post_init__(self) -> None:
        _pos_finite(self.s0, "s0")
        _pos_finite(self.tick, "tick")
        _pos_finite(self.lam, "lam")
        _pos_finite(self.mu, "mu")
        _pos_finite(self.theta_cxl, "theta_cxl")
        _prob(self.p_buy, "p_buy")
        _nonneg_finite(self.density_exponent, "density_exponent")
        _nonneg_finite(self.ref_halflife, "ref_halflife")
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
        self._n_orders_created = 0
        # Reference level for LO bands when the opposite side is empty
        # (keeps book recovery possible; falls back to the seeded mid level).
        self._ref_level = 0
        # Slow absolute-price reference (EMA of the mid level) for anchor="ref".
        # Frozen at the seed mid (level 0) when ref_halflife == 0.
        self._ref_ema = 0.0
        # Limit-order placement distance law P(d) ∝ d**density_exponent, d ∈ [1, band].
        band = int(config.band)
        weights = np.arange(1, band + 1, dtype=np.float64) ** float(config.density_exponent)
        self._dist_cdf = np.cumsum(weights / weights.sum())
        self._dist_cdf[-1] = 1.0
        # Event-size tables (None → unit-size, zero extra RNG draws).
        self._mo_size_cdf = self._size_cdf(config.mo_size_pmf)
        self._lo_size_cdf = self._size_cdf(config.lo_size_pmf)
        self.n_mo_units = 0
        self.n_lo_units = 0
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
        }

    # -- order lifecycle ----------------------------------------------------

    def _new_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def _rest(self, side: Side, level: int, tag: str) -> int:
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
        )
        dq.append(oid)
        self._orders[oid] = order
        self._n_orders_created += 1
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
        self.n_cancellations += 1
        return True

    def _remove_resting_at(self, book: dict[int, deque[int]], level: int, idx: int) -> _Order:
        dq = book[level]
        oid = dq[idx]
        del dq[idx]
        if not dq:
            del book[level]
        order = self._orders.pop(oid)
        return order

    def _consume_best(self, aggressor: Side) -> TradeEvent | None:
        """Match one unit MO against the opposite best (price-time priority)."""
        book = self._asks if aggressor == "buy" else self._bids
        if not book:
            self.n_mo_noop += 1
            return None
        level = min(book) if aggressor == "buy" else max(book)
        order = self._remove_resting_at(book, level, 0)
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
        )
        self.trades.append(trade)
        self.n_fills += 1
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
            return self._cfg.mu, self._cfg.p_buy
        st = self._flow.current()
        return self._cfg.mu * st.intensity_mult, st.p_buy

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
        want_buy = u < bid_rate
        if self._cfg.anchor == "ref":
            # Absolute-space anchoring: LOs deposit around a slow reference level
            # so cumulative liquidity grows with distance from the reference and
            # a metaorder climbing away from it meets a deepening static profile
            # (volume-diffusion / square-root regime). Crossing placements are
            # dropped — that aggressiveness is already in the market-order flow.
            ref = int(round(self._ref_ema))
            k = self._draw_size(self._lo_size_cdf)
            if want_buy:
                level = ref - dist
                if ba is None or level < ba:
                    for _ in range(k):
                        self._rest("buy", level, "zi")
                    self.n_lo_arrivals += 1
                    self.n_lo_units += k
                return
            level = ref + dist
            if bb is None or level > bb:
                for _ in range(k):
                    self._rest("sell", level, "zi")
                self.n_lo_arrivals += 1
                self.n_lo_units += k
            return
        # Touch anchoring (Moret & Lillo market-making setting): band follows the
        # best opposite quote; fall back to the reference level when a side is
        # empty so the book can always recover.
        k = self._draw_size(self._lo_size_cdf)
        if want_buy:
            anchor = ba if ba is not None else self._ref_level + 1
            for _ in range(k):
                self._rest("buy", anchor - dist, "zi")
        else:
            anchor = bb if bb is not None else self._ref_level - 1
            for _ in range(k):
                self._rest("sell", anchor + dist, "zi")
        self.n_lo_arrivals += 1
        self.n_lo_units += k

    def _cancel_event(self) -> None:
        bid_d, ask_d = self.bid_depth, self.ask_depth
        total = bid_d + ask_d
        if total == 0:
            return
        k = int(self._rng.integers(total))
        if k < bid_d:
            book, idx = self._bids, k
        else:
            book, idx = self._asks, k - bid_d
        level = None
        for lvl in sorted(book):
            n = len(book[lvl])
            if idx < n:
                level = lvl
                break
            idx -= n
        if level is None:  # pragma: no cover - depth accounting invariant
            raise RuntimeError("cancellation sampling missed the book")
        self._remove_resting_at(book, level, idx)
        self.n_cancellations += 1

    def step(self) -> str:
        """Advance to the next event; returns the event type drawn."""
        mu_eff, p_buy_eff = self._flow_params()
        lo_rate = 2.0 * self._cfg.lam * self._cfg.band
        mo_rate = 2.0 * mu_eff
        cxl_rate = self._cfg.theta_cxl * float(self.total_depth)
        total = lo_rate + mo_rate + cxl_rate
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"degenerate event rate {total!r}")
        dt = float(self._rng.exponential(1.0 / total))
        self._t += dt
        u = float(self._rng.random()) * total
        self.n_events += 1
        bb, ba = self.best_bid_level, self.best_ask_level
        if bb is not None and ba is not None:
            mid_level = 0.5 * (bb + ba)
            self._ref_level = int(math.floor(mid_level))
            hl = self._cfg.ref_halflife
            if hl > 0.0:
                # EMA of the mid level; frozen (hl == 0) keeps the seed mid.
                alpha = min(1.0, dt / hl)
                self._ref_ema += alpha * (mid_level - self._ref_ema)
        if u < lo_rate:
            self._limit_order_event()
            return "limit"
        if u < lo_rate + mo_rate:
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
    flow: MarkovRegimeFlow | None = None,
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
