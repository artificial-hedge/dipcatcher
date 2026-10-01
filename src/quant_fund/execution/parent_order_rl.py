"""SYNTHETIC offline parent-order DQN with a finite closing-auction event.

This is a small adaptation of Mnih et al. (2015), Nature 518, 529-533,
https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf:
uniform experience replay, epsilon-greedy exploration and a periodically
copied target network. It is neither a reproduction of Graf--Mastrolia nor
evidence of real execution quality. Torch is lazy, CPU-only during fitting;
exported parameters support numpy inference without torch.

Public snapshots and subsequently realized liquidity have separate types
and publication clocks. A child order decided at snapshot i can fill only
at event i+1. The policy receives no future event, realized auction price,
queue allocation, or realized volume. Limit orders are one-event IOC orders;
auction submissions are allowed only at the final pre-auction observation.
The public auction deadline is known, but allocation and clearing are hidden.

All prices, depths, queue allocations and fills are synthetic. The tape is
exogenous: it has no order-driven price feedback, exchange priority model,
hidden liquidity, endogenous competitors or empirically calibrated auction.
Filled-quantity shortfall, fees, terminal unfilled preference and running
inventory preference are reported separately. Inventory penalties are not
cash expenses. Missing quantity is never fabricated into a completed fill.
No broker connectivity, live claim, headline returns or SOTA claim exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory

Array = NDArray[np.float64]
Side = Literal["buy", "sell"]
ActionKind = Literal["wait", "market", "limit", "auction"]
REVISION = "SYNTHETIC_PARENT_ORDER_DQN_v1"
N_FEATURES = 14
N_ACTIONS = 5


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("all clocks must be timezone-aware")
    return value.astimezone(UTC)


def _number(value: float, name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _count(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _json(value: Any) -> bytes:
    def default(obj: Any) -> str:
        if isinstance(obj, datetime):
            return _utc(obj).isoformat()
        raise TypeError(f"unsupported receipt type: {type(obj).__name__}")

    return json.dumps(
        value, default=default, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _code_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _ac_code_hash() -> str:
    return hashlib.sha256(
        Path(almgren_chriss_trajectory.__code__.co_filename).read_bytes()
    ).hexdigest()


class _Backward(Protocol):
    def backward(self) -> None: ...


@dataclass(frozen=True)
class BookSnapshot:
    """Published observable touch, volume forecast and optional auction indication."""

    event_time: datetime
    available_time: datetime
    bid: float
    ask: float
    bid_quantity: float
    ask_quantity: float
    volume_forecast: float
    indicative_price: float | None = None
    indicative_quantity: float = 0.0

    def __post_init__(self) -> None:
        if _utc(self.available_time) < _utc(self.event_time):
            raise ValueError("snapshot publication cannot precede its event")
        for name in ("bid", "ask"):
            _number(getattr(self, name), name, positive=True)
        for name in ("bid_quantity", "ask_quantity", "volume_forecast", "indicative_quantity"):
            _number(getattr(self, name), name)
        if self.bid > self.ask:
            raise ValueError("crossed book snapshots are forbidden")
        if self.indicative_price is not None:
            _number(self.indicative_price, "indicative_price", positive=True)


@dataclass(frozen=True)
class RealizedLiquidity:
    """Hidden until the next event is published; never a policy input.

    Passive allocation is a supplied synthetic queue outcome at the event
    bid (buy) or ask (sell), subject to the submitted limit. Auction allocation
    is capped by both supplied depth and participation times auction volume.
    """

    event_time: datetime
    available_time: datetime
    bid: float
    ask: float
    bid_quantity: float
    ask_quantity: float
    traded_volume: float
    passive_buy_quantity: float = 0.0
    passive_sell_quantity: float = 0.0
    auction_price: float | None = None
    auction_quantity: float = 0.0

    def __post_init__(self) -> None:
        if _utc(self.available_time) < _utc(self.event_time):
            raise ValueError("liquidity publication cannot precede its event")
        for name in ("bid", "ask"):
            _number(getattr(self, name), name, positive=True)
        for name in (
            "bid_quantity",
            "ask_quantity",
            "traded_volume",
            "passive_buy_quantity",
            "passive_sell_quantity",
            "auction_quantity",
        ):
            _number(getattr(self, name), name)
        if self.bid > self.ask:
            raise ValueError("crossed realized books are forbidden")
        if self.auction_price is not None:
            _number(self.auction_price, "auction_price", positive=True)
        elif self.auction_quantity != 0:
            raise ValueError("auction allocation requires an auction price")


@dataclass(frozen=True)
class SyntheticExecutionTape:
    episode_id: str
    seed: int
    observations: tuple[BookSnapshot, ...]
    events: tuple[RealizedLiquidity, ...]
    auction_submit_deadline: datetime

    def __post_init__(self) -> None:
        if (
            not self.episode_id.strip()
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise ValueError("episode_id and integer generator seed are required")
        observations, events = tuple(self.observations), tuple(self.events)
        _count(len(events), "event count", 512)
        if len(observations) != len(events) + 1:
            raise ValueError(
                "one initial observation and one published observation per event required"
            )
        for index, event in enumerate(events):
            before, after = observations[index : index + 2]
            if not (
                _utc(before.available_time) < _utc(event.event_time) <= _utc(event.available_time)
            ):
                raise ValueError(
                    "a next event must occur strictly after the current decision clock"
                )
            if (
                _utc(after.event_time) != _utc(event.event_time)
                or _utc(after.available_time) != _utc(event.available_time)
                or (after.bid, after.ask, after.bid_quantity, after.ask_quantity)
                != (event.bid, event.ask, event.bid_quantity, event.ask_quantity)
            ):
                raise ValueError("published next book must align with its realized event")
            if (event.auction_price is not None) != (index == len(events) - 1):
                raise ValueError("exactly the final event must be a closing auction")
        cutoff = _utc(self.auction_submit_deadline)
        if not (_utc(observations[0].available_time) <= cutoff < _utc(events[-1].event_time)):
            raise ValueError("auction submission deadline must precede closing and follow arrival")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "events", events)

    @property
    def path_sha256(self) -> str:
        """Economic path identity ignores names, seeds and absolute calendar origin.

        Relative event/publication/auction clocks remain bound. Redating a
        training trajectory therefore cannot disguise evaluation path reuse.
        Full calendar and generator metadata are bound by scenario.sha256.
        """
        origin = _utc(self.observations[0].event_time)

        def relative(row: dict[str, Any]) -> dict[str, Any]:
            return {
                k: (_utc(v) - origin).total_seconds() if isinstance(v, datetime) else v
                for k, v in row.items()
            }

        return _hash(
            (
                [relative(asdict(o)) for o in self.observations],
                [relative(asdict(e)) for e in self.events],
                (_utc(self.auction_submit_deadline) - origin).total_seconds(),
            )
        )

    def observations_payload(self) -> list[dict[str, Any]]:
        return [asdict(o) for o in self.observations]


@dataclass(frozen=True)
class ParentOrder:
    quantity: float
    side: Side
    arrival_price: float
    cash_budget: float
    max_participation: float
    deadline: datetime
    unfilled_penalty_per_unit: float = 5.0
    inventory_penalty_per_step: float = 0.0

    def __post_init__(self) -> None:
        for name in ("quantity", "arrival_price", "max_participation"):
            _number(getattr(self, name), name, positive=True)
        for name in ("cash_budget", "unfilled_penalty_per_unit", "inventory_penalty_per_step"):
            _number(getattr(self, name), name)
        if self.side not in ("buy", "sell") or self.max_participation > 1:
            raise ValueError("side must be buy/sell and participation must be at most one")
        if not all(
            math.isfinite(self.quantity * p)
            for p in (self.arrival_price, self.unfilled_penalty_per_unit)
        ):
            raise ValueError("parent notional and terminal preference must be finite")
        _utc(self.deadline)

    @property
    def sign(self) -> int:
        return 1 if self.side == "buy" else -1


@dataclass(frozen=True)
class SimulatorConfig:
    fee_per_unit: float = 0.0
    market_impact_per_unit: float = 0.0

    def __post_init__(self) -> None:
        _number(self.fee_per_unit, "fee_per_unit")
        _number(self.market_impact_per_unit, "market_impact_per_unit")


@dataclass(frozen=True)
class ExecutionScenario:
    tape: SyntheticExecutionTape
    order: ParentOrder
    config: SimulatorConfig = SimulatorConfig()

    def __post_init__(self) -> None:
        if _utc(self.order.deadline) != _utc(self.tape.events[-1].event_time):
            raise ValueError("parent deadline must equal the final auction event")
        first = self.tape.observations[0]
        if not math.isclose(
            self.order.arrival_price, first.bid / 2 + first.ask / 2, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("arrival benchmark must match the initial published midpoint")
        if self.order.side == "sell" and any(
            e.bid <= self.config.market_impact_per_unit * self.order.quantity
            for e in self.tape.events
        ):
            raise ValueError("maximum sell impact would produce a nonpositive execution price")

    @property
    def sha256(self) -> str:
        return _hash((asdict(self.tape), asdict(self.order), asdict(self.config)))


@dataclass(frozen=True)
class ExecutionState:
    """The complete policy interface: no access to the event tape or future fills."""

    book: BookSnapshot
    order: ParentOrder
    time_index: int
    horizon: int
    remaining_inventory: float
    cash_remaining: float
    next_is_auction: bool
    auction_eligible: bool

    def __post_init__(self) -> None:
        _count(self.horizon, "state horizon", 512)
        if (
            isinstance(self.time_index, bool)
            or not isinstance(self.time_index, int)
            or not 0 <= self.time_index <= self.horizon
        ):
            raise ValueError("state index outside the finite horizon")
        _number(self.remaining_inventory, "state remaining inventory")
        _number(self.cash_remaining, "state remaining cash")
        if self.remaining_inventory > self.order.quantity + 1e-10:
            raise ValueError("state remaining inventory exceeds parent quantity")
        if self.next_is_auction != (self.time_index == self.horizon - 1) or (
            self.auction_eligible and not self.next_is_auction
        ):
            raise ValueError("state auction flags do not align with the known schedule")

    @property
    def urgency(self) -> float:
        return (
            (self.remaining_inventory / self.order.quantity)
            * self.horizon
            / max(1, self.horizon - self.time_index)
        )

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0, (_utc(self.order.deadline) - _utc(self.book.available_time)).total_seconds()
        )

    def vector(self) -> Array:
        o, b = self.order, self.book
        mid = (b.bid + b.ask) / 2
        values = np.asarray(
            [
                self.remaining_inventory / o.quantity,
                (self.horizon - self.time_index) / self.horizon,
                min(self.urgency, 10),
                o.sign,
                o.sign * (mid / o.arrival_price - 1) * 100,
                (b.ask - b.bid) / o.arrival_price * 100,
                min(b.bid_quantity / o.quantity, 10),
                min(b.ask_quantity / o.quantity, 10),
                min(b.volume_forecast / o.quantity, 10),
                min(self.cash_remaining / (o.quantity * o.arrival_price), 10),
                float(self.next_is_auction),
                float(self.auction_eligible),
                o.sign * ((b.indicative_price or o.arrival_price) / o.arrival_price - 1) * 100,
                min(b.indicative_quantity / o.quantity, 10),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("state encoding overflow")
        return values

    def action_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[0] = True
        if self.time_index < self.horizon and self.remaining_inventory > 1e-10:
            if self.next_is_auction:
                mask[4] = self.auction_eligible
            else:
                mask[1:4] = True
        return mask


@dataclass(frozen=True)
class TradeAction:
    decision_time: datetime
    kind: ActionKind
    quantity: float
    limit_price: float | None = None

    def __post_init__(self) -> None:
        _utc(self.decision_time)
        _number(self.quantity, "action quantity")
        if self.kind not in ("wait", "market", "limit", "auction"):
            raise ValueError("unsupported action kind")
        if self.kind == "wait" and self.quantity != 0:
            raise ValueError("wait must have zero quantity")
        if self.kind == "limit":
            if self.limit_price is None:
                raise ValueError("limit price is required")
            _number(self.limit_price, "limit price", positive=True)
        elif self.limit_price is not None:
            raise ValueError("only a limit action has a limit price")


def grid_action(state: ExecutionState, index: int) -> TradeAction:
    if (
        isinstance(index, bool)
        or not isinstance(index, (int, np.integer))
        or not 0 <= index < N_ACTIONS
    ):
        raise ValueError("action index outside discrete action grid")
    if not state.action_mask()[index]:
        raise ValueError("action is unavailable at this decision")
    quantity = (
        min(state.remaining_inventory, state.order.quantity * 0.25)
        if index == 1
        else state.remaining_inventory
    )
    kind: ActionKind = ("wait", "market", "market", "limit", "auction")[index]
    price = state.book.bid if state.order.side == "buy" else state.book.ask
    return TradeAction(
        state.book.available_time,
        kind,
        0 if index == 0 else quantity,
        price if index == 3 else None,
    )


@dataclass(frozen=True)
class ExecutionFill:
    event_index: int
    decision_time: datetime
    event_time: datetime
    available_time: datetime
    kind: ActionKind
    quantity: float
    price: float
    fees: float
    participation_capacity: float
    liquidity_capacity: float


@dataclass(frozen=True)
class EpisodeAnalytics:
    completion_fraction: float
    unfilled_quantity: float
    sim_internal_filled_shortfall: float
    sim_internal_fees: float
    sim_internal_terminal_preference: float
    sim_internal_inventory_preference: float
    sim_internal_objective: float
    cash_remaining: float
    auction_filled_quantity: float


def execution_analytics(
    order: ParentOrder,
    fills: Sequence[ExecutionFill],
    remaining: float,
    *,
    inventory_preference: float = 0.0,
) -> EpisodeAnalytics:
    """Independent ledger arithmetic; no invented mark for missing shares."""
    _number(remaining, "remaining inventory")
    _number(inventory_preference, "inventory preference")
    total, shortfall, fees, cash, auction = 0.0, 0.0, 0.0, order.cash_budget, 0.0
    for fill in fills:
        _number(fill.quantity, "fill quantity", positive=True)
        _number(fill.price, "fill price", positive=True)
        _number(fill.fees, "fill fees")
        _number(fill.participation_capacity, "participation capacity")
        _number(fill.liquidity_capacity, "liquidity capacity")
        if not (_utc(fill.decision_time) < _utc(fill.event_time) <= _utc(fill.available_time)):
            raise ValueError("fill occurs before decision or is prematurely available")
        if fill.quantity > min(fill.participation_capacity, fill.liquidity_capacity) + 1e-8:
            raise ValueError("fill exceeds participation or liquidity capacity")
        total += fill.quantity
        fees += fill.fees
        shortfall += order.sign * (fill.price - order.arrival_price) * fill.quantity
        cash -= order.sign * fill.price * fill.quantity + fill.fees
        if cash < -1e-7:
            raise ValueError("fill exceeds cash budget")
        if fill.kind == "auction":
            auction += fill.quantity
    if not math.isclose(total + remaining, order.quantity, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("ledger quantity does not reconcile with parent inventory")
    terminal = order.unfilled_penalty_per_unit * remaining
    return EpisodeAnalytics(
        total / order.quantity,
        remaining,
        shortfall,
        fees,
        terminal,
        inventory_preference,
        shortfall + fees + terminal + inventory_preference,
        max(cash, 0),
        auction,
    )


@dataclass(frozen=True)
class ExecutionEpisode:
    episode_id: str
    policy_id: str
    scenario_sha256: str
    actions: tuple[TradeAction, ...]
    fills: tuple[ExecutionFill, ...]
    remaining_path: tuple[float, ...]
    analytics: EpisodeAnalytics
    decision_latency_ns: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExecutionStep:
    state: ExecutionState
    reward: float
    done: bool
    fill: ExecutionFill | None


class ParentOrderEnvironment:
    """Finite event engine with bounded synthetic fill, cash and inventory ledgers."""

    def __init__(self, scenario: ExecutionScenario) -> None:
        self._scenario = scenario
        self.reset()

    def reset(self) -> ExecutionState:
        self._index = 0
        self._remaining = self._scenario.order.quantity
        self._cash = self._scenario.order.cash_budget
        self._actions: list[TradeAction] = []
        self._fills: list[ExecutionFill] = []
        self._remaining_path = [self._remaining]
        self._inventory_preference = 0.0
        return self.state()

    def state(self) -> ExecutionState:
        tape = self._scenario.tape
        horizon = len(tape.events)
        book = tape.observations[self._index]
        final = self._index == horizon - 1
        return ExecutionState(
            book,
            self._scenario.order,
            self._index,
            horizon,
            self._remaining,
            self._cash,
            final,
            final and _utc(book.available_time) <= _utc(tape.auction_submit_deadline),
        )

    def _validate_action(self, action: TradeAction) -> None:
        state = self.state()
        if self._index == state.horizon:
            raise ValueError("episode is already terminal")
        if _utc(action.decision_time) != _utc(state.book.available_time):
            raise ValueError("action clock must equal the currently published observation")
        if action.quantity > self._remaining + 1e-10:
            raise ValueError("child quantity exceeds remaining inventory")
        if action.kind == "auction" and not state.auction_eligible:
            raise ValueError("auction action is unavailable or submission deadline was missed")
        if state.next_is_auction and action.kind not in ("wait", "auction"):
            raise ValueError("only auction submission or wait is allowed before closing")

    def _resolve_fill(self, action: TradeAction) -> ExecutionFill | None:
        event = self._scenario.tape.events[self._index]
        order, config = self._scenario.order, self._scenario.config
        buy = order.side == "buy"
        participation = order.max_participation * event.traded_volume
        if action.kind == "wait" or action.quantity == 0:
            return None
        if action.kind == "auction":
            assert event.auction_price is not None
            base, depth = event.auction_price, event.auction_quantity
        elif action.kind == "limit":
            assert action.limit_price is not None
            base = event.bid if buy else event.ask
            depth = event.passive_buy_quantity if buy else event.passive_sell_quantity
            if (buy and base > action.limit_price) or (not buy and base < action.limit_price):
                return None
        else:
            base = event.ask if buy else event.bid
            depth = event.ask_quantity if buy else event.bid_quantity
        quantity = min(action.quantity, self._remaining, participation, depth)
        impact = config.market_impact_per_unit if action.kind == "market" else 0.0
        if buy:
            unit = base + config.fee_per_unit
            affordable = (
                self._cash / unit
                if impact == 0
                else 2 * self._cash / (unit + math.sqrt(unit**2 + 4 * impact * self._cash))
            )
            quantity = min(quantity, affordable)
        else:
            # Selling can consume cash when fees/impact exceed gross proceeds.
            unit = config.fee_per_unit - base
            if impact > 0:
                affordable = (-unit + math.sqrt(unit**2 + 4 * impact * self._cash)) / (2 * impact)
                quantity = min(quantity, affordable)
            elif unit > 0:
                quantity = min(quantity, self._cash / unit)
        if quantity <= 1e-12:
            return None
        price = base + order.sign * impact * quantity
        return ExecutionFill(
            self._index,
            action.decision_time,
            event.event_time,
            event.available_time,
            action.kind,
            quantity,
            price,
            config.fee_per_unit * quantity,
            participation,
            depth,
        )

    def step(self, action: TradeAction) -> ExecutionStep:
        self._validate_action(action)
        fill = self._resolve_fill(action)
        order = self._scenario.order
        cost = 0.0
        if fill is not None:
            self._remaining = max(0.0, self._remaining - fill.quantity)
            self._cash -= order.sign * fill.price * fill.quantity + fill.fees
            self._cash = max(0.0, self._cash)
            cost = order.sign * (fill.price - order.arrival_price) * fill.quantity + fill.fees
            self._fills.append(fill)
        self._actions.append(action)
        self._index += 1
        self._remaining_path.append(self._remaining)
        inventory = order.inventory_penalty_per_step * (self._remaining / order.quantity) ** 2
        self._inventory_preference += inventory
        done = self._index == len(self._scenario.tape.events)
        penalty = order.unfilled_penalty_per_unit * self._remaining if done else 0.0
        return ExecutionStep(
            self.state(), -(cost + inventory + penalty) / order.quantity, done, fill
        )

    def result(
        self, policy_id: str, *, decision_latency_ns: Sequence[int] = ()
    ) -> ExecutionEpisode:
        if self._index != len(self._scenario.tape.events):
            raise ValueError("partial episodes cannot emit terminal analytics")
        analytics = execution_analytics(
            self._scenario.order,
            self._fills,
            self._remaining,
            inventory_preference=self._inventory_preference,
        )
        if not math.isclose(analytics.cash_remaining, self._cash, abs_tol=1e-7, rel_tol=1e-9):
            raise RuntimeError("independent cash arithmetic disagrees with simulator")
        return ExecutionEpisode(
            self._scenario.tape.episode_id,
            policy_id,
            self._scenario.sha256,
            tuple(self._actions),
            tuple(self._fills),
            tuple(self._remaining_path),
            analytics,
            tuple(decision_latency_ns),
        )


class ExecutionPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def act(self, state: ExecutionState) -> TradeAction: ...


@dataclass(frozen=True)
class TWAPPolicy:
    policy_id: str = "twap_catchup"

    def act(self, state: ExecutionState) -> TradeAction:
        if state.remaining_inventory <= 1e-10:
            return grid_action(state, 0)
        if state.next_is_auction:
            return grid_action(state, 4 if state.auction_eligible else 0)
        desired_filled = state.order.quantity * (state.time_index + 1) / state.horizon
        quantity = max(0.0, desired_filled - (state.order.quantity - state.remaining_inventory))
        return TradeAction(
            state.book.available_time, "market", min(quantity, state.remaining_inventory)
        )


@dataclass(frozen=True)
class AlmgrenChrissPolicy:
    """Known-horizon AC schedule, catching up missed slices under common constraints."""

    sigma: float = 0.1
    eta: float = 0.01
    risk_aversion: float = 0.1
    policy_id: str = "almgren_chriss_catchup"

    def __post_init__(self) -> None:
        _number(self.sigma, "sigma")
        _number(self.eta, "eta", positive=True)
        _number(self.risk_aversion, "risk_aversion")

    def act(self, state: ExecutionState) -> TradeAction:
        if state.remaining_inventory <= 1e-10 or state.next_is_auction:
            return grid_action(
                state, 4 if state.remaining_inventory > 1e-10 and state.auction_eligible else 0
            )
        holdings = almgren_chriss_trajectory(
            state.order.quantity,
            state.horizon,
            sigma=self.sigma,
            eta=self.eta,
            gamma=0,
            risk_aversion=self.risk_aversion,
        )
        quantity = max(0.0, state.remaining_inventory - float(holdings[state.time_index + 1]))
        return TradeAction(
            state.book.available_time, "market", min(quantity, state.remaining_inventory)
        )


def simulate_episode(policy: ExecutionPolicy, scenario: ExecutionScenario) -> ExecutionEpisode:
    env = ParentOrderEnvironment(scenario)
    timings: list[int] = []
    while env.state().time_index < env.state().horizon:
        state = env.state()
        started = time.perf_counter_ns()
        action = policy.act(state)
        timings.append(time.perf_counter_ns() - started)
        env.step(action)
    return env.result(policy.policy_id, decision_latency_ns=timings)


@dataclass(frozen=True)
class DQNConfig:
    hidden_dim: int = 16
    rounds: int = 100
    batch_size: int = 32
    replay_capacity: int = 4096
    target_update_steps: int = 40
    learning_rate: float = 0.003
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        for name, maximum in (
            ("hidden_dim", 128),
            ("rounds", 10_000),
            ("batch_size", 1024),
            ("replay_capacity", 100_000),
            ("target_update_steps", 100_000),
        ):
            _count(getattr(self, name), name, maximum)
        _number(self.learning_rate, "learning rate", positive=True)
        for name in ("gamma", "epsilon_start", "epsilon_end"):
            if _number(getattr(self, name), name) > 1:
                raise ValueError(f"{name} must be at most one")
        if self.epsilon_end > self.epsilon_start or self.batch_size > self.replay_capacity:
            raise ValueError("epsilon endpoints or replay batch capacity are inconsistent")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be a uint32 integer")


@dataclass(frozen=True)
class TrainingTrace:
    episode_objectives: tuple[float, ...]
    td_losses: tuple[float, ...]
    transitions: int
    optimizer_steps: int
    target_copies: int
    initial_parameters_sha256: str
    final_parameters_sha256: str


class _Replay:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.rows: list[tuple[Array, int, float, Array, bool, NDArray[np.bool_]]] = []
        self.index = 0

    def append(self, row: tuple[Array, int, float, Array, bool, NDArray[np.bool_]]) -> None:
        if len(self.rows) < self.capacity:
            self.rows.append(row)
        else:
            self.rows[self.index] = row
        self.index = (self.index + 1) % self.capacity


class ParentOrderDQN:
    """Small two-layer Q network; fit is the only torch-dependent entry point."""

    policy_id = "parent_order_dqn"

    def __init__(self, config: DQNConfig = DQNConfig()) -> None:
        self.config = config
        self._weights: tuple[Array, ...] | None = None
        self._artifact: dict[str, Any] | None = None
        self._model_sha256: str | None = None
        self.training_trace: TrainingTrace | None = None

    def fit(self, scenarios: Sequence[ExecutionScenario], *, fit_as_of: datetime) -> ParentOrderDQN:
        training = tuple(scenarios)
        cutoff = _utc(fit_as_of)
        if not training or len(training) > 10_000:
            raise ValueError("training requires between 1 and 10000 synthetic scenarios")
        ids = [s.tape.episode_id for s in training]
        hashes = [s.tape.path_sha256 for s in training]
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise ValueError("training episode identities and paths must be unique")
        ends = [_utc(s.tape.events[-1].available_time) for s in training]
        if any(t > cutoff for t in ends) or any(
            end >= _utc(nxt.tape.observations[0].available_time)
            for end, nxt in zip(ends, training[1:], strict=False)
        ):
            raise ValueError(
                "training paths must be chronological, disjoint and fully published by fit_as_of"
            )
        transitions = sum(len(s.tape.events) for s in training) * self.config.rounds
        if transitions > 1_000_000:
            raise ValueError("training exceeds the one-million transition budget")
        if sum(len(s.tape.events) for s in training) > 10_000:
            raise ValueError("training source manifest exceeds ten-thousand event budget")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("ParentOrderDQN.fit requires the optional torch nn extra") from exc

        self._weights, self._artifact, self._model_sha256, self.training_trace = (
            None,
            None,
            None,
            None,
        )
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.config.seed)
                network = nn.Sequential(
                    nn.Linear(N_FEATURES, self.config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.config.hidden_dim, N_ACTIONS),
                )
                target = nn.Sequential(
                    nn.Linear(N_FEATURES, self.config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self.config.hidden_dim, N_ACTIONS),
                )
                target.load_state_dict(network.state_dict())
                target.eval()
                for parameter in target.parameters():
                    parameter.requires_grad_(False)
                initial = self._parameter_hash(network)
                optimizer = torch.optim.Adam(network.parameters(), lr=self.config.learning_rate)
                replay = _Replay(self.config.replay_capacity)
                rng = np.random.default_rng(self.config.seed)
                losses: list[float] = []
                objectives: list[float] = []
                steps, copies, seen = 0, 0, 0
                for _round in range(self.config.rounds):
                    for scenario in training:
                        env = ParentOrderEnvironment(scenario)
                        while env.state().time_index < env.state().horizon:
                            state = env.state()
                            vector, mask = state.vector(), state.action_mask()
                            epsilon = self.config.epsilon_start + (
                                self.config.epsilon_end - self.config.epsilon_start
                            ) * min(1, seen / max(1, transitions * 0.8))
                            if rng.random() < epsilon:
                                action = int(rng.choice(np.flatnonzero(mask)))
                            else:
                                with torch.no_grad():
                                    values = network(
                                        torch.tensor(vector, dtype=torch.float32)
                                    ).numpy()
                                action = int(np.argmax(np.where(mask, values, -np.inf)))
                            step = env.step(grid_action(state, action))
                            replay.append(
                                (
                                    vector,
                                    action,
                                    step.reward,
                                    step.state.vector(),
                                    step.done,
                                    step.state.action_mask(),
                                )
                            )
                            seen += 1
                            if len(replay.rows) >= self.config.batch_size:
                                sample = [
                                    replay.rows[int(i)]
                                    for i in rng.choice(
                                        len(replay.rows), self.config.batch_size, replace=False
                                    )
                                ]
                                states, actions, rewards, next_states, done, masks = zip(
                                    *sample, strict=True
                                )
                                x = torch.tensor(np.stack(states), dtype=torch.float32)
                                nx = torch.tensor(np.stack(next_states), dtype=torch.float32)
                                with torch.no_grad():
                                    nq = (
                                        target(nx)
                                        .masked_fill(~torch.tensor(np.stack(masks)), -torch.inf)
                                        .max(dim=1)
                                        .values
                                    )
                                    expected = (
                                        torch.tensor(rewards, dtype=torch.float32)
                                        + self.config.gamma * (~torch.tensor(done)).float() * nq
                                    )
                                predicted = (
                                    network(x).gather(1, torch.tensor(actions)[:, None]).reshape(-1)
                                )
                                loss = nn.functional.smooth_l1_loss(predicted, expected)
                                if not torch.isfinite(loss):
                                    raise RuntimeError(
                                        "DQN produced nonfinite temporal-difference loss"
                                    )
                                optimizer.zero_grad()
                                cast(_Backward, loss).backward()
                                nn.utils.clip_grad_norm_(network.parameters(), 10.0)
                                optimizer.step()
                                losses.append(float(loss.detach()))
                                steps += 1
                                if steps % self.config.target_update_steps == 0:
                                    target.load_state_dict(network.state_dict())
                                    copies += 1
                        objectives.append(
                            env.result(self.policy_id).analytics.sim_internal_objective
                        )
                if not losses:
                    raise ValueError(
                        "training budget produced no optimizer step; reduce batch size or add episodes"
                    )
                weights = tuple(
                    np.asarray(p.detach().numpy(), dtype=np.float64).copy()
                    for p in network.parameters()
                )
                final = self._parameter_hash(network)
                versions = {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "device": "cpu",
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                }
        finally:
            torch.set_num_threads(old_threads)
        for weight in weights:
            weight.setflags(write=False)
        self._weights = weights
        self.training_trace = TrainingTrace(
            tuple(objectives), tuple(losses), seen, steps, copies, initial, final
        )
        self._artifact = {
            "revision": REVISION,
            "synthetic": True,
            "research_only": True,
            "live_claim": False,
            "fit_as_of": cutoff.isoformat(),
            "config": asdict(self.config),
            "source_sha256": _code_hash(),
            "environment": versions,
            "training_episode_ids": ids,
            "training_path_sha256": hashes,
            "training_scenario_sha256": [s.sha256 for s in training],
            "training_scenarios": [asdict(s) for s in training],
            "trace": asdict(self.training_trace),
            "limitations": _LIMITATIONS,
        }
        self._model_sha256 = _hash(self._payload())
        return self

    @staticmethod
    def _parameter_hash(network: Any) -> str:
        return _hash([p.detach().numpy().astype(float).tolist() for p in network.parameters()])

    def _payload(self) -> dict[str, Any]:
        if self._weights is None or self._artifact is None:
            raise ValueError("DQN must be actually fitted before use")
        return {**self._artifact, "weights": [w.tolist() for w in self._weights]}

    def _require_fit(self) -> dict[str, Any]:
        payload = self._payload()
        if _hash(payload) != self._model_sha256 or payload["config"] != asdict(self.config):
            raise ValueError("model parameters or provenance changed after fitting")
        if payload["source_sha256"] != _code_hash():
            raise ValueError("model source changed since fitting")
        return payload

    @property
    def model_sha256(self) -> str:
        self._require_fit()
        assert self._model_sha256 is not None
        return self._model_sha256

    def q_values(self, state: ExecutionState) -> Array:
        metadata = self._require_fit()
        if _utc(state.book.available_time) <= datetime.fromisoformat(metadata["fit_as_of"]):
            raise ValueError("prediction must follow the model's fit cutoff")
        assert self._weights is not None
        w1, b1, w2, b2 = self._weights
        return np.asarray(np.tanh(state.vector() @ w1.T + b1) @ w2.T + b2, dtype=float)

    def act(self, state: ExecutionState) -> TradeAction:
        values = self.q_values(state)
        return grid_action(state, int(np.argmax(np.where(state.action_mask(), values, -np.inf))))

    def save_artifact(self, path: Path) -> str:
        """Exclusive canonical JSON artifact: inspectable arrays, no pickle/code loader."""
        self._require_fit()
        blob = _json({"model_sha256": self.model_sha256, "payload": self._payload()})
        with path.open("xb") as stream:
            stream.write(blob)
        return hashlib.sha256(blob).hexdigest()

    @classmethod
    def load_artifact(cls, path: Path) -> ParentOrderDQN:
        """Validate plain JSON parameter/provenance identity before offline use.

        This checks content integrity, not authenticity of an unknown author.
        The declared source must equal this module; no arbitrary code executes.
        """
        if path.stat().st_size > 64_000_000:
            raise ValueError("model artifact exceeds 64 MB resource limit")
        return cls._from_record(json.loads(path.read_bytes()))

    @classmethod
    def _from_record(cls, record: Any) -> ParentOrderDQN:
        if not isinstance(record, dict) or set(record) != {"model_sha256", "payload"}:
            raise ValueError("invalid model artifact envelope")
        payload = record["payload"]
        if not isinstance(payload, dict) or _hash(payload) != record["model_sha256"]:
            raise ValueError("model artifact content hash mismatch")
        if (
            payload.get("revision") != REVISION
            or payload.get("synthetic") is not True
            or payload.get("research_only") is not True
            or payload.get("live_claim") is not False
            or payload.get("source_sha256") != _code_hash()
        ):
            raise ValueError("model artifact source or synthetic evidence contract mismatch")
        policy = cls(DQNConfig(**payload["config"]))
        _utc(datetime.fromisoformat(payload["fit_as_of"]))
        training = tuple(_scenario_from_json(s) for s in payload["training_scenarios"])
        if (
            not training
            or len(training) > 10_000
            or [s.tape.episode_id for s in training] != payload["training_episode_ids"]
            or [s.tape.path_sha256 for s in training] != payload["training_path_sha256"]
            or [s.sha256 for s in training] != payload["training_scenario_sha256"]
            or len(set(payload["training_path_sha256"])) != len(training)
        ):
            raise ValueError("artifact training provenance does not bind its source scenarios")
        if any(
            _utc(s.tape.events[-1].available_time) > datetime.fromisoformat(payload["fit_as_of"])
            for s in training
        ):
            raise ValueError("artifact training source was unavailable at fit cutoff")
        weights = tuple(np.asarray(w, dtype=np.float64) for w in payload["weights"])
        shapes = (
            (policy.config.hidden_dim, N_FEATURES),
            (policy.config.hidden_dim,),
            (N_ACTIONS, policy.config.hidden_dim),
            (N_ACTIONS,),
        )
        if len(weights) != len(shapes) or any(
            w.shape != shape or not np.isfinite(w).all()
            for w, shape in zip(weights, shapes, strict=True)
        ):
            raise ValueError("model artifact parameters have invalid shapes or values")
        trace = payload["trace"]
        policy.training_trace = TrainingTrace(
            tuple(trace["episode_objectives"]),
            tuple(trace["td_losses"]),
            trace["transitions"],
            trace["optimizer_steps"],
            trace["target_copies"],
            trace["initial_parameters_sha256"],
            trace["final_parameters_sha256"],
        )
        if policy.training_trace.optimizer_steps < 1 or not policy.training_trace.td_losses:
            raise ValueError("artifact declares no actual optimizer step")
        expected_transitions = sum(len(s.tape.events) for s in training) * policy.config.rounds
        if (
            policy.training_trace.transitions != expected_transitions
            or policy.training_trace.optimizer_steps
            != expected_transitions - policy.config.batch_size + 1
            or len(policy.training_trace.td_losses) != policy.training_trace.optimizer_steps
            or len(policy.training_trace.episode_objectives) != policy.config.rounds * len(training)
            or policy.training_trace.target_copies
            != policy.training_trace.optimizer_steps // policy.config.target_update_steps
        ):
            raise ValueError(
                "artifact training trace is inconsistent with its configured replay budget"
            )
        if _hash([w.tolist() for w in weights]) != policy.training_trace.final_parameters_sha256:
            raise ValueError("artifact trace does not bind the final parameters")
        for weight in weights:
            weight.setflags(write=False)
        policy._weights = weights
        policy._artifact = {k: v for k, v in payload.items() if k != "weights"}
        policy._model_sha256 = record["model_sha256"]
        policy._require_fit()
        return policy


_LIMITATIONS = (
    "SYNTHETIC simulator correctness, never empirical or live execution evidence",
    "exogenous event tape and supplied passive queue/auction allocations; no exchange priority fidelity",
    "linear temporary impact on market IOC only; no permanent impact or participant response",
    "filled-quantity shortfall excludes opportunity cost of unfilled shares; penalties are preferences",
    "one-event IOC limits and one final auction submission; no persistent/cancel-replace orders",
    "small finite-horizon DQN adaptation; no reproduction or SOTA claim",
)


@dataclass(frozen=True)
class ExecutionComparison:
    model_sha256: str
    source_sha256: str
    fit_as_of: str
    training_path_sha256: tuple[str, ...]
    scenarios: tuple[ExecutionScenario, ...]
    episodes: tuple[ExecutionEpisode, ...]
    baseline_config: tuple[tuple[str, float], ...]
    baseline_policy_id: str
    baseline_source_sha256: str
    model_artifact_json: str

    def receipt_payload(self) -> dict[str, Any]:
        """Recompute all ledgers before exposing measured simulated diagnostics."""
        model = json.loads(self.model_artifact_json)
        if (
            _hash(model) != self.model_sha256
            or model["source_sha256"] != self.source_sha256
            or model["fit_as_of"] != self.fit_as_of
            or tuple(model["training_path_sha256"]) != self.training_path_sha256
            or self.source_sha256 != _code_hash()
            or self.baseline_source_sha256 != _ac_code_hash()
        ):
            raise ValueError("comparison model or source identity mismatch")
        expected = {
            (s.tape.episode_id, p)
            for s in self.scenarios
            for p in ("parent_order_dqn", "twap_catchup", self.baseline_policy_id)
        }
        if (
            len(self.episodes) != len(expected)
            or {(e.episode_id, e.policy_id) for e in self.episodes} != expected
        ):
            raise ValueError("comparison must contain one matched episode per policy and scenario")
        if not self.scenarios or set(self.training_path_sha256) & {
            s.tape.path_sha256 for s in self.scenarios
        }:
            raise ValueError("comparison contains no evaluation or reuses a training path")
        if any(
            _utc(s.tape.observations[0].available_time) <= datetime.fromisoformat(self.fit_as_of)
            for s in self.scenarios
        ):
            raise ValueError("comparison evaluation precedes the model fit cutoff")
        loaded = ParentOrderDQN._from_record({"model_sha256": self.model_sha256, "payload": model})
        sigma, eta, risk = (
            dict(self.baseline_config)[k] for k in ("sigma", "eta", "risk_aversion")
        )
        policies: dict[str, ExecutionPolicy] = {
            "parent_order_dqn": loaded,
            "twap_catchup": TWAPPolicy(),
            self.baseline_policy_id: AlmgrenChrissPolicy(sigma, eta, risk, self.baseline_policy_id),
        }
        summaries: dict[str, dict[str, float]] = {}
        for policy_id in dict.fromkeys(e.policy_id for e in self.episodes):
            rows = [e for e in self.episodes if e.policy_id == policy_id]
            summaries[policy_id] = {
                "completion_fraction_mean": float(
                    np.mean([e.analytics.completion_fraction for e in rows])
                ),
                "complete_episode_fraction": float(
                    np.mean([e.analytics.unfilled_quantity <= 1e-8 for e in rows])
                ),
                "sim_internal_filled_shortfall_mean": float(
                    np.mean([e.analytics.sim_internal_filled_shortfall for e in rows])
                ),
                "sim_internal_terminal_preference_mean": float(
                    np.mean([e.analytics.sim_internal_terminal_preference for e in rows])
                ),
                "sim_internal_objective_mean": float(
                    np.mean([e.analytics.sim_internal_objective for e in rows])
                ),
                "decision_latency_p50_ms": float(
                    np.median([ns / 1e6 for e in rows for ns in e.decision_latency_ns])
                ),
                "decision_latency_p95_ms": float(
                    np.quantile([ns / 1e6 for e in rows for ns in e.decision_latency_ns], 0.95)
                ),
            }
        for result in self.episodes:
            scenario = next(s for s in self.scenarios if s.tape.episode_id == result.episode_id)
            verify_episode(scenario, result)
            if len(result.decision_latency_ns) != len(result.actions) or any(
                isinstance(t, bool) or not isinstance(t, int) or t < 0
                for t in result.decision_latency_ns
            ):
                raise ValueError("evaluation must retain measured decision timings")
            reproduced = simulate_episode(policies[result.policy_id], scenario)
            if (reproduced.actions, reproduced.fills, reproduced.remaining_path) != (
                result.actions,
                result.fills,
                result.remaining_path,
            ):
                raise ValueError(
                    "evaluated actions or fills do not reproduce from the declared model/baseline"
                )
        return {
            "revision": REVISION,
            "synthetic": True,
            "evidence_level": "SYNTHETIC_CORRECTNESS",
            "research_only": True,
            "live_claim": False,
            "promote": False,
            "sota_claim": False,
            "model_sha256": self.model_sha256,
            "source_sha256": self.source_sha256,
            "model_artifact": model,
            "baseline_source_sha256": self.baseline_source_sha256,
            "baseline_policy_id": self.baseline_policy_id,
            "fit_as_of": self.fit_as_of,
            "training_path_sha256": self.training_path_sha256,
            "evaluation_path_sha256": [s.tape.path_sha256 for s in self.scenarios],
            "evaluation_scenario_sha256": [s.sha256 for s in self.scenarios],
            "scenarios": [asdict(s) for s in self.scenarios],
            "episodes": [asdict(e) for e in self.episodes],
            "baseline_config": dict(self.baseline_config),
            "diagnostics": summaries,
            "paired_episode_differences": _paired_differences(
                self.episodes, self.baseline_policy_id
            ),
            "latency_scope": "single-process batch-one offline calls on the model artifact environment; no production load or <50ms claim",
            "limitations": _LIMITATIONS,
        }


def _paired_differences(episodes: Sequence[ExecutionEpisode], ac_id: str) -> dict[str, Any]:
    ids = tuple(dict.fromkeys(e.episode_id for e in episodes))
    lookup = {(e.episode_id, e.policy_id): e.analytics for e in episodes}
    rng = np.random.default_rng(7)
    indices = rng.integers(0, len(ids), size=(1000, len(ids)))
    result: dict[str, Any] = {
        "scope": "paired iid synthetic-episode bootstrap; independent generator paths are an assumption, not a market confidence interval",
        "seed": 7,
        "replicates": 1000,
        "episode_count": len(ids),
        "comparisons": {},
    }
    for baseline in ("twap_catchup", ac_id):
        metrics: dict[str, Any] = {}
        for name in (
            "sim_internal_filled_shortfall",
            "sim_internal_objective",
            "completion_fraction",
        ):
            values = np.asarray(
                [
                    getattr(lookup[(i, "parent_order_dqn")], name)
                    - getattr(lookup[(i, baseline)], name)
                    for i in ids
                ]
            )
            ci = (
                np.quantile(values[indices].mean(axis=1), (0.025, 0.975)).tolist()
                if len(ids) >= 2
                else None
            )
            metrics[name] = {
                "mean_difference": float(values.mean()),
                "episode_differences": values.tolist(),
                "ci95": ci,
            }
        result["comparisons"][baseline] = metrics
    return result


def verify_episode(scenario: ExecutionScenario, result: ExecutionEpisode) -> None:
    """Independent time/quantity/capacity/price/fee checks and ledger arithmetic."""
    order, tape, config = scenario.order, scenario.tape, scenario.config
    if result.scenario_sha256 != scenario.sha256 or result.episode_id != tape.episode_id:
        raise ValueError("episode source identity mismatch")
    if (
        len(result.actions) != len(tape.events)
        or len(result.remaining_path) != len(tape.events) + 1
    ):
        raise ValueError("episode must cover the entire declared horizon")
    if not math.isclose(result.remaining_path[0], order.quantity):
        raise ValueError("initial inventory mismatch")
    if len({f.event_index for f in result.fills}) != len(result.fills):
        raise ValueError("at most one fill per child/event")
    fills = {f.event_index: f for f in result.fills}
    if any(i < 0 or i >= len(tape.events) for i in fills):
        raise ValueError("fill event index outside tape")
    for i, (action, event) in enumerate(zip(result.actions, tape.events, strict=True)):
        before, after = result.remaining_path[i : i + 2]
        if not (0 <= after <= before <= order.quantity) or action.quantity > before + 1e-8:
            raise ValueError("inventory path or child quantity violates parent constraint")
        if _utc(action.decision_time) != _utc(tape.observations[i].available_time):
            raise ValueError("action publication alignment mismatch")
        final = i == len(tape.events) - 1
        if action.kind == "auction" and (
            not final or _utc(action.decision_time) > _utc(tape.auction_submit_deadline)
        ):
            raise ValueError("auction submission is invalid")
        if final and action.kind not in ("wait", "auction"):
            raise ValueError("closing event requires an auction action")
        fill = fills.get(i)
        if fill is None:
            if not math.isclose(before, after, abs_tol=1e-8):
                raise ValueError("inventory changed without a fill")
            continue
        if (
            fill.kind != action.kind
            or fill.kind == "wait"
            or fill.quantity > action.quantity + 1e-8
        ):
            raise ValueError("fill does not match submitted action")
        if (fill.decision_time, fill.event_time, fill.available_time) != (
            action.decision_time,
            event.event_time,
            event.available_time,
        ):
            raise ValueError("fill clocks do not match the event")
        buy = order.side == "buy"
        if action.kind == "auction":
            price, capacity = event.auction_price, event.auction_quantity
        elif action.kind == "limit":
            price = event.bid if buy else event.ask
            capacity = event.passive_buy_quantity if buy else event.passive_sell_quantity
            assert action.limit_price is not None
            if (buy and price > action.limit_price) or (not buy and price < action.limit_price):
                raise ValueError("limit fill violates submitted price")
        else:
            price = (
                event.ask if buy else event.bid
            ) + order.sign * config.market_impact_per_unit * fill.quantity
            capacity = event.ask_quantity if buy else event.bid_quantity
        if (
            price is None
            or not math.isclose(fill.price, price, abs_tol=1e-9)
            or not math.isclose(fill.fees, config.fee_per_unit * fill.quantity, abs_tol=1e-9)
            or not math.isclose(fill.liquidity_capacity, capacity, abs_tol=1e-9)
            or not math.isclose(
                fill.participation_capacity,
                order.max_participation * event.traded_volume,
                abs_tol=1e-9,
            )
            or not math.isclose(before - after, fill.quantity, abs_tol=1e-8)
        ):
            raise ValueError(
                "fill price, fees, capacity or inventory arithmetic disagrees with event"
            )
    inventory = order.inventory_penalty_per_step * sum(
        (q / order.quantity) ** 2 for q in result.remaining_path[1:]
    )
    recomputed = execution_analytics(
        order, result.fills, result.remaining_path[-1], inventory_preference=inventory
    )
    for name, value in asdict(recomputed).items():
        if not math.isclose(value, getattr(result.analytics, name), abs_tol=1e-7, rel_tol=1e-9):
            raise ValueError("reported episode analytics disagree with independent ledger")


def evaluate_agent(
    policy: ParentOrderDQN,
    scenarios: Sequence[ExecutionScenario],
    *,
    ac: AlmgrenChrissPolicy = AlmgrenChrissPolicy(),
) -> ExecutionComparison:
    metadata = policy._require_fit()
    if not ac.policy_id.strip() or ac.policy_id in (policy.policy_id, TWAPPolicy().policy_id):
        raise ValueError("baseline policy identity must be distinct and nonempty")
    evaluation = tuple(scenarios)
    if not evaluation or len(evaluation) > 10_000:
        raise ValueError("evaluation requires between 1 and 10000 synthetic scenarios")
    ids = [s.tape.episode_id for s in evaluation]
    hashes = [s.tape.path_sha256 for s in evaluation]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("evaluation identities and paths must be unique")
    if set(ids) & set(metadata["training_episode_ids"]) or set(hashes) & set(
        metadata["training_path_sha256"]
    ):
        raise ValueError("training and evaluation episodes or paths overlap")
    cutoff = datetime.fromisoformat(metadata["fit_as_of"])
    if any(_utc(s.tape.observations[0].available_time) <= cutoff for s in evaluation):
        raise ValueError("evaluation must start strictly after the model fit cutoff")
    ends = [_utc(s.tape.events[-1].available_time) for s in evaluation]
    if any(
        end >= _utc(nxt.tape.observations[0].available_time)
        for end, nxt in zip(ends, evaluation[1:], strict=False)
    ):
        raise ValueError("evaluation scenarios must be chronological and nonoverlapping")
    episodes = tuple(simulate_episode(p, s) for s in evaluation for p in (policy, TWAPPolicy(), ac))
    comparison = ExecutionComparison(
        policy.model_sha256,
        _code_hash(),
        metadata["fit_as_of"],
        tuple(metadata["training_path_sha256"]),
        evaluation,
        episodes,
        (("sigma", ac.sigma), ("eta", ac.eta), ("risk_aversion", ac.risk_aversion)),
        ac.policy_id,
        _ac_code_hash(),
        _json(policy._payload()).decode(),
    )
    comparison.receipt_payload()
    return comparison


def write_execution_receipt(path: Path, comparison: ExecutionComparison) -> str:
    """Write-once canonical content-bound simulated comparison receipt."""
    payload = comparison.receipt_payload()
    digest = _hash(payload)
    with path.open("xb") as stream:
        stream.write(_json({"receipt_sha256": digest, "payload": payload}))
    return digest


def _scenario_from_json(raw: dict[str, Any]) -> ExecutionScenario:
    def clocks(row: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
        return {
            k: datetime.fromisoformat(v) if k in names and isinstance(v, str) else v
            for k, v in row.items()
        }

    tape = raw["tape"]
    observations = tuple(
        BookSnapshot(**clocks(o, ("event_time", "available_time"))) for o in tape["observations"]
    )
    events = tuple(
        RealizedLiquidity(**clocks(e, ("event_time", "available_time"))) for e in tape["events"]
    )
    deadline = tape["auction_submit_deadline"]
    parsed = SyntheticExecutionTape(
        tape["episode_id"],
        tape["seed"],
        observations,
        events,
        datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline,
    )
    return ExecutionScenario(
        parsed, ParentOrder(**clocks(raw["order"], ("deadline",))), SimulatorConfig(**raw["config"])
    )


def verify_execution_receipt(path: Path) -> dict[str, Any]:
    """Verify the immutable hash, sources, model actions and independent ledgers.

    Original decision timing samples are checked for coverage and preserved;
    timing measurements cannot be reproduced deterministically. Retraining
    is deliberately separate from verification of the frozen policy artifact.
    """
    if path.stat().st_size > 128_000_000:
        raise ValueError("execution receipt exceeds the 128 MB resource limit")
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict) or set(record) != {"receipt_sha256", "payload"}:
        raise ValueError("invalid execution receipt envelope")
    payload = record["payload"]
    if not isinstance(payload, dict) or _hash(payload) != record["receipt_sha256"]:
        raise ValueError("execution receipt content hash mismatch")
    scenarios = tuple(_scenario_from_json(s) for s in payload["scenarios"])
    episodes = []
    for raw in payload["episodes"]:
        actions = tuple(
            TradeAction(**{**a, "decision_time": datetime.fromisoformat(a["decision_time"])})
            for a in raw["actions"]
        )
        parsed_fills: list[ExecutionFill] = []
        for f in raw["fills"]:
            fields: dict[str, Any] = {
                k: datetime.fromisoformat(v)
                if k in ("decision_time", "event_time", "available_time")
                else v
                for k, v in f.items()
            }
            parsed_fills.append(ExecutionFill(**fields))
        fills = tuple(parsed_fills)
        episodes.append(
            ExecutionEpisode(
                raw["episode_id"],
                raw["policy_id"],
                raw["scenario_sha256"],
                actions,
                fills,
                tuple(raw["remaining_path"]),
                EpisodeAnalytics(**raw["analytics"]),
                tuple(raw["decision_latency_ns"]),
            )
        )
    comparison = ExecutionComparison(
        payload["model_sha256"],
        payload["source_sha256"],
        payload["fit_as_of"],
        tuple(payload["training_path_sha256"]),
        scenarios,
        tuple(episodes),
        tuple(payload["baseline_config"].items()),
        payload["baseline_policy_id"],
        payload["baseline_source_sha256"],
        _json(payload["model_artifact"]).decode(),
    )
    reproduced = comparison.receipt_payload()
    if _json(reproduced) != _json(payload):
        raise ValueError(
            "execution receipt claims disagree with frozen policy replay and independent arithmetic"
        )
    return {"receipt_sha256": record["receipt_sha256"], "payload": reproduced}
