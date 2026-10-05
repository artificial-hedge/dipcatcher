"""Replay a declared order-level log from an explicitly empty initial book.

This is a local model, not an exchange protocol. Each view uses only rows whose
available_time is at or before its decision, optionally capped by through_sequence.
Sequence numbers must be a contiguous unique prefix from first_sequence, and
event clocks must be nondecreasing and no later than publication. Future rows
cannot create duplicate IDs or sequence faults in an earlier view. A missing,
ambiguous or invalid transition stops replay; the returned book is the last valid
prefix and is explicitly marked blocked. Nothing after a fault is applied.

Add introduces a never-before-used order ID. Reduce/execute subtract positive
quantity, removing zero balances; delete removes the entire remaining order.
Replace keeps the order ID and side, sets price/remaining quantity and always
loses FIFO priority, even at unchanged price. Execute either requires the oldest
order at its price or targets any named order, as explicitly requested. It never
matches another order automatically. Crossed/locked books are reported, not
silently repaired or rejected: no venue matching rules are inferred.

Without through_sequence, observed_prefix certifies only replay of supplied
visible events, not external feed completeness. Integer ticks have no inferred
currency/tick size; quantities have no inferred lot multiplier. No broker calls,
market authenticity, trade settlement or executable liquidity are established.

Bounds: 10000 events, 16 snapshot queries, at most 160000 declared replay rows
(the conservative event_count * query_count charge includes invisible rows),
200 diagnostics and 50 orders/price levels per snapshot page. Sorting costs
O(Q N log N); transitions and FIFO bookkeeping are O(1) expected per applied row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Positive = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
Sequence = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Side = Literal["bid", "ask"]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("clock must be an explicit timezone-aware ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Event(InputModel):
    sequence: Sequence
    event_time: AwareDatetime
    available_time: AwareDatetime
    action: Literal["add", "reduce", "replace", "execute", "delete"]
    order_id: Name
    side: Side | None = None
    price_ticks: Positive | None = None
    quantity: Positive | None = None

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def action_fields(self) -> Self:
        if self.action == "add":
            if self.side is None or self.price_ticks is None or self.quantity is None:
                raise ValueError("add requires side, price_ticks and quantity")
        elif self.action == "replace":
            if self.side is not None or self.price_ticks is None or self.quantity is None:
                raise ValueError("replace requires price_ticks/quantity and cannot change side")
        elif self.side is not None or self.price_ticks is not None:
            raise ValueError("only add/replace may declare side or price")
        elif (self.action == "delete") != (self.quantity is None):
            raise ValueError("reduce/execute require quantity; delete cannot carry quantity")
        return self


class SnapshotQuery(InputModel):
    decision_time: AwareDatetime
    through_sequence: Sequence | None = None

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    stream_id: Name
    events: list[Event] = Field(max_length=10_000)
    snapshots: list[SnapshotQuery] = Field(min_length=1, max_length=16)
    first_sequence: Sequence = 0
    execution_policy: Literal["fifo_at_price", "named_order"] = "fifo_at_price"
    max_replay_rows: int = Field(default=100_000, strict=True, ge=0, le=160_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    order_offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    order_limit: int = Field(default=25, strict=True, ge=1, le=50)
    level_offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    level_limit: int = Field(default=25, strict=True, ge=1, le=50)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if len(self.events) * len(self.snapshots) > self.max_replay_rows:
            raise ValueError("event_count * snapshot_count exceeds max_replay_rows")
        if any(
            row.through_sequence is not None and row.through_sequence < self.first_sequence
            for row in self.snapshots
        ):
            raise ValueError("through_sequence cannot precede first_sequence")
        return self


class Finding(OutputModel):
    snapshot_row_index: int
    code: str
    expected_sequence: int
    event_row_indexes: list[int]


class OrderResult(OutputModel):
    order_id: str
    side: Side
    price_ticks: int
    remaining_quantity: int
    original_add_row_index: int
    priority_event_row_index: int
    last_event_row_index: int
    priority_sequence: int
    fifo_position_at_price: int


class PriceLevel(OutputModel):
    side: Side
    price_ticks: int
    total_quantity: int
    order_count: int
    first_order_id: str
    first_priority_event_row_index: int


class SnapshotResult(OutputModel):
    snapshot_row_index: int
    status: Literal["observed_prefix", "complete_requested_prefix", "blocked"]
    visible_selected_row_count: int
    applied_row_count: int
    last_applied_sequence: int | None
    last_applied_event_time: datetime | None
    maximum_applied_available_time: datetime | None
    blocked_at_sequence: int | None
    active_order_count: int
    price_level_count: int
    bid_quantity: int
    ask_quantity: int
    best_bid_ticks: int | None
    best_ask_ticks: int | None
    crossed_or_locked: bool
    executed_quantity: int
    reduced_quantity: int
    deleted_quantity: int
    orders: list[OrderResult]
    order_offset: int
    orders_have_more: bool
    levels: list[PriceLevel]
    level_offset: int
    levels_have_more: bool


class Output(OutputModel):
    passed: bool
    stream_id: str
    execution_policy: Literal["fifo_at_price", "named_order"]
    declared_replay_rows: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    snapshots: list[SnapshotResult]
    initial_state: Literal["declared_empty"] = "declared_empty"
    external_feed_completeness_verified: Literal[False] = False
    venue_protocol_verified: Literal[False] = False


@dataclass
class _Order:
    side: Side
    price: int
    quantity: int
    original: int
    priority: int
    last: int
    sequence: int


def _remove_order(
    order_id: str,
    orders: dict[str, _Order],
    queues: dict[tuple[Side, int], dict[str, None]],
    totals: Counter[tuple[Side, int]],
) -> None:
    order = orders.pop(order_id)
    key = (order.side, order.price)
    totals[key] -= order.quantity
    del queues[key][order_id]
    if not queues[key]:
        del queues[key]
        del totals[key]


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    diagnostics: list[Finding] = []
    results: list[SnapshotResult] = []

    def fail(code: str, rows: list[int], expected: int, query_index: int) -> int:
        issues[code] += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Finding(
                    snapshot_row_index=query_index,
                    code=code,
                    expected_sequence=expected,
                    event_row_indexes=rows[:2],
                )
            )
        return expected

    for query_index, query in enumerate(request.snapshots):
        groups: dict[int, list[int]] = defaultdict(list)
        for index, event in enumerate(request.events):
            if event.available_time <= query.decision_time and (
                query.through_sequence is None or event.sequence <= query.through_sequence
            ):
                groups[event.sequence].append(index)
        orders: dict[str, _Order] = {}
        queues: dict[tuple[Side, int], dict[str, None]] = {}
        totals: Counter[tuple[Side, int]] = Counter()
        seen: set[str] = set()
        expected = request.first_sequence
        applied = executed = reduced = deleted = 0
        last_time: datetime | None = None
        maximum: datetime | None = None
        blocked: int | None = None

        for sequence in sorted(groups):
            rows = groups[sequence]
            if sequence != expected:
                blocked = fail("sequence_gap_or_before_start", rows, expected, query_index)
                break
            if len(rows) != 1:
                blocked = fail("ambiguous_sequence", rows, expected, query_index)
                break
            index = rows[0]
            event = request.events[index]
            if event.event_time > event.available_time:
                blocked = fail("publication_precedes_event", rows, expected, query_index)
                break
            if last_time is not None and event.event_time < last_time:
                blocked = fail("event_clock_regression", rows, expected, query_index)
                break
            old = orders.get(event.order_id)
            if event.action == "add":
                if event.order_id in seen:
                    blocked = fail("reused_order_id", rows, expected, query_index)
                    break
                assert event.side is not None and event.price_ticks is not None
                assert event.quantity is not None
                order = _Order(
                    event.side, event.price_ticks, event.quantity, index, index, index, sequence
                )
                orders[event.order_id] = order
                seen.add(event.order_id)
                queues.setdefault((order.side, order.price), {})[event.order_id] = None
                totals[order.side, order.price] += order.quantity
            elif old is None:
                blocked = fail("unknown_or_inactive_order", rows, expected, query_index)
                break
            elif event.action == "replace":
                assert event.price_ticks is not None and event.quantity is not None
                replacement = _Order(
                    old.side,
                    event.price_ticks,
                    event.quantity,
                    old.original,
                    index,
                    index,
                    sequence,
                )
                _remove_order(event.order_id, orders, queues, totals)
                orders[event.order_id] = replacement
                queues.setdefault((replacement.side, replacement.price), {})[event.order_id] = None
                totals[replacement.side, replacement.price] += replacement.quantity
            elif event.action == "delete":
                deleted += old.quantity
                _remove_order(event.order_id, orders, queues, totals)
            else:
                assert event.quantity is not None
                key = (old.side, old.price)
                if event.quantity > old.quantity:
                    blocked = fail("quantity_exceeds_remaining", rows, expected, query_index)
                    break
                if (
                    event.action == "execute"
                    and request.execution_policy == "fifo_at_price"
                    and next(iter(queues[key])) != event.order_id
                ):
                    blocked = fail("execution_violates_fifo_at_price", rows, expected, query_index)
                    break
                if event.action == "execute":
                    executed += event.quantity
                else:
                    reduced += event.quantity
                if event.quantity == old.quantity:
                    _remove_order(event.order_id, orders, queues, totals)
                else:
                    old.quantity -= event.quantity
                    totals[key] -= event.quantity
                    old.last = index
            applied += 1
            expected += 1
            last_time = event.event_time
            maximum = (
                event.available_time if maximum is None else max(maximum, event.available_time)
            )
        if (
            blocked is None
            and query.through_sequence is not None
            and expected <= query.through_sequence
        ):
            blocked = fail("requested_suffix_unavailable", [], expected, query_index)
        keys = sorted(
            queues, key=lambda key: (key[0] == "ask", -key[1] if key[0] == "bid" else key[1])
        )
        order_results: list[OrderResult] = []
        position = 0
        for key in keys:
            for fifo, order_id in enumerate(queues[key]):
                if request.order_offset <= position < request.order_offset + request.order_limit:
                    order = orders[order_id]
                    order_results.append(
                        OrderResult(
                            order_id=order_id,
                            side=order.side,
                            price_ticks=order.price,
                            remaining_quantity=order.quantity,
                            original_add_row_index=order.original,
                            priority_event_row_index=order.priority,
                            last_event_row_index=order.last,
                            priority_sequence=order.sequence,
                            fifo_position_at_price=fifo,
                        )
                    )
                position += 1
        levels: list[PriceLevel] = []
        for key in keys[request.level_offset : request.level_offset + request.level_limit]:
            first = next(iter(queues[key]))
            levels.append(
                PriceLevel(
                    side=key[0],
                    price_ticks=key[1],
                    total_quantity=totals[key],
                    order_count=len(queues[key]),
                    first_order_id=first,
                    first_priority_event_row_index=orders[first].priority,
                )
            )
        best_bid = max((price for side, price in keys if side == "bid"), default=None)
        best_ask = min((price for side, price in keys if side == "ask"), default=None)
        results.append(
            SnapshotResult(
                snapshot_row_index=query_index,
                status="blocked"
                if blocked is not None
                else "observed_prefix"
                if query.through_sequence is None
                else "complete_requested_prefix",
                visible_selected_row_count=sum(map(len, groups.values())),
                applied_row_count=applied,
                last_applied_sequence=None if applied == 0 else expected - 1,
                last_applied_event_time=last_time,
                maximum_applied_available_time=maximum,
                blocked_at_sequence=blocked,
                active_order_count=len(orders),
                price_level_count=len(queues),
                bid_quantity=sum(value for (side, _), value in totals.items() if side == "bid"),
                ask_quantity=sum(value for (side, _), value in totals.items() if side == "ask"),
                best_bid_ticks=best_bid,
                best_ask_ticks=best_ask,
                crossed_or_locked=best_bid is not None
                and best_ask is not None
                and best_bid >= best_ask,
                executed_quantity=executed,
                reduced_quantity=reduced,
                deleted_quantity=deleted,
                orders=order_results,
                order_offset=request.order_offset,
                orders_have_more=request.order_offset + len(order_results) < len(orders),
                levels=levels,
                level_offset=request.level_offset,
                levels_have_more=request.level_offset + len(levels) < len(queues),
            )
        )
    return Output(
        passed=not issues,
        stream_id=request.stream_id,
        execution_policy=request.execution_policy,
        declared_replay_rows=len(request.events) * len(request.snapshots),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=sum(issues.values()) - len(diagnostics),
        snapshots=results,
    )


OPERATION = Operation(
    id="skills.reconstruct_order_book",
    kind="skill",
    description="Replay an explicit integer order log into bounded FIFO/price snapshots, stopping at the first visible sequence or transition fault.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
