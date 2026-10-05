"""Audit declared capacity trees and instantaneous interval demand as of a clock.

Only decision-visible rows participate, including identity uniqueness. Future
duplicates cannot invalidate an earlier capacity or reservation. Resource nodes
have one parent, one unit and a fixed ceiling over [valid_from, valid_to). Each
child must share its parent's unit, fit inside its validity, and know its parent
when published. Child ceilings are limits, not exclusive allocations: their sum
need not fit the parent until actual simultaneous demand is declared.

Reservations name a resource and its declared owner, must be known by their start,
and fit the resource validity. Each reservation consumes its quantity once at that
node and every ancestor. Consumption intervals bind a reservation's owner, fit
inside that reservation and must be published at/after completion. Overlapping
consumptions are additive instantaneous quantities, not cumulative meter readings.
Consumption is checked against its reservation and independently against ancestor
capacities. Reserved and consumed demand are never added together.

All visible structural/reference declarations are audited. Quantity sweeps are
clipped to the explicit [audit_start, audit_end) window; no future usage is inferred.
Invalid declarations are excluded from aggregates and counted, preventing passed
or fully included declarations. visible_declarations_complete means all visible
rows were included, not that any external population is complete or capacity
limits pass. Overcommit does not remove otherwise valid demand.
These are caller-declared ownership and capacity facts; no scheduler allocation,
hardware measurement, actual authorization or real unit conversion is verified.

Bounds: 500 resources, 10000 reservations and 10000 consumption rows; at most 64
nodes per ancestry path and 500000 valid demand/ancestor propagations. Resource
ancestry inspection is bounded separately by 500 * 64. Endpoint sweeps do not
materialize time grids. Diagnostics/resource pages hold at most 200 rows, and each
excess span carries at most five source witnesses from its maximum-demand segment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Quantity = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]


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


class Resource(InputModel):
    resource_id: Name
    parent_resource_id: Name | None = None
    owner_id: Name
    unit: Name
    capacity: int = Field(strict=True, ge=0, le=2**63 - 1)
    valid_from: AwareDatetime
    valid_to: AwareDatetime
    available_time: AwareDatetime

    @field_validator("valid_from", "valid_to", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Reservation(InputModel):
    reservation_id: Name
    resource_id: Name
    owner_id: Name
    quantity: Quantity
    start: AwareDatetime
    end: AwareDatetime
    available_time: AwareDatetime

    @field_validator("start", "end", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Consumption(InputModel):
    consumption_id: Name
    reservation_id: Name
    owner_id: Name
    quantity: Quantity
    start: AwareDatetime
    end: AwareDatetime
    available_time: AwareDatetime

    @field_validator("start", "end", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    resources: list[Resource] = Field(max_length=500)
    reservations: list[Reservation] = Field(default_factory=list, max_length=10_000)
    consumptions: list[Consumption] = Field(default_factory=list, max_length=10_000)
    decision_time: AwareDatetime
    audit_start: AwareDatetime
    audit_end: AwareDatetime
    max_propagations: int = Field(default=100_000, strict=True, ge=0, le=500_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=500)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("decision_time", "audit_start", "audit_end", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def interval(self) -> Self:
        if self.audit_start >= self.audit_end:
            raise ValueError("audit window must be nonempty and forward")
        return self


class Finding(OutputModel):
    code: str
    resource_row_index: int | None = None
    other_resource_row_index: int | None = None
    reservation_row_index: int | None = None
    consumption_row_index: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    maximum_quantity: int | None = None
    allowed_quantity: int | None = None
    reservation_row_indexes: list[int] = Field(default_factory=list)
    consumption_row_indexes: list[int] = Field(default_factory=list)


class DemandSummary(OutputModel):
    maximum_quantity: int
    quantity_microseconds: int
    excess_duration_microseconds: int
    excess_quantity_microseconds: int
    excess_span_count: int


class ResourceResult(OutputModel):
    resource_row_index: int
    resource_id: str
    ancestry_valid: bool
    ancestor_resource_row_indexes: list[int]
    declared_capacity: int
    reservations: DemandSummary | None
    consumptions: DemandSummary | None
    maximum_source_available_time: datetime | None


class Output(OutputModel):
    passed: bool
    visible_declarations_complete: bool
    resource_count: int
    visible_resource_count: int
    valid_resource_count: int
    visible_reservation_count: int
    included_reservation_count: int
    excluded_reservation_count: int
    visible_consumption_count: int
    included_consumption_count: int
    excluded_consumption_count: int
    propagated_demand_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    resources: list[ResourceResult]
    offset: int
    has_more: bool
    demand_scope: Literal["explicit_audit_window_only"] = "explicit_audit_window_only"
    declared_ownership_verified: Literal[False] = False
    physical_consumption_verified: Literal[False] = False


@dataclass(frozen=True)
class _Demand:
    row: int
    start: datetime
    end: datetime
    quantity: int


_SpanSink = Callable[[datetime, datetime, int, list[int]], None]


def _sweep(
    demands: list[_Demand], capacity: int, start: datetime, end: datetime, emit: _SpanSink
) -> DemandSummary:
    boundaries: dict[datetime, list[tuple[int, int]]] = defaultdict(list)
    boundaries[start]
    boundaries[end]
    for demand in demands:
        left, right = max(start, demand.start), min(end, demand.end)
        if left < right:
            boundaries[left].append((demand.row, demand.quantity))
            boundaries[right].append((demand.row, -demand.quantity))
    active: dict[int, None] = {}
    quantity = maximum = area = duration = excess_area = spans = 0
    previous = start
    span_start: datetime | None = None
    span_end = start
    span_peak = 0
    witnesses: list[int] = []
    for point in sorted(boundaries):
        if point > previous:
            delta = point - previous
            micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
            maximum = max(maximum, quantity)
            area += quantity * micros
            if quantity > capacity:
                duration += micros
                excess_area += (quantity - capacity) * micros
                if span_start is None:
                    span_start, span_peak = previous, quantity
                    witnesses = list(islice(active, 5))
                elif quantity > span_peak:
                    span_peak = quantity
                    witnesses = list(islice(active, 5))
                span_end = point
            elif span_start is not None:
                emit(span_start, span_end, span_peak, witnesses)
                spans += 1
                span_start = None
        for row, change in boundaries[point]:
            quantity += change
            if change > 0:
                active[row] = None
            else:
                active.pop(row, None)
        previous = point
    if span_start is not None:
        emit(span_start, span_end, span_peak, witnesses)
        spans += 1
    return DemandSummary(
        maximum_quantity=maximum,
        quantity_microseconds=area,
        excess_duration_microseconds=duration,
        excess_quantity_microseconds=excess_area,
        excess_span_count=spans,
    )


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    diagnostics: list[Finding] = []

    def report(finding: Finding) -> None:
        issues[finding.code] += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    def span_sink(
        code: str,
        capacity: int,
        resource_index: int | None = None,
        reservation_index: int | None = None,
        consumption_witnesses: bool = False,
    ) -> _SpanSink:
        def emit(start: datetime, end: datetime, maximum: int, witnesses: list[int]) -> None:
            report(
                Finding(
                    code=code,
                    resource_row_index=resource_index,
                    reservation_row_index=reservation_index,
                    start=start,
                    end=end,
                    maximum_quantity=maximum,
                    allowed_quantity=capacity,
                    reservation_row_indexes=[] if consumption_witnesses else witnesses,
                    consumption_row_indexes=witnesses if consumption_witnesses else [],
                )
            )

        return emit

    resources: dict[str, list[int]] = defaultdict(list)
    visible_resources: list[int] = []
    for index, row in enumerate(request.resources):
        if row.available_time <= request.decision_time:
            resources[row.resource_id].append(index)
            visible_resources.append(index)
    unique_resources = {name: rows[0] for name, rows in resources.items() if len(rows) == 1}
    invalid_resources: set[int] = set()
    parents: dict[int, int | None] = {}
    for index in visible_resources:
        node = request.resources[index]
        code = None
        parent = None
        if len(resources[node.resource_id]) != 1:
            code = "ambiguous_resource_id"
        elif node.valid_from >= node.valid_to:
            code = "invalid_resource_window"
        elif node.parent_resource_id is not None:
            parent = unique_resources.get(node.parent_resource_id)
            if parent is None:
                code = "unresolved_resource_parent"
            else:
                parent_node = request.resources[parent]
                if node.unit != parent_node.unit:
                    code = "resource_unit_mismatch"
                elif (
                    node.valid_from < parent_node.valid_from or node.valid_to > parent_node.valid_to
                ):
                    code = "child_validity_outside_parent"
                elif node.available_time < parent_node.available_time:
                    code = "parent_not_available_at_child_publication"
        parents[index] = parent
        if code is not None:
            invalid_resources.add(index)
            report(Finding(code=code, resource_row_index=index, other_resource_row_index=parent))
    paths: dict[int, tuple[int, ...]] = {}
    for index in visible_resources:
        trail: list[int] = []
        seen: set[int] = set()
        current: int | None = index
        while current is not None:
            if current in seen or current in invalid_resources or len(trail) >= 64:
                break
            trail.append(current)
            seen.add(current)
            current = parents[current]
        if current is None:
            paths[index] = tuple(trail)
        elif index not in invalid_resources:
            report(
                Finding(
                    code="resource_ancestry_invalid_or_too_deep",
                    resource_row_index=index,
                    other_resource_row_index=current,
                )
            )

    reservations: dict[str, list[int]] = defaultdict(list)
    visible_reservations: list[int] = []
    for index, reserved in enumerate(request.reservations):
        if reserved.available_time <= request.decision_time:
            reservations[reserved.reservation_id].append(index)
            visible_reservations.append(index)
    unique_reservations = {name: rows[0] for name, rows in reservations.items() if len(rows) == 1}
    targets: dict[int, int] = {}
    for index in visible_reservations:
        reserved = request.reservations[index]
        target = unique_resources.get(reserved.resource_id)
        code = None
        if len(reservations[reserved.reservation_id]) != 1:
            code = "ambiguous_reservation_id"
        elif reserved.start >= reserved.end:
            code = "invalid_reservation_window"
        elif reserved.available_time > reserved.start:
            code = "reservation_published_after_start"
        elif target is None or target not in paths:
            code = "unresolved_reservation_resource"
        else:
            node = request.resources[target]
            if reserved.owner_id != node.owner_id:
                code = "reservation_owner_mismatch"
            elif node.available_time > reserved.available_time:
                code = "resource_not_available_at_reservation_publication"
            elif reserved.start < node.valid_from or reserved.end > node.valid_to:
                code = "reservation_outside_resource_validity"
        if code is not None:
            report(Finding(code=code, resource_row_index=target, reservation_row_index=index))
        else:
            assert target is not None
            targets[index] = target
    visible_consumptions: list[int] = []
    consumption_ids: Counter[str] = Counter()
    for index, consumed in enumerate(request.consumptions):
        if consumed.available_time <= request.decision_time:
            visible_consumptions.append(index)
            consumption_ids[consumed.consumption_id] += 1
    bindings: dict[int, int] = {}
    for index in visible_consumptions:
        consumed = request.consumptions[index]
        reservation = unique_reservations.get(consumed.reservation_id)
        code = None
        if consumption_ids[consumed.consumption_id] != 1:
            code = "ambiguous_consumption_id"
        elif consumed.start >= consumed.end:
            code = "invalid_consumption_window"
        elif consumed.available_time < consumed.end:
            code = "consumption_published_before_completion"
        elif reservation is None or reservation not in targets:
            code = "unresolved_consumption_reservation"
        else:
            parent_reservation = request.reservations[reservation]
            if consumed.owner_id != parent_reservation.owner_id:
                code = "consumption_owner_mismatch"
            elif parent_reservation.available_time > consumed.start:
                code = "reservation_not_available_at_consumption_start"
            elif consumed.start < parent_reservation.start or consumed.end > parent_reservation.end:
                code = "consumption_outside_reservation_window"
        if code is not None:
            report(
                Finding(code=code, reservation_row_index=reservation, consumption_row_index=index)
            )
        else:
            assert reservation is not None
            bindings[index] = reservation

    propagations = sum(len(paths[target]) for target in targets.values()) + sum(
        len(paths[targets[reservation]]) for reservation in bindings.values()
    )
    if propagations > request.max_propagations:
        raise ValueError("valid reservation/consumption ancestry work exceeds max_propagations")
    reserved_demands: dict[int, list[_Demand]] = defaultdict(list)
    consumed_demands: dict[int, list[_Demand]] = defaultdict(list)
    reservation_consumption: dict[int, list[_Demand]] = defaultdict(list)
    maxima = {
        index: max(request.resources[ancestor].available_time for ancestor in path)
        for index, path in paths.items()
    }
    for index, target in targets.items():
        reserved = request.reservations[index]
        demand = _Demand(index, reserved.start, reserved.end, reserved.quantity)
        for ancestor in paths[target]:
            reserved_demands[ancestor].append(demand)
            maxima[ancestor] = max(maxima[ancestor], reserved.available_time)
    for index, reservation in bindings.items():
        consumed = request.consumptions[index]
        demand = _Demand(index, consumed.start, consumed.end, consumed.quantity)
        reservation_consumption[reservation].append(demand)
        for ancestor in paths[targets[reservation]]:
            consumed_demands[ancestor].append(demand)
            maxima[ancestor] = max(maxima[ancestor], consumed.available_time)
    for reservation, demands in reservation_consumption.items():
        reserved = request.reservations[reservation]
        _sweep(
            demands,
            reserved.quantity,
            request.audit_start,
            request.audit_end,
            span_sink(
                "consumption_exceeds_reservation",
                reserved.quantity,
                reservation_index=reservation,
                consumption_witnesses=True,
            ),
        )
    results: list[ResourceResult] = []
    for position, index in enumerate(visible_resources):
        node = request.resources[index]
        reserved_summary = consumed_summary = None
        if index in paths:
            reserved_summary = _sweep(
                reserved_demands[index],
                node.capacity,
                request.audit_start,
                request.audit_end,
                span_sink("reservations_exceed_capacity", node.capacity, resource_index=index),
            )
            consumed_summary = _sweep(
                consumed_demands[index],
                node.capacity,
                request.audit_start,
                request.audit_end,
                span_sink(
                    "consumption_exceeds_capacity",
                    node.capacity,
                    resource_index=index,
                    consumption_witnesses=True,
                ),
            )
        if request.offset <= position < request.offset + request.limit:
            results.append(
                ResourceResult(
                    resource_row_index=index,
                    resource_id=node.resource_id,
                    ancestry_valid=index in paths,
                    ancestor_resource_row_indexes=list(paths.get(index, ()))[1:],
                    declared_capacity=node.capacity,
                    reservations=reserved_summary,
                    consumptions=consumed_summary,
                    maximum_source_available_time=maxima.get(index),
                )
            )
    complete = (
        len(paths) == len(visible_resources)
        and len(targets) == len(visible_reservations)
        and len(bindings) == len(visible_consumptions)
    )
    return Output(
        passed=not issues,
        visible_declarations_complete=complete,
        resource_count=len(request.resources),
        visible_resource_count=len(visible_resources),
        valid_resource_count=len(paths),
        visible_reservation_count=len(visible_reservations),
        included_reservation_count=len(targets),
        excluded_reservation_count=len(visible_reservations) - len(targets),
        visible_consumption_count=len(visible_consumptions),
        included_consumption_count=len(bindings),
        excluded_consumption_count=len(visible_consumptions) - len(bindings),
        propagated_demand_count=propagations,
        issue_counts=dict(sorted(issues.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=sum(issues.values()) - len(diagnostics),
        resources=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(visible_resources),
    )


OPERATION = Operation(
    id="skills.audit_resource_reservations",
    kind="skill",
    description="Audit visible capacity trees, declared ownership, reservation and consumption subwindows, and exact hierarchical overcommit without time-grid expansion.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
