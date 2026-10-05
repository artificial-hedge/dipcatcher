"""Check bounded clock-difference systems with exact integer microseconds.

Each constraint bounds to_clock - from_clock. An upper bound becomes a directed
edge from -> to; a lower bound becomes to -> from with negated weight. Timestamp
pins add two equality edges to an explicit reference whose value is the Unix
epoch. A negative cycle is a contradiction; its source inequalities are returned
in cycle order and their weights sum to a strictly negative integer.

Bellman-Ford starts every distance at zero, equivalent to zero-weight edges from
a synthetic super-source. It returns one feasible assignment, not estimated or
uniquely determined clocks. Only weak components containing a supplied pin have
absolute candidate clocks. Other components expose relative offsets only.

Difference-constraint construction and feasibility criterion: MIT 6.046J,
Lecture 11 (2015), pp. 5-6:
https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/
312f4a419009b58f8147b75975db4347_MIT6_046JS15_lec11.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Delay = Annotated[int, Field(strict=True, ge=-(10**18), le=10**18)]
Bound = Literal["maximum_delay", "minimum_delay", "pin_upper", "pin_lower"]
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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


def _epoch_microseconds(clock: datetime) -> int:
    delta = clock - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class Constraint(InputModel):
    from_clock: Name
    to_clock: Name
    min_delay_microseconds: Delay | None = None
    max_delay_microseconds: Delay | None = None

    @model_validator(mode="after")
    def at_least_one_bound(self) -> Self:
        if self.min_delay_microseconds is None and self.max_delay_microseconds is None:
            raise ValueError("a constraint must supply at least one delay bound")
        # Inverted bounds are intentional valid input to an infeasibility audit.
        return self


class Pin(InputModel):
    clock_id: Name
    time: AwareDatetime

    @field_validator("time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    clock_ids: list[Name] = Field(min_length=1, max_length=256)
    constraints: list[Constraint] = Field(max_length=2_048)
    pins: list[Pin] = Field(default_factory=list, max_length=256)
    max_edge_evaluations: int = Field(default=1_000_000, strict=True, ge=0, le=2_000_000)

    @model_validator(mode="after")
    def references_and_work_budget(self) -> Self:
        declared = set(self.clock_ids)
        if len(declared) != len(self.clock_ids):
            raise ValueError("clock_ids must be distinct")
        if any(
            row.from_clock not in declared or row.to_clock not in declared
            for row in self.constraints
        ) or any(row.clock_id not in declared for row in self.pins):
            raise ValueError("all constraint and pin clocks must be declared in clock_ids")
        edge_count = sum(
            (row.min_delay_microseconds is not None) + (row.max_delay_microseconds is not None)
            for row in self.constraints
        ) + 2 * len(self.pins)
        if (len(self.clock_ids) + bool(self.pins)) * edge_count > self.max_edge_evaluations:
            raise ValueError("vertex_count * edge_count exceeds max_edge_evaluations")
        return self


@dataclass(frozen=True)
class _Edge:
    source: int
    target: int
    weight: int
    source_kind: Literal["constraint", "pin"]
    source_row_index: int
    bound: Bound


class WitnessEdge(OutputModel):
    from_clock: str | None
    to_clock: str | None
    maximum_difference_microseconds: int
    source_kind: Literal["constraint", "pin"]
    source_row_index: int
    bound: Bound


class Assignment(OutputModel):
    clock_id: str
    component_index: int
    component_origin_clock: str
    relative_to_component_origin_microseconds: int
    status: Literal["unanchored", "anchored", "anchored_outside_datetime_range"]
    candidate_epoch_microseconds: int | None
    candidate_time: datetime | None


class Output(OutputModel):
    feasible: bool
    clock_count: int
    constraint_count: int
    pin_count: int
    vertex_count: int
    edge_count: int
    worst_case_edge_evaluations: int
    edge_evaluations: int
    relaxation_count: int
    completed_passes: int
    component_count: int
    anchored_component_count: int
    assignment_semantics: Literal["one_feasible_integer_assignment_not_estimated_clocks"] = (
        "one_feasible_integer_assignment_not_estimated_clocks"
    )
    unanchored_absolute_clocks: Literal["not_inferred"] = "not_inferred"
    witness_null_clock: Literal["pinned_unix_epoch_reference"] = "pinned_unix_epoch_reference"
    witness_edge_semantics: Literal["to_clock_minus_from_clock_at_most_weight"] = (
        "to_clock_minus_from_clock_at_most_weight"
    )
    negative_cycle_weight_microseconds: int | None
    negative_cycle: list[WitnessEdge]
    assignments: list[Assignment]


def _components(request: Input, indexes: dict[str, int]) -> tuple[list[int], list[int], set[int]]:
    adjacency: list[list[int]] = [[] for _ in request.clock_ids]
    for row in request.constraints:
        first, second = indexes[row.from_clock], indexes[row.to_clock]
        adjacency[first].append(second)
        adjacency[second].append(first)
    components = [-1] * len(request.clock_ids)
    origins: list[int] = []
    for origin in range(len(request.clock_ids)):
        if components[origin] != -1:
            continue
        component_index = len(origins)
        origins.append(origin)
        components[origin] = component_index
        pending = [origin]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if components[neighbor] == -1:
                    components[neighbor] = component_index
                    pending.append(neighbor)
    anchored = {components[indexes[pin.clock_id]] for pin in request.pins}
    return components, origins, anchored


def _negative_cycle(
    last_changed: int, predecessors: list[int | None], edges: list[_Edge], vertex_count: int
) -> list[_Edge]:
    cursor = last_changed
    for _ in range(vertex_count):
        predecessor = predecessors[cursor]
        if predecessor is None:
            raise RuntimeError("negative-cycle predecessor invariant failed")
        cursor = edges[predecessor].source
    start = cursor
    backward: list[_Edge] = []
    while True:
        predecessor = predecessors[cursor]
        if predecessor is None:
            raise RuntimeError("negative-cycle witness is missing a source edge")
        edge = edges[predecessor]
        backward.append(edge)
        cursor = edge.source
        if cursor == start:
            break
        if len(backward) >= vertex_count:
            raise RuntimeError("negative-cycle witness exceeds the vertex bound")
    cycle = list(reversed(backward))
    if sum(edge.weight for edge in cycle) >= 0:
        raise RuntimeError("negative-cycle witness has no negative total weight")
    return cycle


def execute(request: Input, context: OperationContext) -> Output:
    indexes = {name: index for index, name in enumerate(request.clock_ids)}
    reference = len(request.clock_ids)
    vertex_count = reference + bool(request.pins)
    edges: list[_Edge] = []
    for index, row in enumerate(request.constraints):
        first, second = indexes[row.from_clock], indexes[row.to_clock]
        if row.max_delay_microseconds is not None:
            edges.append(
                _Edge(
                    first, second, row.max_delay_microseconds, "constraint", index, "maximum_delay"
                )
            )
        if row.min_delay_microseconds is not None:
            edges.append(
                _Edge(
                    second, first, -row.min_delay_microseconds, "constraint", index, "minimum_delay"
                )
            )
    for index, pin in enumerate(request.pins):
        clock_index = indexes[pin.clock_id]
        pinned_value = _epoch_microseconds(pin.time)
        edges.append(_Edge(reference, clock_index, pinned_value, "pin", index, "pin_upper"))
        edges.append(_Edge(clock_index, reference, -pinned_value, "pin", index, "pin_lower"))

    distances = [0] * vertex_count
    predecessors: list[int | None] = [None] * vertex_count
    edge_evaluations = 0
    relaxation_count = 0
    passes = 0
    last_changed: int | None = None
    for _ in range(vertex_count):
        last_changed = None
        for edge_index, edge in enumerate(edges):
            edge_evaluations += 1
            candidate = distances[edge.source] + edge.weight
            if candidate < distances[edge.target]:
                distances[edge.target] = candidate
                predecessors[edge.target] = edge_index
                last_changed = edge.target
                relaxation_count += 1
        passes += 1
        if last_changed is None:
            break

    components, origins, anchored = _components(request, indexes)
    witness: list[WitnessEdge] = []
    cycle_weight: int | None = None
    assignments: list[Assignment] = []
    if last_changed is not None:
        cycle = _negative_cycle(last_changed, predecessors, edges, vertex_count)
        cycle_weight = sum(edge.weight for edge in cycle)
        witness = [
            WitnessEdge(
                from_clock=request.clock_ids[edge.source] if edge.source < reference else None,
                to_clock=request.clock_ids[edge.target] if edge.target < reference else None,
                maximum_difference_microseconds=edge.weight,
                source_kind=edge.source_kind,
                source_row_index=edge.source_row_index,
                bound=edge.bound,
            )
            for edge in cycle
        ]
    else:
        for clock_index, name in enumerate(request.clock_ids):
            component = components[clock_index]
            origin = origins[component]
            epoch_microseconds: int | None = None
            candidate_time: datetime | None = None
            status: Literal["unanchored", "anchored", "anchored_outside_datetime_range"] = (
                "unanchored"
            )
            if component in anchored:
                epoch_microseconds = distances[clock_index] - distances[reference]
                try:
                    candidate_time = _EPOCH + timedelta(microseconds=epoch_microseconds)
                    status = "anchored"
                except OverflowError:
                    status = "anchored_outside_datetime_range"
            assignments.append(
                Assignment(
                    clock_id=name,
                    component_index=component,
                    component_origin_clock=request.clock_ids[origin],
                    relative_to_component_origin_microseconds=(
                        distances[clock_index] - distances[origin]
                    ),
                    status=status,
                    candidate_epoch_microseconds=epoch_microseconds,
                    candidate_time=candidate_time,
                )
            )
    return Output(
        feasible=last_changed is None,
        clock_count=len(request.clock_ids),
        constraint_count=len(request.constraints),
        pin_count=len(request.pins),
        vertex_count=vertex_count,
        edge_count=len(edges),
        worst_case_edge_evaluations=vertex_count * len(edges),
        edge_evaluations=edge_evaluations,
        relaxation_count=relaxation_count,
        completed_passes=passes,
        component_count=len(origins),
        anchored_component_count=len(anchored),
        negative_cycle_weight_microseconds=cycle_weight,
        negative_cycle=witness,
        assignments=assignments,
    )


OPERATION = Operation(
    id="skills.audit_temporal_constraints",
    kind="skill",
    description=(
        "Solve bounded min/max clock differences exactly in integer microseconds, with "
        "source-edge negative-cycle witnesses, optional timestamp pins, and unanchored offsets."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
