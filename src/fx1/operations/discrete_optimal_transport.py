"""Balanced discrete transport by exact residual-network augmentation.

Supplied source/target masses are normalized separately. Costs are arbitrary
finite nonnegative labels, not necessarily a metric. Integer capacities and
integer costs are obtained without rounding from exact binary-float fractions.
Own Bellman-Ford shortest augmenting paths allow reverse arcs to reroute mass.
Final super-source potentials certify feasibility, complementary slackness and
zero primal/dual gap exactly. No entropy regularization or greedy shortcut.
Reference: https://courses.csail.mit.edu/6.854/21/Scribe/s9-minCostFlow/s9-minCostFlow.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Mass = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Cost = Annotated[float, Field(strict=True, ge=0, le=1e100)]


class Input(InputModel):
    source_masses: list[Mass] = Field(min_length=1, max_length=16)
    target_masses: list[Mass] = Field(min_length=1, max_length=16)
    costs: list[list[Cost]] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if not any(self.source_masses) or not any(self.target_masses):
            raise ValueError("both source and target must have positive total mass")
        if len(self.costs) != len(self.source_masses) or any(
            len(row) != len(self.target_masses) for row in self.costs
        ):
            raise ValueError("costs must have one row per source and one column per target")
        return self


class ExactNumber(OutputModel):
    numerator: str
    denominator: str
    value: float
    preview_underflow: bool


class Flow(OutputModel):
    source_index: int
    target_index: int
    mass: ExactNumber
    cost: float


class Output(OutputModel):
    source_probabilities: list[ExactNumber]
    target_probabilities: list[ExactNumber]
    positive_flows: list[Flow]
    minimum_cost: ExactNumber
    source_dual_potentials: list[ExactNumber]
    target_dual_potentials: list[ExactNumber]
    dual_objective: ExactNumber
    exact_primal_dual_gap: Literal["0"] = "0"
    exact_marginals_satisfied: Literal[True] = True
    exact_complementary_slackness: Literal[True] = True
    augmentation_count: int
    edge_examinations: int
    mass_convention: Literal["separately_normalized_supplied_masses"] = (
        "separately_normalized_supplied_masses"
    )
    tie_convention: Literal["strict_distance_improvement_in_stable_edge_order"] = (
        "strict_distance_improvement_in_stable_edge_order"
    )
    optimizer_uniqueness_verified: Literal[False] = False
    cost_metric_verified: Literal[False] = False
    availability_validated: Literal[False] = False


@dataclass
class _Edge:
    first: int
    last: int
    capacity: int
    cost: int
    reverse: int


def _bounded_integer(value: int) -> int:
    if value.bit_length() > 12_000:
        raise ValueError("calculation exceeds the 12000-bit integer arithmetic budget")
    return value


def _number(value: Fraction) -> ExactNumber:
    _bounded_integer(value.numerator)
    _bounded_integer(value.denominator)
    try:
        preview = float(value)
    except OverflowError as error:
        raise ValueError("result exceeds finite binary64 range") from error
    if not math.isfinite(preview):
        raise ValueError("result exceeds finite binary64 range")
    return ExactNumber(
        numerator=str(value.numerator),
        denominator=str(value.denominator),
        value=preview,
        preview_underflow=bool(value) and preview == 0,
    )


def _add(edges: list[_Edge], first: int, last: int, capacity: int, cost: int) -> int:
    index = len(edges)
    edges.append(_Edge(first, last, capacity, cost, index + 1))
    edges.append(_Edge(last, first, 0, -cost, index))
    return index


def _shortest(
    edges: list[_Edge], nodes: int, start: int | None, work: list[int]
) -> tuple[list[int | None], list[int | None]]:
    distance: list[int | None] = [None] * nodes if start is not None else [0] * nodes
    if start is not None:
        distance[start] = 0
    parent: list[int | None] = [None] * nodes
    # An extra pass explicitly rejects a negative residual cycle; it never
    # substitutes a capped, uncertified solution for an optimum.
    for iteration in range(nodes):
        changed = False
        for index, edge in enumerate(edges):
            work[0] += 1
            if work[0] > 2_000_000:
                raise ValueError("transport exceeds the 2000000 edge-examination budget")
            prefix = distance[edge.first]
            if edge.capacity <= 0 or prefix is None:
                continue
            candidate = _bounded_integer(prefix + edge.cost)
            previous = distance[edge.last]
            if previous is None or candidate < previous:
                if iteration == nodes - 1:
                    raise ArithmeticError(
                        "negative residual cycle prevents an optimality certificate"
                    )
                distance[edge.last] = candidate
                parent[edge.last] = index
                changed = True
        if not changed:
            break
    return distance, parent


def execute(request: Input, context: OperationContext) -> Output:
    source = [Fraction(value) for value in request.source_masses]
    target = [Fraction(value) for value in request.target_masses]
    source_total, target_total = sum(source, Fraction()), sum(target, Fraction())
    source = [value / source_total for value in source]
    target = [value / target_total for value in target]
    mass_denominator = _bounded_integer(math.lcm(*(value.denominator for value in source + target)))
    supplies = [value.numerator * (mass_denominator // value.denominator) for value in source]
    demands = [value.numerator * (mass_denominator // value.denominator) for value in target]
    costs = [[Fraction(value) for value in row] for row in request.costs]
    cost_denominator = max(value.denominator for row in costs for value in row)
    scaled_costs = [
        [
            _bounded_integer(value.numerator * (cost_denominator // value.denominator))
            for value in row
        ]
        for row in costs
    ]
    n, m = len(source), len(target)
    start, sink, nodes = n + m, n + m + 1, n + m + 2
    edges: list[_Edge] = []
    for index, supply in enumerate(supplies):
        _add(edges, start, index, supply, 0)
    transport_edges: list[list[int]] = []
    for index in range(n):
        transport_edges.append(
            [
                _add(edges, index, n + column, mass_denominator + 1, scaled_costs[index][column])
                for column in range(m)
            ]
        )
    for column, demand in enumerate(demands):
        _add(edges, n + column, sink, demand, 0)
    remaining, augmentations = mass_denominator, 0
    work = [0]
    while remaining:
        if augmentations >= 2048:
            raise ValueError("transport exceeds the 2048 augmentation budget")
        distance, parent = _shortest(edges, nodes, start, work)
        if distance[sink] is None:
            raise ArithmeticError(
                "complete transport graph unexpectedly has no feasible augmenting path"
            )
        path: list[int] = []
        node = sink
        seen: set[int] = set()
        while node != start:
            if node in seen or parent[node] is None:
                raise ArithmeticError("invalid shortest-path predecessor chain")
            seen.add(node)
            edge_index = parent[node]
            assert edge_index is not None
            path.append(edge_index)
            node = edges[edge_index].first
        amount = min(remaining, *(edges[index].capacity for index in path))
        if amount <= 0:
            raise ArithmeticError("augmenting path must carry strictly positive mass")
        for index in path:
            edges[index].capacity -= amount
            edges[edges[index].reverse].capacity += amount
        remaining -= amount
        augmentations += 1
    potentials, _ = _shortest(edges, nodes, None, work)
    dual_source = [Fraction(-int(potentials[index] or 0), cost_denominator) for index in range(n)]
    dual_target = [
        Fraction(int(potentials[n + column] or 0), cost_denominator) for column in range(m)
    ]
    row_sums, column_sums = [0] * n, [0] * m
    integer_objective = 0
    flows: list[Flow] = []
    for index in range(n):
        for column in range(m):
            edge = edges[transport_edges[index][column]]
            amount = edges[edge.reverse].capacity
            row_sums[index] += amount
            column_sums[column] += amount
            slack = costs[index][column] - dual_source[index] - dual_target[column]
            if slack < 0 or (amount and slack != 0):
                raise ArithmeticError("exact dual feasibility or complementary slackness failed")
            integer_objective = _bounded_integer(integer_objective + amount * edge.cost)
            if amount:
                flows.append(
                    Flow(
                        source_index=index,
                        target_index=column,
                        mass=_number(Fraction(amount, mass_denominator)),
                        cost=request.costs[index][column],
                    )
                )
    if row_sums != supplies or column_sums != demands:
        raise ArithmeticError("transport marginals do not match the exact requested masses")
    objective = Fraction(integer_objective, mass_denominator * cost_denominator)
    dual = sum((mass * value for mass, value in zip(source, dual_source, strict=True)), Fraction())
    dual += sum((mass * value for mass, value in zip(target, dual_target, strict=True)), Fraction())
    if objective != dual:
        raise ArithmeticError("exact primal and dual transport objectives disagree")
    return Output(
        source_probabilities=[_number(value) for value in source],
        target_probabilities=[_number(value) for value in target],
        positive_flows=flows,
        minimum_cost=_number(objective),
        source_dual_potentials=[_number(value) for value in dual_source],
        target_dual_potentials=[_number(value) for value in dual_target],
        dual_objective=_number(dual),
        augmentation_count=augmentations,
        edge_examinations=work[0],
    )


OPERATION = Operation(
    id="features.discrete_optimal_transport",
    kind="feature",
    description="Solve a bounded balanced discrete transport problem by exact residual-network augmentation, returning sparse flows and exact marginal, dual-feasibility and zero-gap certificates.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
