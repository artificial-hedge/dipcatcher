"""Exact directed maximum flow with a residual-reachable minimum-cut certificate.

Own Edmonds-Karp BFS augments shortest residual paths. Each source edge has its
own paired reverse arc, so parallel and antiparallel edges retain independent
capacities and source-row lineage. Self-loops remain in output with zero flow.
Zero-capacity edges are permitted. There are no lower bounds or edge costs.
BFS visits arcs in source-row insertion order; this selects one maximum flow,
without a uniqueness claim. Source and sink must be distinct declared vertices.

Binary64 capacities are scaled exactly to common integer units. Returned edge
flows satisfy capacities, intermediate-vertex conservation and source/sink
balance exactly. The vertices reachable from the source in the final positive
residual graph determine a cut whose capacity is checked against the flow value.
Exact numerator/denominator strings are authoritative; previews are rounded.
Bounds: 32 vertices, 256 source edges, 8192 augmentations, two million residual
arc examinations, capacities <=1e100 and 12000-bit exact serialization. A work
limit raises instead of returning an uncertified partial flow.
Reference: https://web.stanford.edu/class/archive/cs/cs161/cs161.1166/lectures/lecture17.pdf
"""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class Edge(InputModel):
    edge_id: Name
    source: Name
    target: Name
    capacity: float = Field(strict=True, ge=0, le=1e100)


class Input(InputModel):
    vertices: list[Name] = Field(min_length=2, max_length=32)
    edges: list[Edge] = Field(max_length=256)
    source: Name
    sink: Name

    @model_validator(mode="after")
    def graph_identity(self) -> Self:
        names = set(self.vertices)
        if len(names) != len(self.vertices):
            raise ValueError("vertices must have unique names")
        if self.source not in names or self.sink not in names or self.source == self.sink:
            raise ValueError("source and sink must be distinct declared vertices")
        if len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("source edge IDs must be unique")
        if any(edge.source not in names or edge.target not in names for edge in self.edges):
            raise ValueError("edge endpoints must name declared vertices")
        return self


class ExactNumber(OutputModel):
    numerator: str
    denominator: str
    value: float


class EdgeFlow(OutputModel):
    edge_row_index: int
    edge_id: str
    source_index: int
    target_index: int
    capacity: float
    flow: ExactNumber
    forward_residual_capacity: ExactNumber
    crosses_minimum_cut: bool


class Output(OutputModel):
    vertices: list[str]
    source_index: int
    sink_index: int
    maximum_flow: ExactNumber
    minimum_cut_capacity: ExactNumber
    source_side_vertex_indices: list[int]
    sink_side_vertex_indices: list[int]
    cut_edge_row_indices: list[int]
    edge_flows: list[EdgeFlow]
    vertex_net_outflows: list[ExactNumber]
    augmentation_count: int
    residual_arc_examinations: int
    exact_capacity_and_conservation_verified: Literal[True] = True
    exact_flow_cut_equality_verified: Literal[True] = True
    optimizer_uniqueness_verified: Literal[False] = False
    external_network_verified: Literal[False] = False


@dataclass
class _Arc:
    source: int
    target: int
    remaining: int
    reverse: int


def _exact(value: Fraction) -> ExactNumber:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("maximum-flow result exceeds 12000-bit exact budget")
    try:
        preview = float(value)
    except OverflowError as error:
        raise ValueError("maximum-flow result overflows binary64") from error
    if not isfinite(preview) or (value and preview == 0):
        raise ValueError("nonzero maximum-flow result is outside finite binary64 range")
    return ExactNumber(
        numerator=str(value.numerator), denominator=str(value.denominator), value=preview
    )


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.vertices)
    positions = {name: index for index, name in enumerate(request.vertices)}
    source, sink = positions[request.source], positions[request.sink]
    ratios = [edge.capacity.as_integer_ratio() for edge in request.edges]
    denominator = max((divisor for _, divisor in ratios), default=1)
    capacities = [numerator * (denominator // divisor) for numerator, divisor in ratios]
    arcs: list[_Arc] = []
    adjacency: list[list[int]] = [[] for _ in request.vertices]
    for row, edge in enumerate(request.edges):
        first, last = positions[edge.source], positions[edge.target]
        index = len(arcs)
        arcs.extend((_Arc(first, last, capacities[row], index + 1), _Arc(last, first, 0, index)))
        adjacency[first].append(index)
        adjacency[last].append(index + 1)
    examinations = augmentations = total = 0
    reachable: set[int] = set()
    while True:
        predecessor: list[int | None] = [None] * count
        reachable = {source}
        pending = deque([source])
        while pending and sink not in reachable:
            vertex = pending.popleft()
            for index in adjacency[vertex]:
                examinations += 1
                if examinations > 2_000_000:
                    raise ValueError("maximum flow exceeds two million residual arc examinations")
                arc = arcs[index]
                if arc.remaining > 0 and arc.target not in reachable:
                    reachable.add(arc.target)
                    predecessor[arc.target] = index
                    pending.append(arc.target)
                    if arc.target == sink:
                        break
        if sink not in reachable:
            break
        if augmentations >= 8_192:
            raise ValueError("maximum flow exceeds 8192 augmentations")
        path: list[int] = []
        vertex = sink
        while vertex != source:
            previous_arc = predecessor[vertex]
            if previous_arc is None:
                raise ValueError("residual path has no predecessor")
            path.append(previous_arc)
            vertex = arcs[previous_arc].source
        amount = min(arcs[index].remaining for index in path)
        if amount <= 0:
            raise ValueError("residual augmentation must be positive")
        for index in path:
            arcs[index].remaining -= amount
            arcs[arcs[index].reverse].remaining += amount
        total += amount
        augmentations += 1
    balances = [0] * count
    rows: list[EdgeFlow] = []
    cut_edges: list[int] = []
    cut_capacity = 0
    for index, edge in enumerate(request.edges):
        forward, reverse = arcs[2 * index], arcs[2 * index + 1]
        flow = reverse.remaining
        if not 0 <= flow <= capacities[index] or forward.remaining + flow != capacities[index]:
            raise ValueError("source-edge capacity certificate failed")
        balances[forward.source] += flow
        balances[forward.target] -= flow
        crosses = forward.source in reachable and forward.target not in reachable
        if crosses:
            cut_edges.append(index)
            cut_capacity += capacities[index]
        rows.append(
            EdgeFlow(
                edge_row_index=index,
                edge_id=edge.edge_id,
                source_index=forward.source,
                target_index=forward.target,
                capacity=edge.capacity,
                flow=_exact(Fraction(flow, denominator)),
                forward_residual_capacity=_exact(Fraction(forward.remaining, denominator)),
                crosses_minimum_cut=crosses,
            )
        )
    if (
        balances[source] != total
        or balances[sink] != -total
        or any(balance for vertex, balance in enumerate(balances) if vertex not in (source, sink))
    ):
        raise ValueError("exact flow conservation certificate failed")
    if cut_capacity != total:
        raise ValueError("exact maximum-flow/minimum-cut certificate failed")
    return Output(
        vertices=request.vertices,
        source_index=source,
        sink_index=sink,
        maximum_flow=_exact(Fraction(total, denominator)),
        minimum_cut_capacity=_exact(Fraction(cut_capacity, denominator)),
        source_side_vertex_indices=sorted(reachable),
        sink_side_vertex_indices=[vertex for vertex in range(count) if vertex not in reachable],
        cut_edge_row_indices=cut_edges,
        edge_flows=rows,
        vertex_net_outflows=[_exact(Fraction(balance, denominator)) for balance in balances],
        augmentation_count=augmentations,
        residual_arc_examinations=examinations,
    )


OPERATION = Operation(
    id="features.maximum_flow",
    kind="feature",
    description="Compute exact directed maximum flow with per-source-edge lineage, conservation and an equal-capacity residual minimum cut.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
