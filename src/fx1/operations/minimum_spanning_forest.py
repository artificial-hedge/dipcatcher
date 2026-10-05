"""Stable Kruskal forests with exact totals and fundamental-cycle diagnostics.

Undirected source edges remain distinct, including parallel edges. Kruskal
processes (binary64 weight, source row index) order using an own union-find.
Signed weights are permitted; self-loops are always excluded. Isolated vertices
are singleton components whose forest has zero edges and zero total weight.

For every unselected non-loop edge, the unique selected-forest path is inspected.
Its weight must be at least the largest path-edge weight. Equality gives an
optimal one-edge exchange; all equal maximum-weight path edges are marked
exchangeable. A selected edge with no such exchange belongs to every minimum
forest. A strictly heavier unselected edge belongs to none. These cycle checks
certify optimality and whether the source-edge set of the minimum forest is
unique; they do not enumerate all optimal forests. Equal input weights alone do
not imply nonuniqueness. Rounded totals have authoritative exact rational pairs.

Bounds: 128 vertices, 1024 edges, weights in [-1e100, 1e100], 300000 forest-arc
examinations, at most 128 emitted exchange witnesses and 12000-bit exact totals.
An exchange witness adds its unselected edge and removes its selected edge.
Reference: Sedgewick and Wayne, https://algs4.cs.princeton.edu/43mst/
"""

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
Classification = Literal[
    "selected_required",
    "selected_exchangeable",
    "unselected_exchangeable",
    "excluded_heavier_than_path",
    "excluded_self_loop",
]


class Edge(InputModel):
    edge_id: Name
    first_vertex: Name
    second_vertex: Name
    weight: float = Field(strict=True, ge=-1e100, le=1e100)


class Input(InputModel):
    vertices: list[Name] = Field(min_length=1, max_length=128)
    edges: list[Edge] = Field(max_length=1024)
    max_exchange_witnesses: int = Field(default=32, strict=True, ge=0, le=128)

    @model_validator(mode="after")
    def identities(self) -> Self:
        names = set(self.vertices)
        if len(names) != len(self.vertices):
            raise ValueError("vertices must have unique names")
        if len({edge.edge_id for edge in self.edges}) != len(self.edges):
            raise ValueError("source edge IDs must be unique")
        if any(
            edge.first_vertex not in names or edge.second_vertex not in names for edge in self.edges
        ):
            raise ValueError("edge endpoints must name declared vertices")
        return self


class ExactNumber(OutputModel):
    numerator: str
    denominator: str
    value: float


class EdgeClassification(OutputModel):
    edge_row_index: int
    edge_id: str
    first_vertex_index: int
    second_vertex_index: int
    weight: float
    selected: bool
    classification: Classification
    maximum_selected_path_weight: float | None


class Exchange(OutputModel):
    add_edge_row_index: int
    remove_edge_row_index: int


class Component(OutputModel):
    vertex_indices: list[int]
    selected_edge_row_indices: list[int]
    total_weight: ExactNumber
    unique_minimum_edge_set: bool


class Output(OutputModel):
    vertices: list[str]
    selected_edge_row_indices: list[int]
    total_weight: ExactNumber
    components: list[Component]
    edge_classifications: list[EdgeClassification]
    unique_minimum_edge_set: bool
    exchange_witnesses: list[Exchange]
    exchangeable_unselected_edge_count: int
    omitted_exchange_witness_count: int
    forest_arc_examinations: int
    exact_cycle_optimality_verified: Literal[True] = True
    external_graph_verified: Literal[False] = False


@dataclass
class _UnionFind:
    parent: list[int]
    size: list[int]

    def find(self, vertex: int) -> int:
        while self.parent[vertex] != vertex:
            self.parent[vertex] = self.parent[self.parent[vertex]]
            vertex = self.parent[vertex]
        return vertex

    def join(self, first: int, second: int) -> bool:
        first, second = self.find(first), self.find(second)
        if first == second:
            return False
        if (self.size[first], -first) < (self.size[second], -second):
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        return True


def _exact(value: Fraction) -> ExactNumber:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("forest total exceeds 12000-bit exact budget")
    preview = float(value)
    if not isfinite(preview) or (value and preview == 0):
        raise ValueError("nonzero forest total is outside finite binary64 range")
    return ExactNumber(
        numerator=str(value.numerator), denominator=str(value.denominator), value=preview
    )


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.vertices)
    positions = {name: index for index, name in enumerate(request.vertices)}
    endpoints = [
        (positions[edge.first_vertex], positions[edge.second_vertex]) for edge in request.edges
    ]
    union = _UnionFind(list(range(count)), [1] * count)
    selected: list[int] = []
    adjacency: list[list[tuple[int, int]]] = [[] for _ in request.vertices]
    for index in sorted(
        range(len(request.edges)), key=lambda row: (request.edges[row].weight, row)
    ):
        first, second = endpoints[index]
        if union.join(first, second):
            selected.append(index)
            adjacency[first].append((second, index))
            adjacency[second].append((first, index))
    selected_set = set(selected)
    exchangeable: set[int] = set()
    maxima: dict[int, float] = {}
    witnesses: list[Exchange] = []
    exchange_count = examinations = 0
    for index, edge in enumerate(request.edges):
        first, second = endpoints[index]
        if index in selected_set or first == second:
            continue
        predecessor: dict[int, tuple[int, int]] = {}
        visited = {first}
        pending = [first]
        while pending and second not in visited:
            current = pending.pop()
            for neighbor, path_edge in adjacency[current]:
                examinations += 1
                if examinations > 300_000:
                    raise ValueError("forest cycle checks exceed 300000 arc examinations")
                if neighbor not in visited:
                    visited.add(neighbor)
                    predecessor[neighbor] = (current, path_edge)
                    pending.append(neighbor)
        if second not in visited:
            raise ValueError("selected forest fails to span an original edge")
        path: list[int] = []
        current = second
        while current != first:
            current, path_edge = predecessor[current]
            path.append(path_edge)
        maximum = max(request.edges[row].weight for row in path)
        maxima[index] = maximum
        if edge.weight < maximum:
            raise ValueError("minimum-forest cycle certificate failed")
        if edge.weight == maximum:
            replacements = [row for row in path if request.edges[row].weight == maximum]
            exchangeable.add(index)
            exchangeable.update(replacements)
            exchange_count += 1
            if len(witnesses) < request.max_exchange_witnesses:
                witnesses.append(
                    Exchange(add_edge_row_index=index, remove_edge_row_index=min(replacements))
                )
    grouped: dict[int, list[int]] = {}
    for vertex in range(count):
        grouped.setdefault(union.find(vertex), []).append(vertex)
    components: list[Component] = []
    for vertices in sorted(grouped.values(), key=lambda members: members[0]):
        members = set(vertices)
        tree_edges = [index for index in selected if endpoints[index][0] in members]
        if len(tree_edges) != len(vertices) - 1:
            raise ValueError("component tree edge count certificate failed")
        components.append(
            Component(
                vertex_indices=vertices,
                selected_edge_row_indices=tree_edges,
                total_weight=_exact(
                    sum((Fraction(request.edges[row].weight) for row in tree_edges), Fraction())
                ),
                unique_minimum_edge_set=all(row not in exchangeable for row in tree_edges),
            )
        )
    classifications: list[EdgeClassification] = []
    for index, edge in enumerate(request.edges):
        first, second = endpoints[index]
        classification: Classification
        if first == second:
            classification = "excluded_self_loop"
        elif index in selected_set:
            classification = (
                "selected_exchangeable" if index in exchangeable else "selected_required"
            )
        elif index in exchangeable:
            classification = "unselected_exchangeable"
        else:
            classification = "excluded_heavier_than_path"
        classifications.append(
            EdgeClassification(
                edge_row_index=index,
                edge_id=edge.edge_id,
                first_vertex_index=first,
                second_vertex_index=second,
                weight=edge.weight,
                selected=index in selected_set,
                classification=classification,
                maximum_selected_path_weight=maxima.get(index),
            )
        )
    return Output(
        vertices=request.vertices,
        selected_edge_row_indices=selected,
        total_weight=_exact(
            sum((Fraction(request.edges[row].weight) for row in selected), Fraction())
        ),
        components=components,
        edge_classifications=classifications,
        unique_minimum_edge_set=exchange_count == 0,
        exchange_witnesses=witnesses,
        exchangeable_unselected_edge_count=exchange_count,
        omitted_exchange_witness_count=exchange_count - len(witnesses),
        forest_arc_examinations=examinations,
    )


OPERATION = Operation(
    id="features.minimum_spanning_forest",
    kind="feature",
    description="Construct a stable Kruskal forest with exact weight totals, source-edge classifications and verified optimal exchanges/uniqueness.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
