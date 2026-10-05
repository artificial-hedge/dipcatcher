"""Exact grounded-Laplacian electrical resistance and optional tree mass.

Edges are undirected positive conductances, not resistances. Parallel source
rows add conductance and remain distinct edges in spanning-tree products;
self-loops are rejected. Each connected component grounds its smallest input
vertex index. Own rational LDL factorization/triangular solves invert the
grounded Laplacian. For a requested unit-current source/sink pair, voltage
difference equals effective resistance. Disconnected pairs have infinite
resistance; equal vertices have zero resistance, including isolated vertices.

Optional potentials align with all input vertices, with nulls outside the
pair's component. They use the component's fixed ground, not a grounded sink.
The optional tree mass is the sum of products of conductances over spanning
trees, from the grounded determinant. A singleton's empty tree has mass one;
the whole graph has mass zero when disconnected. This is not a spanning-forest
mass or a count unless all conductances equal one.

Inputs: <=20 vertices, <=128 edges, <=200 queried pairs, conductances [1e-6,1e6].
Rational arithmetic and serialized exact values are capped at 12000 bits.
Float previews reject nonzero underflow/overflow. This operation interprets
the supplied graph only; it makes no causal, market or observed-network claim.
Electrical and weighted matrix-tree references:
https://www.cs.yale.edu/homes/spielman/561/lect14-18.pdf
https://www.cs.yale.edu/homes/vishnoi/Lxb-Web.pdf
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class Edge(InputModel):
    source: Name
    target: Name
    conductance: float = Field(strict=True, ge=1e-6, le=1e6)


class Pair(InputModel):
    source: Name
    target: Name


class Input(InputModel):
    vertices: list[Name] = Field(min_length=1, max_length=20)
    edges: list[Edge] = Field(max_length=128)
    pairs: list[Pair] = Field(default_factory=list, max_length=200)
    include_potentials: bool = Field(default=False, strict=True)
    include_spanning_tree_mass: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def declared_vertices(self) -> Self:
        names = set(self.vertices)
        if len(names) != len(self.vertices):
            raise ValueError("vertex names must be unique")
        endpoints = [(row.source, row.target) for row in self.edges] + [
            (row.source, row.target) for row in self.pairs
        ]
        if any(source not in names or target not in names for source, target in endpoints):
            raise ValueError("all edge and query endpoints must be declared vertices")
        if any(row.source == row.target for row in self.edges):
            raise ValueError("self-loops are not supported")
        return self


class ExactValue(OutputModel):
    value: float
    numerator: str
    denominator: str


class Component(OutputModel):
    component_index: int
    vertex_indices: list[int]
    vertex_names: list[str]
    grounded_vertex_index: int
    edge_row_indices: list[int]
    spanning_tree_mass: ExactValue | None


class PairResult(OutputModel):
    query_index: int
    source_index: int
    target_index: int
    component_index: int | None
    status: Literal["finite", "infinite_disconnected"]
    effective_resistance: ExactValue | None
    potentials: list[float | None] | None


class Output(OutputModel):
    vertex_count: int
    edge_count: int
    component_count: int
    components: list[Component]
    pairs: list[PairResult]
    whole_graph_spanning_tree_mass: ExactValue | None
    parallel_edges_summed: Literal[True] = True
    edge_weight_interpretation: Literal["conductance"] = "conductance"
    resistance_units: Literal["inverse_conductance_units"] = "inverse_conductance_units"
    grounded_system_and_pair_energy_checked_exactly: Literal[True] = True
    graph_observation_truth_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("graph resistance arithmetic exceeds 12000-bit budget")
    return value


def _number(value: Fraction) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("graph summary overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError("nonzero graph summary is outside finite binary64 range")
    return result


def _exact(value: Fraction) -> ExactValue:
    preview = _number(value)
    return ExactValue(
        value=preview, numerator=str(value.numerator), denominator=str(value.denominator)
    )


def _grounded_inverse(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], Fraction]:
    size = len(matrix)
    lower = [[Fraction(i == j) for j in range(size)] for i in range(size)]
    diagonal: list[Fraction] = []
    determinant = Fraction(1)
    for column in range(size):
        pivot = _bounded(
            matrix[column][column]
            - sum((lower[column][k] ** 2 * diagonal[k] for k in range(column)), Fraction())
        )
        if pivot <= 0:
            raise ValueError("connected grounded Laplacian is not positive definite")
        diagonal.append(pivot)
        determinant = _bounded(determinant * pivot)
        for row in range(column + 1, size):
            lower[row][column] = _bounded(
                (
                    matrix[row][column]
                    - sum(
                        (lower[row][k] * lower[column][k] * diagonal[k] for k in range(column)),
                        Fraction(),
                    )
                )
                / pivot
            )
    inverse = [[Fraction() for _ in range(size)] for _ in range(size)]
    for basis in range(size):
        forward: list[Fraction] = []
        for row in range(size):
            forward.append(
                _bounded(
                    Fraction(row == basis)
                    - sum((lower[row][j] * forward[j] for j in range(row)), Fraction())
                )
            )
        solution = [Fraction()] * size
        for row in range(size - 1, -1, -1):
            solution[row] = _bounded(
                forward[row] / diagonal[row]
                - sum((lower[j][row] * solution[j] for j in range(row + 1, size)), Fraction())
            )
        for row, value in enumerate(solution):
            inverse[row][basis] = value
        for row in range(size):
            if sum((matrix[row][j] * solution[j] for j in range(size)), Fraction()) != (
                row == basis
            ):
                raise ValueError("grounded Laplacian solve failed its exact residual check")
    return inverse, determinant


def execute(request: Input, context: OperationContext) -> Output:
    size = len(request.vertices)
    positions = {name: index for index, name in enumerate(request.vertices)}
    adjacency: list[set[int]] = [set() for _ in range(size)]
    laplacian = [[Fraction() for _ in range(size)] for _ in range(size)]
    edges: list[tuple[int, int, Fraction]] = []
    for row in request.edges:
        source, target = positions[row.source], positions[row.target]
        weight = Fraction(row.conductance)
        edges.append((source, target, weight))
        adjacency[source].add(target)
        adjacency[target].add(source)
        laplacian[source][source] += weight
        laplacian[target][target] += weight
        laplacian[source][target] -= weight
        laplacian[target][source] -= weight
    unseen = set(range(size))
    groups: list[list[int]] = []
    while unseen:
        pending = [min(unseen)]
        found: set[int] = set()
        while pending:
            vertex = pending.pop()
            if vertex in found:
                continue
            found.add(vertex)
            pending.extend(adjacency[vertex] - found)
        unseen.difference_update(found)
        groups.append(sorted(found))
    membership = {vertex: index for index, group in enumerate(groups) for vertex in group}
    inverse_by_component: list[dict[tuple[int, int], Fraction]] = []
    tree_masses: list[Fraction] = []
    component_results: list[Component] = []
    for index, group in enumerate(groups):
        retained = group[1:]
        inverse, determinant = _grounded_inverse(
            [[laplacian[i][j] for j in retained] for i in retained]
        )
        inverse_by_component.append(
            {
                (i, j): inverse[row][column]
                for row, i in enumerate(retained)
                for column, j in enumerate(retained)
            }
        )
        tree_masses.append(determinant)
        component_results.append(
            Component(
                component_index=index,
                vertex_indices=group,
                vertex_names=[request.vertices[vertex] for vertex in group],
                grounded_vertex_index=group[0],
                edge_row_indices=[
                    row for row, (source, _, _) in enumerate(edges) if membership[source] == index
                ],
                spanning_tree_mass=_exact(determinant)
                if request.include_spanning_tree_mass
                else None,
            )
        )
    results: list[PairResult] = []
    for query_index, pair in enumerate(request.pairs):
        source, target = positions[pair.source], positions[pair.target]
        component = membership[source]
        if component != membership[target]:
            results.append(
                PairResult(
                    query_index=query_index,
                    source_index=source,
                    target_index=target,
                    component_index=None,
                    status="infinite_disconnected",
                    effective_resistance=None,
                    potentials=None,
                )
            )
            continue
        grounded_inverse = inverse_by_component[component]
        voltages = {
            vertex: grounded_inverse.get((vertex, source), Fraction())
            - grounded_inverse.get((vertex, target), Fraction())
            for vertex in groups[component]
        }
        resistance = _bounded(voltages[source] - voltages[target])
        energy = _bounded(
            sum(
                (
                    weight * (voltages[left] - voltages[right]) ** 2
                    for left, right, weight in edges
                    if membership[left] == component
                ),
                Fraction(),
            )
        )
        if resistance < 0 or energy != resistance or (source != target and resistance == 0):
            raise ValueError("effective resistance failed exact positivity/energy checks")
        results.append(
            PairResult(
                query_index=query_index,
                source_index=source,
                target_index=target,
                component_index=component,
                status="finite",
                effective_resistance=_exact(resistance),
                potentials=[
                    _number(voltages[vertex]) if vertex in voltages else None
                    for vertex in range(size)
                ]
                if request.include_potentials
                else None,
            )
        )
    return Output(
        vertex_count=size,
        edge_count=len(edges),
        component_count=len(groups),
        components=component_results,
        pairs=results,
        whole_graph_spanning_tree_mass=_exact(tree_masses[0] if len(groups) == 1 else Fraction())
        if request.include_spanning_tree_mass
        else None,
    )


OPERATION = Operation(
    id="features.graph_effective_resistance",
    kind="feature",
    description="Solve exact grounded conductance Laplacians for pairwise effective resistance and optional weighted spanning-tree mass.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
