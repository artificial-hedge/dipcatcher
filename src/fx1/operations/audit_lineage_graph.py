"""Audit a supplied dependency graph, without claiming receipt verification.

Edges run from dependency to dependent. Duplicate declarations and unknown
references are structural failures. Cycles are identified by iterative strongly
connected components; the count is cyclic components, never an exponential
enumeration of all cycles. Topological order is returned only for a nonempty,
fully valid graph. Hashes and source truth are outside this operation's scope.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from heapq import heapify, heappop, heappush
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

NodeId = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]


class Edge(InputModel):
    dependency_id: NodeId
    dependent_id: NodeId


class Input(InputModel):
    nodes: list[NodeId] = Field(default_factory=list, max_length=10_000)
    edges: list[Edge] = Field(default_factory=list, max_length=50_000)
    max_diagnostics: int = Field(
        default=50,
        strict=True,
        ge=0,
        le=100,
        description="Maximum examples per category: duplicate nodes, duplicate edges, unknown edges, cycles.",
    )
    max_indices_per_example: int = Field(default=10, strict=True, ge=1, le=10)
    max_nodes_per_cycle_component: int = Field(default=10, strict=True, ge=1, le=10)


class DuplicateNode(OutputModel):
    node_id: str
    declaration_count: int
    node_indices: list[int]
    omitted_indices: int


class DuplicateEdge(OutputModel):
    dependency_id: str
    dependent_id: str
    declaration_count: int
    edge_indices: list[int]
    omitted_indices: int


class UnknownEdge(OutputModel):
    edge_index: int
    dependency_id: str
    dependent_id: str
    unknown_node_ids: list[str]


class CyclicComponent(OutputModel):
    component_index: int
    node_count: int
    internal_edge_count: int
    node_ids: list[str]
    omitted_node_ids: int


class Output(OutputModel):
    verification_scope: Literal["supplied_graph_structure_only"] = "supplied_graph_structure_only"
    receipt_verification: Literal[False] = False
    edge_orientation: Literal["dependency_to_dependent"] = "dependency_to_dependent"
    assessment: Literal["valid", "invalid", "no_nodes"]
    passed: bool | None
    node_declarations: int
    distinct_nodes: int
    edge_declarations: int
    distinct_resolved_edges: int
    duplicate_node_groups: int
    extra_node_declarations: int
    duplicate_edge_groups: int
    extra_edge_declarations: int
    unknown_reference_edges: int
    distinct_unknown_node_ids: int
    cyclic_components: int
    cyclic_nodes: int
    known_graph_is_acyclic: bool | None
    dependent_nodes_blocked_by_cycles: int
    topological_order: list[str] | None
    duplicate_nodes: list[DuplicateNode]
    duplicate_edges: list[DuplicateEdge]
    unknown_edges: list[UnknownEdge]
    cycles: list[CyclicComponent]
    omitted_duplicate_node_groups: int
    omitted_duplicate_edge_groups: int
    omitted_unknown_edges: int
    omitted_cyclic_components: int


def _strong_components(
    adjacency: dict[str, set[str]], reverse: dict[str, set[str]]
) -> list[list[str]]:
    """Kosaraju's two passes with explicit stacks, independent of recursion depth."""
    visited: set[str] = set()
    finished: list[str] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[str, Iterator[str]]] = [(start, iter(sorted(adjacency[start])))]
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
            except StopIteration:
                finished.append(node)
                stack.pop()
                continue
            if child not in visited:
                visited.add(child)
                stack.append((child, iter(sorted(adjacency[child]))))

    assigned: set[str] = set()
    components: list[list[str]] = []
    for start in reversed(finished):
        if start in assigned:
            continue
        assigned.add(start)
        pending = [start]
        component: list[str] = []
        while pending:
            node = pending.pop()
            component.append(node)
            for parent in reverse[node]:
                if parent not in assigned:
                    assigned.add(parent)
                    pending.append(parent)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def execute(request: Input, context: OperationContext) -> Output:
    node_indices: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(request.nodes):
        node_indices[node].append(index)
    known = set(node_indices)
    adjacency: dict[str, set[str]] = {node: set() for node in known}
    reverse: dict[str, set[str]] = {node: set() for node in known}
    edge_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    unknown_edges: list[UnknownEdge] = []
    unknown_count = 0
    unknown_nodes: set[str] = set()
    for index, edge in enumerate(request.edges):
        dependency, dependent = edge.dependency_id, edge.dependent_id
        edge_indices[(dependency, dependent)].append(index)
        absent = {dependency, dependent} - known
        if absent:
            unknown_count += 1
            unknown_nodes.update(absent)
            if len(unknown_edges) < request.max_diagnostics:
                unknown_edges.append(
                    UnknownEdge(
                        edge_index=index,
                        dependency_id=dependency,
                        dependent_id=dependent,
                        unknown_node_ids=sorted(absent),
                    )
                )
        else:
            adjacency[dependency].add(dependent)
            reverse[dependent].add(dependency)

    duplicate_nodes: list[DuplicateNode] = []
    duplicate_node_groups = extra_nodes = 0
    for node, indices in sorted(node_indices.items()):
        if len(indices) < 2:
            continue
        duplicate_node_groups += 1
        extra_nodes += len(indices) - 1
        if len(duplicate_nodes) < request.max_diagnostics:
            examples = indices[: request.max_indices_per_example]
            duplicate_nodes.append(
                DuplicateNode(
                    node_id=node,
                    declaration_count=len(indices),
                    node_indices=examples,
                    omitted_indices=len(indices) - len(examples),
                )
            )
    duplicate_edges: list[DuplicateEdge] = []
    duplicate_edge_groups = extra_edges = 0
    for (dependency, dependent), indices in sorted(edge_indices.items()):
        if len(indices) < 2:
            continue
        duplicate_edge_groups += 1
        extra_edges += len(indices) - 1
        if len(duplicate_edges) < request.max_diagnostics:
            examples = indices[: request.max_indices_per_example]
            duplicate_edges.append(
                DuplicateEdge(
                    dependency_id=dependency,
                    dependent_id=dependent,
                    declaration_count=len(indices),
                    edge_indices=examples,
                    omitted_indices=len(indices) - len(examples),
                )
            )

    components = _strong_components(adjacency, reverse)
    cyclic = [
        component
        for component in components
        if len(component) > 1 or component[0] in adjacency[component[0]]
    ]
    cycles: list[CyclicComponent] = []
    for index, component in enumerate(cyclic[: request.max_diagnostics]):
        members = set(component)
        node_examples = component[: request.max_nodes_per_cycle_component]
        cycles.append(
            CyclicComponent(
                component_index=index,
                node_count=len(component),
                internal_edge_count=sum(len(adjacency[node] & members) for node in component),
                node_ids=node_examples,
                omitted_node_ids=len(component) - len(node_examples),
            )
        )

    # Kahn elimination distinguishes true cycle members from acyclic nodes
    # whose dependencies can never complete because an upstream cycle remains.
    indegree = {node: len(reverse[node]) for node in known}
    ready = [node for node in known if indegree[node] == 0]
    heapify(ready)
    ordered: list[str] = []
    while ready:
        node = heappop(ready)
        ordered.append(node)
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heappush(ready, child)

    invalid = bool(extra_nodes or extra_edges or unknown_count or cyclic)
    cycle_nodes = sum(map(len, cyclic))
    passed = False if invalid else (True if known else None)
    return Output(
        assessment="invalid" if invalid else ("valid" if known else "no_nodes"),
        passed=passed,
        node_declarations=len(request.nodes),
        distinct_nodes=len(known),
        edge_declarations=len(request.edges),
        distinct_resolved_edges=sum(map(len, adjacency.values())),
        duplicate_node_groups=duplicate_node_groups,
        extra_node_declarations=extra_nodes,
        duplicate_edge_groups=duplicate_edge_groups,
        extra_edge_declarations=extra_edges,
        unknown_reference_edges=unknown_count,
        distinct_unknown_node_ids=len(unknown_nodes),
        cyclic_components=len(cyclic),
        cyclic_nodes=cycle_nodes,
        known_graph_is_acyclic=not cyclic if known else None,
        dependent_nodes_blocked_by_cycles=len(known) - len(ordered) - cycle_nodes,
        topological_order=ordered if passed else None,
        duplicate_nodes=duplicate_nodes,
        duplicate_edges=duplicate_edges,
        unknown_edges=unknown_edges,
        cycles=cycles,
        omitted_duplicate_node_groups=duplicate_node_groups - len(duplicate_nodes),
        omitted_duplicate_edge_groups=duplicate_edge_groups - len(duplicate_edges),
        omitted_unknown_edges=unknown_count - len(unknown_edges),
        omitted_cyclic_components=len(cyclic) - len(cycles),
    )


OPERATION = Operation(
    id="skills.audit_lineage_graph",
    kind="skill",
    description=(
        "Audit supplied dependency nodes/edges for duplicate declarations, unknown references, "
        "and strongly connected cycles using iterative graph algorithms. Return deterministic "
        "topological order only for valid graphs; this does not verify receipts or provenance."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
