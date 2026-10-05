"""Reconcile caller-declared affine unit maps using exact rational arithmetic.

An edge declares y = scale*x + offset, from its source unit to its target unit.
Names have no built-in physical meaning; dimensions are caller-supplied exponent
maps with zero powers removed. Every usable scale must be nonzero so its inverse
exists. No real-world constants, logarithmic maps or value-dependent conversions
are inferred. Duplicate unit identifiers are ambiguous, including equal rows.

A breadth-first spanning forest assigns exact reference-to-unit maps, choosing
lexical reference names and input-edge order. Each edge is compared with those
maps. A disagreement produces a simple tree-path-plus-edge closed witness whose
composition is not the identity. Counts describe disagreeing declared edges
against this forest, not enumeration of every possible contradictory cycle.
Composed query maps are returned only for connected, consistent components.

Bounds: 128 units, 16 dimensional axes/unit, 2048 edges, 1000 queries, 100 findings,
20 complete cycle witnesses and 512 witness edges in total. Rational inputs have
numerators/denominators of magnitude at most 1e12. Every intermediate normalized
numerator/denominator is bounded by max_rational_bits (1024 default, 2048 maximum);
exceeding it fails the request without partial consistency claims. At most 128
tree steps are considered per retained cycle. Outputs page up to 100 queries.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64, pattern=r"\S")]
Power = Annotated[int, Field(strict=True, ge=-16, le=16)]


class Rational(InputModel):
    numerator: int = Field(strict=True, ge=-(10**12), le=10**12)
    denominator: int = Field(default=1, strict=True, ge=1, le=10**12)


class Unit(InputModel):
    unit_id: Name
    dimensions: dict[Name, Power] = Field(max_length=16)


class Conversion(InputModel):
    from_unit: Name
    to_unit: Name
    scale: Rational
    offset: Rational


class Query(InputModel):
    from_unit: Name
    to_unit: Name


class Input(InputModel):
    units: list[Unit] = Field(min_length=1, max_length=128)
    conversions: list[Conversion] = Field(max_length=2048)
    queries: list[Query] = Field(default_factory=list, max_length=1000)
    duplicate_conversion_policy: Literal["allow_equivalent", "reject"] = "allow_equivalent"
    max_rational_bits: int = Field(default=1024, strict=True, ge=128, le=2048)
    max_diagnostics: int = Field(default=50, strict=True, ge=1, le=100)
    max_cycle_witnesses: int = Field(default=5, strict=True, ge=1, le=20)
    max_witness_edges: int = Field(default=256, strict=True, ge=1, le=512)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=100)


class ExactValue(OutputModel):
    numerator: str
    denominator: str


class Finding(OutputModel):
    code: str
    unit_row_index: int | None = None
    conversion_row_index: int | None = None
    other_conversion_row_index: int | None = None
    unit_id: str | None = None


class WitnessStep(OutputModel):
    conversion_row_index: int
    direction: Literal["declared", "inverse"]
    from_unit: str
    to_unit: str
    scale: ExactValue
    offset: ExactValue


class Cycle(OutputModel):
    disagreeing_conversion_row_index: int
    start_unit: str
    steps: list[WitnessStep]
    composed_scale: ExactValue
    composed_offset: ExactValue


class UnitMap(OutputModel):
    unit_row_index: int
    unit_id: str
    reference_unit: str | None
    component_index: int | None
    status: Literal["ambiguous_unit_id", "consistent", "inconsistent_component"]
    reference_to_unit_scale: ExactValue | None
    reference_to_unit_offset: ExactValue | None


class QueryResult(OutputModel):
    query_row_index: int
    from_unit: str
    to_unit: str
    status: Literal[
        "resolved", "unknown_or_ambiguous_unit", "disconnected", "inconsistent_component"
    ]
    scale: ExactValue | None
    offset: ExactValue | None


class Output(OutputModel):
    passed: bool = Field(
        description="Whether supplied unit/conversion declarations form a valid consistent graph; query outcomes do not affect this flag."
    )
    declared_unit_count: int
    declared_conversion_count: int
    duplicate_conversion_policy: Literal["allow_equivalent", "reject"]
    arithmetic_bit_limit: int
    usable_conversion_count: int
    duplicate_conversion_count: int
    component_count: int
    inconsistent_component_count: int
    disagreeing_conversion_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    cycles: list[Cycle]
    omitted_cycle_witnesses: int
    witness_edge_count: int
    unit_maps: list[UnitMap]
    queries: list[QueryResult]
    query_count: int
    offset: int
    has_more: bool
    physical_constants_verified: Literal[False] = False
    composition_scope: Literal["declared_exact_affine_maps_only"] = (
        "declared_exact_affine_maps_only"
    )


@dataclass(frozen=True)
class _Step:
    source: int
    target: int
    row: int
    forward: bool
    scale: Fraction
    offset: Fraction


def _fraction(value: Rational) -> Fraction:
    return Fraction(value.numerator, value.denominator)


def _bounded(value: Fraction, bits: int) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > bits:
        raise ValueError("exact affine composition exceeds max_rational_bits")
    return value


def _exact(value: Fraction) -> ExactValue:
    return ExactValue(numerator=str(value.numerator), denominator=str(value.denominator))


def _reverse(step: _Step, bits: int) -> _Step:
    return _Step(
        step.target,
        step.source,
        step.row,
        not step.forward,
        _bounded(1 / step.scale, bits),
        _bounded(-step.offset / step.scale, bits),
    )


def _closed_path(
    edge: _Step, parents: list[_Step | None], depths: list[int], bits: int
) -> list[_Step]:
    """Walk target up to the LCA, down to source, then close using this edge."""
    left, right = edge.target, edge.source
    upward: list[_Step] = []
    downward: list[_Step] = []
    while left != right:
        if depths[left] >= depths[right]:
            parent = parents[left]
            if parent is None:
                raise ValueError("internal conversion witness has no parent")
            upward.append(_reverse(parent, bits))
            left = parent.source
        else:
            parent = parents[right]
            if parent is None:
                raise ValueError("internal conversion witness has no parent")
            downward.append(parent)
            right = parent.source
    return upward + list(reversed(downward)) + [edge]


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    diagnostics: list[Finding] = []

    def report(
        code: str,
        unit_row_index: int | None = None,
        conversion_row_index: int | None = None,
        other_conversion_row_index: int | None = None,
        unit_id: str | None = None,
    ) -> None:
        issues[code] += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Finding(
                    code=code,
                    unit_row_index=unit_row_index,
                    conversion_row_index=conversion_row_index,
                    other_conversion_row_index=other_conversion_row_index,
                    unit_id=unit_id,
                )
            )

    declarations: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(request.units):
        declarations[unit.unit_id].append(index)
    unique = {name: rows[0] for name, rows in declarations.items() if len(rows) == 1}
    for index, unit in enumerate(request.units):
        if len(declarations[unit.unit_id]) != 1:
            report("ambiguous_unit_id", unit_row_index=index, unit_id=unit.unit_id)
    dimensions = [
        tuple(sorted((name, power) for name, power in row.dimensions.items() if power))
        for row in request.units
    ]
    adjacency: list[list[_Step]] = [[] for _ in request.units]
    edges: list[_Step] = []
    equivalent: dict[tuple[int, int, Fraction, Fraction], int] = {}
    duplicate_count = 0
    for index, row in enumerate(request.conversions):
        source, target = unique.get(row.from_unit), unique.get(row.to_unit)
        if source is None or target is None:
            report("unknown_or_ambiguous_conversion_unit", conversion_row_index=index)
            continue
        if dimensions[source] != dimensions[target]:
            report("dimension_mismatch", conversion_row_index=index)
            continue
        scale, offset = _fraction(row.scale), _fraction(row.offset)
        if scale == 0:
            report("noninvertible_zero_scale", conversion_row_index=index)
            continue
        key = (source, target, scale, offset)
        if key in equivalent:
            duplicate_count += 1
            if request.duplicate_conversion_policy == "reject":
                report(
                    "duplicate_conversion",
                    conversion_row_index=index,
                    other_conversion_row_index=equivalent[key],
                )
        else:
            equivalent[key] = index
        step = _Step(source, target, index, True, scale, offset)
        edges.append(step)
        adjacency[source].append(step)
        adjacency[target].append(_reverse(step, request.max_rational_bits))

    components = [-1] * len(request.units)
    scales = [Fraction(1) for _ in request.units]
    offsets = [Fraction(0) for _ in request.units]
    parents: list[_Step | None] = [None for _ in request.units]
    depths = [0] * len(request.units)
    roots: list[int] = []
    for name in sorted(unique):
        root = unique[name]
        if components[root] != -1:
            continue
        component = len(roots)
        roots.append(root)
        components[root] = component
        pending = deque([root])
        while pending:
            source = pending.popleft()
            for edge in adjacency[source]:
                target = edge.target
                if components[target] != -1:
                    continue
                scales[target] = _bounded(edge.scale * scales[source], request.max_rational_bits)
                offsets[target] = _bounded(
                    _bounded(edge.scale * offsets[source], request.max_rational_bits) + edge.offset,
                    request.max_rational_bits,
                )
                parents[target] = edge
                depths[target] = depths[source] + 1
                components[target] = component
                pending.append(target)

    inconsistent: set[int] = set()
    cycles: list[Cycle] = []
    disagreement_count = witness_edges = 0
    for edge in edges:
        expected_scale = _bounded(edge.scale * scales[edge.source], request.max_rational_bits)
        expected_offset = _bounded(
            _bounded(edge.scale * offsets[edge.source], request.max_rational_bits) + edge.offset,
            request.max_rational_bits,
        )
        if (expected_scale, expected_offset) == (scales[edge.target], offsets[edge.target]):
            continue
        disagreement_count += 1
        inconsistent.add(components[edge.source])
        report("contradictory_conversion", conversion_row_index=edge.row)
        if len(cycles) >= request.max_cycle_witnesses or witness_edges >= request.max_witness_edges:
            continue
        path = _closed_path(edge, parents, depths, request.max_rational_bits)
        if witness_edges + len(path) > request.max_witness_edges:
            continue
        composed_scale, composed_offset = Fraction(1), Fraction(0)
        steps: list[WitnessStep] = []
        for step in path:
            composed_scale = _bounded(step.scale * composed_scale, request.max_rational_bits)
            composed_offset = _bounded(
                _bounded(step.scale * composed_offset, request.max_rational_bits) + step.offset,
                request.max_rational_bits,
            )
            steps.append(
                WitnessStep(
                    conversion_row_index=step.row,
                    direction="declared" if step.forward else "inverse",
                    from_unit=request.units[step.source].unit_id,
                    to_unit=request.units[step.target].unit_id,
                    scale=_exact(step.scale),
                    offset=_exact(step.offset),
                )
            )
        if (composed_scale, composed_offset) == (Fraction(1), Fraction(0)):
            raise ValueError("internal conversion witness unexpectedly composes to identity")
        cycles.append(
            Cycle(
                disagreeing_conversion_row_index=edge.row,
                start_unit=request.units[path[0].source].unit_id,
                steps=steps,
                composed_scale=_exact(composed_scale),
                composed_offset=_exact(composed_offset),
            )
        )
        witness_edges += len(path)

    unit_maps: list[UnitMap] = []
    for index, unit in enumerate(request.units):
        component = components[index]
        resolved = component >= 0 and component not in inconsistent
        unit_maps.append(
            UnitMap(
                unit_row_index=index,
                unit_id=unit.unit_id,
                reference_unit=request.units[roots[component]].unit_id if component >= 0 else None,
                component_index=component if component >= 0 else None,
                status="ambiguous_unit_id"
                if component < 0
                else "consistent"
                if resolved
                else "inconsistent_component",
                reference_to_unit_scale=_exact(scales[index]) if resolved else None,
                reference_to_unit_offset=_exact(offsets[index]) if resolved else None,
            )
        )
    queries: list[QueryResult] = []
    for index in range(request.offset, min(len(request.queries), request.offset + request.limit)):
        query = request.queries[index]
        source, target = unique.get(query.from_unit), unique.get(query.to_unit)
        status: Literal[
            "resolved", "unknown_or_ambiguous_unit", "disconnected", "inconsistent_component"
        ]
        scale_result = offset_result = None
        if source is None or target is None:
            status = "unknown_or_ambiguous_unit"
        elif components[source] != components[target]:
            status = "disconnected"
        elif components[source] in inconsistent:
            status = "inconsistent_component"
        else:
            status = "resolved"
            scale = _bounded(scales[target] / scales[source], request.max_rational_bits)
            offset = _bounded(
                offsets[target] - _bounded(scale * offsets[source], request.max_rational_bits),
                request.max_rational_bits,
            )
            scale_result, offset_result = _exact(scale), _exact(offset)
        queries.append(
            QueryResult(
                query_row_index=index,
                from_unit=query.from_unit,
                to_unit=query.to_unit,
                status=status,
                scale=scale_result,
                offset=offset_result,
            )
        )
    return Output(
        passed=not issues,
        declared_unit_count=len(request.units),
        declared_conversion_count=len(request.conversions),
        duplicate_conversion_policy=request.duplicate_conversion_policy,
        arithmetic_bit_limit=request.max_rational_bits,
        usable_conversion_count=len(edges),
        duplicate_conversion_count=duplicate_count,
        component_count=len(roots),
        inconsistent_component_count=len(inconsistent),
        disagreeing_conversion_count=disagreement_count,
        issue_counts=dict(sorted(issues.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=sum(issues.values()) - len(diagnostics),
        cycles=cycles,
        omitted_cycle_witnesses=disagreement_count - len(cycles),
        witness_edge_count=witness_edges,
        unit_maps=unit_maps,
        queries=queries,
        query_count=len(request.queries),
        offset=request.offset,
        has_more=request.offset + len(queries) < len(request.queries),
    )


OPERATION = Operation(
    id="skills.audit_unit_conversions",
    kind="skill",
    description="Audit declared dimensional affine unit conversions with exact rational graph composition and contradictory-cycle source witnesses.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
