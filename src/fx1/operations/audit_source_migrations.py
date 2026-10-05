"""Audit declared partition migrations and route through explicit event cutovers.

A visible plan owns one dataset/key namespace and half-open integer key range.
Events before cutover_time route to old_source; events at/after it route to
new_source. decision_time controls metadata visibility, not the event coordinate;
future event routing is permitted and does not claim an observation exists.
Partitions explicitly declare their old/new side, source owner and identity.
There is no fallback between sides or implicit selection among overlapping rows.

Both phase coverages are measured from visible, structurally valid partitions.
Old coverage is required for historical routing at every visible decision. New
coverage is required once the cutover clock is reached; before then its missing
spans are pending readiness information. Same-side overlap is invalid; dual-source
key coverage is expected migration duplication and is measured separately.

Queries retain invalid visible candidate partitions, including duplicate IDs, so
they cannot disappear and cause a healthy overlapping row to be silently chosen.
A locally resolved route does not certify the rest of its plan's coverage. All
IDs, ownership and ranges are declarations, not verified source bytes or records.
Global findings and passed cover all declarations, including future metadata.
Coverage and routing build separate visible-only identity/validity indexes:
future duplicate IDs cannot veto a currently unique visible source or plan.
Bounds: 128 plans, 10000 partitions, 1000 queries, 500000 candidate comparisons,
200 findings and 200 query results/page. Key universes are never materialized.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import islice
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Endpoint = Annotated[int, Field(strict=True, ge=-(2**63), le=2**63 - 1)]
Side = Literal["old", "new"]


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


class Plan(InputModel):
    migration_id: Name
    dataset_id: Name
    key_namespace: Name
    old_source: Name
    new_source: Name
    start: Endpoint
    end: Endpoint
    cutover_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("cutover_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Partition(InputModel):
    partition_id: Name
    migration_id: Name
    dataset_id: Name
    key_namespace: Name
    source: Name
    side: Side
    start: Endpoint
    end: Endpoint
    available_time: AwareDatetime

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Query(InputModel):
    migration_id: Name
    key: Endpoint
    event_time: AwareDatetime

    @field_validator("event_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    plans: list[Plan] = Field(min_length=1, max_length=128)
    partitions: list[Partition] = Field(max_length=10_000)
    queries: list[Query] = Field(default_factory=list, max_length=1000)
    decision_time: AwareDatetime
    max_route_comparisons: int = Field(default=100_000, strict=True, ge=0, le=500_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def route_work(self) -> Self:
        counts = Counter(part.migration_id for part in self.partitions)
        if sum(counts[query.migration_id] for query in self.queries) > self.max_route_comparisons:
            raise ValueError("query/partition candidate work exceeds max_route_comparisons")
        return self


class Finding(OutputModel):
    code: str
    plan_row_index: int | None = None
    side: Side | None = None
    partition_row_indexes: list[int] = Field(default_factory=list)
    start: int | None = None
    end: int | None = None


class Coverage(OutputModel):
    required_at_decision: bool
    usable_partition_count: int
    covered_units: int
    missing_units: int
    overlapping_units: int
    gap_count: int


class PlanResult(OutputModel):
    plan_row_index: int
    migration_id: str
    visible_at_decision: bool
    visible_identity_unique: bool
    old_coverage: Coverage | None
    new_coverage: Coverage | None
    dual_source_covered_units: int | None
    passed: bool


class QueryResult(OutputModel):
    query_row_index: int
    status: Literal[
        "resolved",
        "unresolved_plan",
        "plan_not_available",
        "invalid_plan",
        "outside_key_scope",
        "missing_partition",
        "ambiguous_partitions",
        "invalid_partition",
    ]
    plan_row_index: int | None
    prescribed_side: Side | None
    prescribed_source: str | None
    candidate_count: int
    candidate_partition_row_indexes: list[int]
    omitted_candidate_indexes: int
    selected_partition_row_index: int | None
    maximum_source_available_time: datetime | None


class Output(OutputModel):
    passed: bool
    plan_count: int
    partition_count: int
    query_count: int
    route_comparisons: int
    plans: list[PlanResult]
    queries: list[QueryResult]
    query_status_counts: dict[str, int]
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    offset: int
    has_more: bool
    fallback_performed: Literal[False] = False
    source_ownership_or_data_verified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid_plans: set[int] = set()

    def report(
        code: str,
        plan: int | None = None,
        side: Side | None = None,
        rows: list[int] | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        issues[code] += 1
        if plan is not None:
            invalid_plans.add(plan)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    plan_row_index=plan,
                    side=side,
                    partition_row_indexes=rows or [],
                    start=start,
                    end=end,
                )
            )

    plan_rows: dict[str, list[int]] = defaultdict(list)
    part_rows: dict[str, list[int]] = defaultdict(list)
    for index, plan in enumerate(request.plans):
        plan_rows[plan.migration_id].append(index)
    for index, part in enumerate(request.partitions):
        part_rows[part.migration_id].append(index)
    unique = {name: rows[0] for name, rows in plan_rows.items() if len(rows) == 1}
    visible_plan_rows: dict[str, list[int]] = defaultdict(list)
    for index, plan in enumerate(request.plans):
        if plan.available_time <= request.decision_time:
            visible_plan_rows[plan.migration_id].append(index)
    visible_unique = {name: rows[0] for name, rows in visible_plan_rows.items() if len(rows) == 1}
    structural = [True] * len(request.plans)
    for index, plan in enumerate(request.plans):
        if len(plan_rows[plan.migration_id]) != 1:
            report("ambiguous_migration_id", plan=index)
        if plan.start >= plan.end or plan.old_source == plan.new_source:
            report("invalid_migration_geometry_or_source_pair", plan=index)
            structural[index] = False
    partition_ids = Counter(part.partition_id for part in request.partitions)
    for index, part in enumerate(request.partitions):
        owner = unique.get(part.migration_id)
        if partition_ids[part.partition_id] != 1:
            report("ambiguous_partition_id", plan=owner, side=part.side, rows=[index])
        if part.start >= part.end:
            report("empty_or_inverted_partition", plan=owner, side=part.side, rows=[index])
        if owner is None:
            report("unresolved_partition_migration", side=part.side, rows=[index])
            continue
        plan = request.plans[owner]
        expected_source = plan.old_source if part.side == "old" else plan.new_source
        if (part.dataset_id, part.key_namespace, part.source) != (
            plan.dataset_id,
            plan.key_namespace,
            expected_source,
        ):
            report(
                "partition_identity_or_ownership_mismatch", plan=owner, side=part.side, rows=[index]
            )
        if part.start < plan.start or part.end > plan.end:
            report("partition_outside_migration_scope", plan=owner, side=part.side, rows=[index])

    visible_partition_ids = Counter(
        part.partition_id
        for part in request.partitions
        if part.available_time <= request.decision_time
    )
    route_invalid_parts: set[int] = set()
    for index, part in enumerate(request.partitions):
        if part.available_time > request.decision_time:
            continue
        visible_owner = visible_unique.get(part.migration_id)
        if (
            visible_partition_ids[part.partition_id] != 1
            or part.start >= part.end
            or visible_owner is None
        ):
            route_invalid_parts.add(index)
            continue
        visible_plan = request.plans[visible_owner]
        owner_source = visible_plan.old_source if part.side == "old" else visible_plan.new_source
        if (
            (part.dataset_id, part.key_namespace, part.source)
            != (visible_plan.dataset_id, visible_plan.key_namespace, owner_source)
            or part.start < visible_plan.start
            or part.end > visible_plan.end
        ):
            route_invalid_parts.add(index)
            if unique.get(part.migration_id) != visible_owner:
                report(
                    "visible_partition_identity_or_extent_mismatch",
                    plan=visible_owner,
                    side=part.side,
                    rows=[index],
                )

    def coverage(
        plan_index: int, side: Side, required: bool
    ) -> tuple[Coverage, list[tuple[int, int]]]:
        plan = request.plans[plan_index]
        usable = [
            row
            for row in part_rows[plan.migration_id]
            if row not in route_invalid_parts
            and request.partitions[row].side == side
            and request.partitions[row].available_time <= request.decision_time
        ]
        boundaries: dict[int, list[tuple[int, bool]]] = defaultdict(list)
        boundaries[plan.start]
        boundaries[plan.end]
        for row in usable:
            part = request.partitions[row]
            boundaries[part.start].append((row, True))
            boundaries[part.end].append((row, False))
        active: dict[int, None] = {}
        previous = plan.start
        covered = overlap = gaps = 0
        union: list[tuple[int, int]] = []
        for point in sorted(boundaries):
            if point > previous:
                if active:
                    covered += point - previous
                    if union and union[-1][1] == previous:
                        union[-1] = (union[-1][0], point)
                    else:
                        union.append((previous, point))
                    if len(active) > 1:
                        overlap += point - previous
                        report(
                            "same_source_overlap_span",
                            plan=plan_index,
                            side=side,
                            rows=list(islice(active, 2)),
                            start=previous,
                            end=point,
                        )
                else:
                    gaps += 1
                    if required:
                        report(
                            "required_source_gap",
                            plan=plan_index,
                            side=side,
                            start=previous,
                            end=point,
                        )
            for row, adding in boundaries[point]:
                if adding:
                    active[row] = None
                else:
                    active.pop(row, None)
            previous = point
        return Coverage(
            required_at_decision=required,
            usable_partition_count=len(usable),
            covered_units=covered,
            missing_units=plan.end - plan.start - covered,
            overlapping_units=overlap,
            gap_count=gaps,
        ), union

    plan_results: list[PlanResult] = []
    for index, plan in enumerate(request.plans):
        visible = plan.available_time <= request.decision_time
        old = new = None
        dual: int | None = None
        visible_identity_unique = visible_unique.get(plan.migration_id) == index
        if visible_identity_unique and structural[index]:
            old, old_union = coverage(index, "old", True)
            new, new_union = coverage(index, "new", request.decision_time >= plan.cutover_time)
            left = right = 0
            dual = 0
            while left < len(old_union) and right < len(new_union):
                first, second = old_union[left], new_union[right]
                dual += max(0, min(first[1], second[1]) - max(first[0], second[0]))
                if first[1] <= second[1]:
                    left += 1
                else:
                    right += 1
        plan_results.append(
            PlanResult(
                plan_row_index=index,
                migration_id=plan.migration_id,
                visible_at_decision=visible,
                visible_identity_unique=visible_identity_unique,
                old_coverage=old,
                new_coverage=new,
                dual_source_covered_units=dual,
                passed=index not in invalid_plans,
            )
        )

    results: list[QueryResult] = []
    query_counts: Counter[str] = Counter()
    comparisons = 0
    for index, query in enumerate(request.queries):
        plan_index = visible_unique.get(query.migration_id)
        side: Side | None = None
        source = None
        selected = None
        maximum = None
        candidates: list[int] = []
        status: Literal[
            "resolved",
            "unresolved_plan",
            "plan_not_available",
            "invalid_plan",
            "outside_key_scope",
            "missing_partition",
            "ambiguous_partitions",
            "invalid_partition",
        ]
        if plan_index is None:
            status = (
                "plan_not_available"
                if query.migration_id in plan_rows and not visible_plan_rows[query.migration_id]
                else "unresolved_plan"
            )
        elif not structural[plan_index]:
            status = "invalid_plan"
        else:
            plan = request.plans[plan_index]
            if plan.available_time > request.decision_time:
                status = "plan_not_available"
            elif not plan.start <= query.key < plan.end:
                status = "outside_key_scope"
            else:
                side = "old" if query.event_time < plan.cutover_time else "new"
                source = plan.old_source if side == "old" else plan.new_source
                for row in part_rows[plan.migration_id]:
                    comparisons += 1
                    part = request.partitions[row]
                    if (
                        part.side == side
                        and part.available_time <= request.decision_time
                        and part.start <= query.key < part.end
                    ):
                        candidates.append(row)
                if not candidates:
                    status = "missing_partition"
                elif len(candidates) > 1:
                    status = "ambiguous_partitions"
                elif candidates[0] in route_invalid_parts:
                    status = "invalid_partition"
                else:
                    status = "resolved"
                    selected = candidates[0]
                    maximum = max(plan.available_time, request.partitions[selected].available_time)
        query_counts[status] += 1
        if request.offset <= index < request.offset + request.limit:
            results.append(
                QueryResult(
                    query_row_index=index,
                    status=status,
                    plan_row_index=plan_index,
                    prescribed_side=side,
                    prescribed_source=source,
                    candidate_count=len(candidates),
                    candidate_partition_row_indexes=candidates[:10],
                    omitted_candidate_indexes=max(0, len(candidates) - 10),
                    selected_partition_row_index=selected,
                    maximum_source_available_time=maximum,
                )
            )
    return Output(
        passed=not issues,
        plan_count=len(request.plans),
        partition_count=len(request.partitions),
        query_count=len(request.queries),
        route_comparisons=comparisons,
        plans=plan_results,
        queries=results,
        query_status_counts=dict(sorted(query_counts.items())),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.queries),
    )


OPERATION = Operation(
    id="skills.audit_source_migrations",
    kind="skill",
    description="Audit source-partition cutover ownership and coverage, then route visible event/key queries to the prescribed source with no fallback.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
