"""Audit staged version releases, causal inputs and declared permanent supersession.

stage_order is an explicit pipeline. Every supplied milestone requires all prior
stages; completion/publication must satisfy start <= completion <= availability,
and the previous stage must be available by the next stage's start. Dependencies
bind a consumer stage to an exact component/version/stage. The dependency's stage
and manifest must be available and effective, and not superseded, at consumer
start. Cross-component/version edges are evaluated at stage level, so interleaved
workflows are not falsely classified as cycles of whole release manifests.

Manifest availability is when its identity/metadata are knowable; effective
validity is [valid_from, valid_to). A terminal pipeline declaration activates at
max(valid_from, manifest availability, final-stage availability), provided that
clock precedes valid_to and all pipeline milestones have locally valid clocks.
An explicit supersedes edge takes permanent effect then and requires its target
to have activated strictly earlier. Expiry of a replacement never revives an old
version. Supersession reflects declared publication, independently of whether its
dependencies later pass this audit; an invalid replacement cannot cause silent
fallback to the retired version. No implicit version ordering is used.

Queries select an exact version or require exactly one observable, effective,
unsuperseded candidate at the requested stage. Candidate selection includes
invalid stages so they cannot be silently dropped in favor of another version.
Availability maxima are returned only for resolved, valid dependency chains.
passed describes declaration consistency; unresolved queries have explicit
statuses and do not themselves make otherwise consistent declarations invalid.

Bounds: 3000 manifests, 16 pipeline stages, 10000 total milestones, 20000 declared
dependencies, 10000 supersedes declarations and 1000 queries. Candidate-version
work is explicitly capped at 500000. Stage DAG processing is linear in nodes and
edges; diagnostic and query pages each hold at most 200 rows, with at most ten
candidate source indexes per result. No artifact bytes or deployments are verified.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]


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


class Milestone(InputModel):
    stage_id: Name
    started_time: AwareDatetime
    completed_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("started_time", "completed_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Dependency(InputModel):
    consumer_stage: Name
    component_id: Name
    version_id: Name
    stage_id: Name


class Release(InputModel):
    component_id: Name
    version_id: Name
    available_time: AwareDatetime
    valid_from: AwareDatetime
    valid_to: AwareDatetime | None = None
    milestones: list[Milestone] = Field(max_length=32)
    dependencies: list[Dependency] = Field(default_factory=list, max_length=128)
    supersedes: list[Name] = Field(default_factory=list, max_length=128)

    @field_validator("available_time", "valid_from", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("valid_to", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)


class Query(InputModel):
    component_id: Name
    version_id: Name | None = None
    stage_id: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    stage_order: list[Name] = Field(min_length=1, max_length=16)
    releases: list[Release] = Field(max_length=3000)
    queries: list[Query] = Field(default_factory=list, max_length=1000)
    max_candidate_versions: int = Field(default=100_000, strict=True, ge=0, le=500_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=1000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def work_bounds(self) -> Self:
        if len(set(self.stage_order)) != len(self.stage_order):
            raise ValueError("stage_order must contain unique stage IDs")
        if (
            sum(len(row.milestones) for row in self.releases) > 10_000
            or sum(len(row.dependencies) for row in self.releases) > 20_000
            or sum(len(row.supersedes) for row in self.releases) > 10_000
        ):
            raise ValueError("release milestone/dependency/supersession totals exceed their bounds")
        components = Counter(row.component_id for row in self.releases)
        versions = Counter((row.component_id, row.version_id) for row in self.releases)
        work = sum(
            components[row.component_id]
            if row.version_id is None
            else versions[row.component_id, row.version_id]
            for row in self.queries
        )
        if work > self.max_candidate_versions:
            raise ValueError("query candidate-version work exceeds max_candidate_versions")
        return self


class Finding(OutputModel):
    code: str
    release_row_index: int
    milestone_row_index: int | None = None
    dependency_row_index: int | None = None
    source_release_row_index: int | None = None
    source_milestone_row_index: int | None = None


class QueryResult(OutputModel):
    query_row_index: int
    status: Literal[
        "resolved",
        "unknown_stage",
        "unknown_component_or_version",
        "ambiguous_version_identity",
        "not_eligible",
        "ambiguous_active_versions",
        "invalid_stage",
    ]
    candidate_count: int
    candidate_release_row_indexes: list[int]
    omitted_candidate_indexes: int
    release_row_index: int | None
    milestone_row_index: int | None
    maximum_source_available_time: datetime | None
    superseded_at: datetime | None
    ineligibility_reason: (
        Literal["stage_not_declared", "not_available", "outside_effective_validity", "superseded"]
        | None
    )


class Output(OutputModel):
    passed: bool
    release_count: int
    milestone_count: int
    dependency_count: int
    declared_supersession_count: int
    applied_supersession_edge_count: int
    invalid_manifest_count: int
    invalid_stage_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    query_count: int
    candidate_versions_examined: int
    query_status_counts: dict[str, int]
    queries: list[QueryResult]
    offset: int
    has_more: bool
    artifact_contents_verified: Literal[False] = False
    deployment_or_publication_truth_verified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid_manifests: set[int] = set()
    invalid_nodes: set[int] = set()
    locations: list[tuple[int, int]] = []
    nodes: list[Milestone] = []
    release_nodes: list[list[int]] = [[] for _ in request.releases]
    stage_maps: list[dict[str, int]] = []
    declared_stage_availability: list[dict[str, datetime]] = []

    def report(
        code: str,
        release: int,
        node: int | None = None,
        dependency: int | None = None,
        source: int | None = None,
        source_node: int | None = None,
    ) -> None:
        issues[code] += 1
        if node is None:
            invalid_manifests.add(release)
        else:
            invalid_nodes.add(node)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    release_row_index=release,
                    milestone_row_index=None if node is None else locations[node][1],
                    dependency_row_index=dependency,
                    source_release_row_index=source,
                    source_milestone_row_index=None
                    if source_node is None
                    else locations[source_node][1],
                )
            )

    versions: dict[tuple[str, str], list[int]] = defaultdict(list)
    components: dict[str, list[int]] = defaultdict(list)
    for index, release in enumerate(request.releases):
        versions[release.component_id, release.version_id].append(index)
        components[release.component_id].append(index)
        counts = Counter(row.stage_id for row in release.milestones)
        mapping: dict[str, int] = {}
        stage_available: dict[str, datetime] = {}
        for local_index, milestone in enumerate(release.milestones):
            node = len(nodes)
            nodes.append(milestone)
            locations.append((index, local_index))
            release_nodes[index].append(node)
            previous_available = stage_available.get(milestone.stage_id)
            stage_available[milestone.stage_id] = (
                milestone.available_time
                if previous_available is None
                else min(previous_available, milestone.available_time)
            )
            if counts[milestone.stage_id] == 1:
                mapping[milestone.stage_id] = node
            else:
                report("ambiguous_stage_id", release=index, node=node)
            if milestone.stage_id not in request.stage_order:
                report("unknown_pipeline_stage", release=index, node=node)
            if not milestone.started_time <= milestone.completed_time <= milestone.available_time:
                report("invalid_stage_clock_order", release=index, node=node)
        stage_maps.append(mapping)
        declared_stage_availability.append(stage_available)
    unique = {key: rows[0] for key, rows in versions.items() if len(rows) == 1}
    for index, release in enumerate(request.releases):
        if len(versions[release.component_id, release.version_id]) != 1:
            report("ambiguous_component_version", release=index)
        if release.valid_to is not None and release.valid_to <= release.valid_from:
            report("empty_or_inverted_effective_validity", release=index)

    children: list[list[tuple[int, int | None]]] = [[] for _ in nodes]
    indegree = [0] * len(nodes)
    for index, mapping in enumerate(stage_maps):
        for stage_position, stage in enumerate(request.stage_order[1:], start=1):
            pipeline_node = mapping.get(stage)
            if pipeline_node is None:
                continue
            previous = mapping.get(request.stage_order[stage_position - 1])
            if previous is None:
                report("missing_previous_pipeline_stage", release=index, node=pipeline_node)
                continue
            children[previous].append((pipeline_node, None))
            indegree[pipeline_node] += 1
            if nodes[previous].available_time > nodes[pipeline_node].started_time:
                report(
                    "stage_starts_before_prior_publication",
                    release=index,
                    node=pipeline_node,
                    source=index,
                    source_node=previous,
                )

    activation: list[datetime | None] = [None] * len(request.releases)
    for index, release in enumerate(request.releases):
        final = stage_maps[index].get(request.stage_order[-1])
        pipeline = [stage_maps[index].get(stage) for stage in request.stage_order]
        if (
            index in invalid_manifests
            or final is None
            or any(node is None or node in invalid_nodes for node in pipeline)
        ):
            continue
        clock = max(release.valid_from, release.available_time, nodes[final].available_time)
        if release.valid_to is None or clock < release.valid_to:
            activation[index] = clock
    superseded: list[datetime | None] = [None] * len(request.releases)
    supersession_edges = 0
    for index, release in enumerate(request.releases):
        for target_version, count in Counter(release.supersedes).items():
            if count > 1:
                report("duplicate_supersedes_declaration", release=index)
            target = unique.get((release.component_id, target_version))
            if target is None:
                report("unresolved_superseded_version", release=index)
                continue
            supersession_clock = activation[index]
            if supersession_clock is None:
                continue
            target_clock = activation[target]
            if target_clock is None or target_clock >= supersession_clock:
                report(
                    "supersession_requires_earlier_activated_version", release=index, source=target
                )
                continue
            prior = superseded[target]
            superseded[target] = (
                supersession_clock if prior is None else min(prior, supersession_clock)
            )
            supersession_edges += 1

    for index, release in enumerate(request.releases):
        seen: set[tuple[str, str, str, str]] = set()
        for dependency_index, dependency in enumerate(release.dependencies):
            key = (
                dependency.consumer_stage,
                dependency.component_id,
                dependency.version_id,
                dependency.stage_id,
            )
            consumer = stage_maps[index].get(dependency.consumer_stage)
            if key in seen:
                report(
                    "duplicate_dependency_declaration",
                    release=index,
                    node=consumer,
                    dependency=dependency_index,
                )
            seen.add(key)
            if consumer is None:
                report("unresolved_consumer_stage", release=index, dependency=dependency_index)
                continue
            source = unique.get((dependency.component_id, dependency.version_id))
            source_node = None if source is None else stage_maps[source].get(dependency.stage_id)
            if source is None or source_node is None:
                report(
                    "unresolved_dependency_version_stage",
                    release=index,
                    node=consumer,
                    dependency=dependency_index,
                    source=source,
                )
                continue
            children[source_node].append((consumer, dependency_index))
            indegree[consumer] += 1
            source_release, source_stage = request.releases[source], nodes[source_node]
            start = nodes[consumer].started_time
            if max(source_release.available_time, source_stage.available_time) > start:
                report(
                    "dependency_not_available_at_stage_start",
                    release=index,
                    node=consumer,
                    dependency=dependency_index,
                    source=source,
                    source_node=source_node,
                )
            if start < source_release.valid_from or (
                source_release.valid_to is not None and start >= source_release.valid_to
            ):
                report(
                    "dependency_outside_effective_validity",
                    release=index,
                    node=consumer,
                    dependency=dependency_index,
                    source=source,
                    source_node=source_node,
                )
            retired = superseded[source]
            if retired is not None and start >= retired:
                report(
                    "dependency_version_superseded",
                    release=index,
                    node=consumer,
                    dependency=dependency_index,
                    source=source,
                    source_node=source_node,
                )
    for index in invalid_manifests:
        invalid_nodes.update(release_nodes[index])
    maximum = [
        max(request.releases[release].available_time, nodes[node].available_time)
        for node, (release, _) in enumerate(locations)
    ]
    pending = deque(node for node, degree in enumerate(indegree) if degree == 0)
    visited: set[int] = set()
    while pending:
        node = pending.popleft()
        visited.add(node)
        for target, edge_dependency in children[node]:
            maximum[target] = max(maximum[target], maximum[node])
            if node in invalid_nodes:
                report(
                    "invalid_dependency_or_prior_stage",
                    release=locations[target][0],
                    node=target,
                    dependency=edge_dependency,
                    source=locations[node][0],
                    source_node=node,
                )
            indegree[target] -= 1
            if indegree[target] == 0:
                pending.append(target)
    for node in range(len(nodes)):
        if node not in visited:
            report("cyclic_or_cycle_dependent_stage", release=locations[node][0], node=node)

    query_results: list[QueryResult] = []
    query_counts: Counter[str] = Counter()
    work = 0
    for index, query in enumerate(request.queries):
        declared = (
            components[query.component_id]
            if query.version_id is None
            else versions[query.component_id, query.version_id]
        )
        work += len(declared)
        candidates: list[tuple[int, int | None]] = []
        for release_index in declared:
            release = request.releases[release_index]
            query_node = stage_maps[release_index].get(query.stage_id)
            stage_available_time = declared_stage_availability[release_index].get(query.stage_id)
            retired = superseded[release_index]
            if (
                stage_available_time is not None
                and release.valid_from <= query.decision_time
                and (release.valid_to is None or query.decision_time < release.valid_to)
                and max(release.available_time, stage_available_time) <= query.decision_time
                and (retired is None or query.decision_time < retired)
            ):
                candidates.append((release_index, query_node))
        status: Literal[
            "resolved",
            "unknown_stage",
            "unknown_component_or_version",
            "ambiguous_version_identity",
            "not_eligible",
            "ambiguous_active_versions",
            "invalid_stage",
        ]
        selected_release = selected_local = None
        resolved_maximum = retired_at = None
        reason: (
            Literal[
                "stage_not_declared", "not_available", "outside_effective_validity", "superseded"
            ]
            | None
        ) = None
        if query.stage_id not in request.stage_order:
            status = "unknown_stage"
        elif not declared:
            status = "unknown_component_or_version"
        elif query.version_id is not None and len(declared) != 1:
            status = "ambiguous_version_identity"
        elif not candidates:
            status = "not_eligible"
            if query.version_id is not None:
                selected_release = declared[0]
                selected_node = stage_maps[selected_release].get(query.stage_id)
                selected_local = None if selected_node is None else locations[selected_node][1]
                declaration = request.releases[selected_release]
                stage_clock = declared_stage_availability[selected_release].get(query.stage_id)
                retirement = superseded[selected_release]
                retired_at = (
                    retirement
                    if retirement is not None and retirement <= query.decision_time
                    else None
                )
                if stage_clock is None:
                    reason = "stage_not_declared"
                elif max(declaration.available_time, stage_clock) > query.decision_time:
                    reason = "not_available"
                elif retired_at is not None:
                    reason = "superseded"
                else:
                    reason = "outside_effective_validity"
        elif len(candidates) > 1:
            status = "ambiguous_active_versions"
        else:
            selected_release, selected_node = candidates[0]
            selected_local = None if selected_node is None else locations[selected_node][1]
            retirement = superseded[selected_release]
            retired_at = (
                retirement if retirement is not None and retirement <= query.decision_time else None
            )
            status = (
                "invalid_stage"
                if selected_node is None or selected_node in invalid_nodes
                else "resolved"
            )
            if status == "resolved" and selected_node is not None:
                resolved_maximum = maximum[selected_node]
        query_counts[status] += 1
        if request.offset <= index < request.offset + request.limit:
            query_results.append(
                QueryResult(
                    query_row_index=index,
                    status=status,
                    candidate_count=len(candidates),
                    candidate_release_row_indexes=[row for row, _ in candidates[:10]],
                    omitted_candidate_indexes=max(0, len(candidates) - 10),
                    release_row_index=selected_release,
                    milestone_row_index=selected_local,
                    maximum_source_available_time=resolved_maximum,
                    superseded_at=retired_at,
                    ineligibility_reason=reason,
                )
            )
    return Output(
        passed=not issues,
        release_count=len(request.releases),
        milestone_count=len(nodes),
        dependency_count=sum(len(row.dependencies) for row in request.releases),
        declared_supersession_count=sum(len(row.supersedes) for row in request.releases),
        applied_supersession_edge_count=supersession_edges,
        invalid_manifest_count=len(invalid_manifests),
        invalid_stage_count=len(invalid_nodes),
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        query_count=len(request.queries),
        candidate_versions_examined=work,
        query_status_counts=dict(sorted(query_counts.items())),
        queries=query_results,
        offset=request.offset,
        has_more=request.offset + len(query_results) < len(request.queries),
    )


OPERATION = Operation(
    id="skills.audit_dependency_releases",
    kind="skill",
    description="Audit exact version/stage release dependencies, causal availability, explicit supersession and decision-time eligibility without inferring version order.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
