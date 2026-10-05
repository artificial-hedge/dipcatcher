"""Audit supplied feature lineage against concrete source clocks and field versions.

Source identifiers must identify exactly one input source record. A duplicate id
is ambiguous even when its records look equal; no row is silently selected.
The maximum clock is certified only when every declared dependency resolves.
This audit checks caller-supplied lineage consistency, not source authenticity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]


def _clock(value: object) -> datetime:
    if isinstance(value, str):
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


class SourceRecord(InputModel):
    source_id: Name
    source: Name
    revision_id: Name
    available_time: AwareDatetime
    field_versions: dict[Name, Name] = Field(min_length=1, max_length=64)

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Dependency(InputModel):
    feature_field: Name
    source_id: Name
    source_field: Name
    source_field_version: Name


class FeatureRow(InputModel):
    security_id: Name
    decision_time: AwareDatetime
    max_source_available_time: AwareDatetime | None = None
    feature_set_version: Name | None = None
    feature_versions: dict[Name, Name | None] = Field(min_length=1, max_length=64)
    dependencies: list[Dependency] = Field(max_length=128)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("max_source_available_time", mode="before")
    @classmethod
    def optional_clock(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)


class Input(InputModel):
    sources: list[SourceRecord] = Field(max_length=10_000)
    features: list[FeatureRow] = Field(min_length=1, max_length=10_000)
    required_feature_versions: dict[Name, Name] = Field(default_factory=dict, max_length=64)
    required_feature_set_version: Name | None = None
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def bounded_dependency_work(self) -> Self:
        if sum(len(row.dependencies) for row in self.features) > 100_000:
            raise ValueError("at most 100000 dependencies may be audited in one call")
        return self


class Finding(OutputModel):
    code: str
    feature_row_index: int | None = None
    feature_field: str | None = None
    source_id: str | None = None
    source_row_index: int | None = None
    dependency_index: int | None = None


class RowAudit(OutputModel):
    feature_row_index: int
    security_id: str
    valid: bool
    issue_count: int
    dependency_count: int
    resolved_source_count: int
    complete_reference_resolution: bool
    actual_max_source_available_time: datetime | None
    claimed_max_source_available_time: datetime | None


class Output(OutputModel):
    feature_row_count: int
    valid_feature_rows: int
    invalid_feature_rows: int
    source_row_count: int
    duplicate_source_ids: int
    passed: bool
    issue_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    offset: int
    next_offset: int | None
    rows: list[RowAudit]


def execute(request: Input, context: OperationContext) -> Output:
    source_indices: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(request.sources):
        source_indices[source.source_id].append(index)
    counts: Counter[str] = Counter()
    diagnostics: list[Finding] = []

    def report(finding: Finding) -> None:
        counts[finding.code] += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    duplicated_ids = 0
    for source_id, indices in sorted(source_indices.items()):
        if len(indices) > 1:
            duplicated_ids += 1
            report(
                Finding(
                    code="duplicate_source_id", source_id=source_id, source_row_index=indices[1]
                )
            )

    audits: list[RowAudit] = []
    invalid = 0
    for row_index, row in enumerate(request.features):
        issue_count = 0

        def row_issue(
            code: str,
            *,
            field: str | None = None,
            source_id: str | None = None,
            source_index: int | None = None,
            dependency_index: int | None = None,
            _feature_row_index: int = row_index,
        ) -> None:
            nonlocal issue_count
            issue_count += 1
            report(
                Finding(
                    code=code,
                    feature_row_index=_feature_row_index,
                    feature_field=field,
                    source_id=source_id,
                    source_row_index=source_index,
                    dependency_index=dependency_index,
                )
            )

        if row.feature_set_version is None:
            row_issue("missing_feature_set_version")
        elif (
            request.required_feature_set_version is not None
            and row.feature_set_version != request.required_feature_set_version
        ):
            row_issue("feature_set_version_mismatch")
        for field, version in row.feature_versions.items():
            if version is None:
                row_issue("missing_feature_field_version", field=field)
            elif (
                field in request.required_feature_versions
                and version != request.required_feature_versions[field]
            ):
                row_issue("feature_field_version_mismatch", field=field)
        for field in request.required_feature_versions:
            if field not in row.feature_versions:
                row_issue("missing_required_feature_field", field=field)

        if row.max_source_available_time is None:
            row_issue("missing_claimed_max_availability")
        elif row.max_source_available_time > row.decision_time:
            row_issue("claimed_availability_after_decision")

        dependency_fields: set[str] = set()
        dependencies_seen: set[tuple[str, str, str]] = set()
        resolved_ids: set[str] = set()
        checked_future_ids: set[str] = set()
        complete = bool(row.dependencies)
        actual_max: datetime | None = None
        for dependency_index, dependency in enumerate(row.dependencies):
            field = dependency.feature_field
            dependency_fields.add(field)
            key = field, dependency.source_id, dependency.source_field
            if key in dependencies_seen:
                complete = False
                row_issue(
                    "duplicate_dependency",
                    field=field,
                    source_id=dependency.source_id,
                    dependency_index=dependency_index,
                )
            dependencies_seen.add(key)
            if field not in row.feature_versions:
                complete = False
                row_issue(
                    "dependency_for_undeclared_feature",
                    field=field,
                    source_id=dependency.source_id,
                    dependency_index=dependency_index,
                )
            indices = source_indices.get(dependency.source_id, [])
            if not indices:
                complete = False
                row_issue(
                    "dangling_source_reference",
                    field=field,
                    source_id=dependency.source_id,
                    dependency_index=dependency_index,
                )
                continue
            if len(indices) != 1:
                complete = False
                row_issue(
                    "ambiguous_source_reference",
                    field=field,
                    source_id=dependency.source_id,
                    dependency_index=dependency_index,
                )
                continue
            source_index = indices[0]
            source = request.sources[source_index]
            resolved_ids.add(source.source_id)
            if actual_max is None or source.available_time > actual_max:
                actual_max = source.available_time
            if (
                source.available_time > row.decision_time
                and source.source_id not in checked_future_ids
            ):
                checked_future_ids.add(source.source_id)
                row_issue(
                    "source_available_after_decision",
                    source_id=source.source_id,
                    source_index=source_index,
                )
            if dependency.source_field not in source.field_versions:
                complete = False
                row_issue(
                    "missing_source_field",
                    field=field,
                    source_id=source.source_id,
                    source_index=source_index,
                    dependency_index=dependency_index,
                )
            elif source.field_versions[dependency.source_field] != dependency.source_field_version:
                complete = False
                row_issue(
                    "source_field_version_mismatch",
                    field=field,
                    source_id=source.source_id,
                    source_index=source_index,
                    dependency_index=dependency_index,
                )

        for field in row.feature_versions:
            if field not in dependency_fields:
                complete = False
                row_issue("missing_feature_dependencies", field=field)
        if (
            complete
            and actual_max is not None
            and row.max_source_available_time is not None
            and actual_max != row.max_source_available_time
        ):
            row_issue("claimed_max_availability_mismatch")
        invalid += int(issue_count > 0)
        if request.offset <= row_index < request.offset + request.limit:
            audits.append(
                RowAudit(
                    feature_row_index=row_index,
                    security_id=row.security_id,
                    valid=issue_count == 0,
                    issue_count=issue_count,
                    dependency_count=len(row.dependencies),
                    resolved_source_count=len(resolved_ids),
                    complete_reference_resolution=complete,
                    actual_max_source_available_time=actual_max,
                    claimed_max_source_available_time=row.max_source_available_time,
                )
            )
    total_issues = sum(counts.values())
    stop = request.offset + len(audits)
    return Output(
        feature_row_count=len(request.features),
        valid_feature_rows=len(request.features) - invalid,
        invalid_feature_rows=invalid,
        source_row_count=len(request.sources),
        duplicate_source_ids=duplicated_ids,
        passed=total_issues == 0,
        issue_count=total_issues,
        issue_counts=dict(sorted(counts.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=total_issues - len(diagnostics),
        offset=request.offset,
        next_offset=stop if stop < len(request.features) else None,
        rows=audits,
    )


OPERATION = Operation(
    id="skills.audit_feature_provenance",
    kind="skill",
    description=(
        "Audit supplied feature/source lineage: availability maxima and decision clocks, "
        "dangling or ambiguous sources, duplicate dependencies, missing dependencies, and "
        "feature/source field versions. This checks consistency, not source authenticity."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
