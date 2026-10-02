"""Distinguish identical repeats from contradictory records of the same vintage."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StrictBool, StrictFloat, StrictInt, field_validator

from fx1.operations.base import (
    InputModel,
    Operation,
    OperationContext,
    OutputModel,
    canonical_json,
)

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Cell = (
    Annotated[str, Field(strict=True, max_length=1024)]
    | StrictBool
    | StrictInt
    | StrictFloat
    | None
)
ConflictField = Literal["source", "available_time", "values"]


def _aware_clock(value: object) -> datetime:
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


class Record(InputModel):
    security_id: Name
    event_time: AwareDatetime
    revision_id: Name
    available_time: AwareDatetime
    source: Name
    values: dict[Name, Cell] = Field(min_length=1, max_length=32)

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _aware_clock(value)


class Input(InputModel):
    records: list[Record] = Field(min_length=1, max_length=10_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    max_indices_per_group: int = Field(default=20, strict=True, ge=2, le=200)


class Finding(OutputModel):
    security_id: str
    event_time: datetime
    revision_id: str
    status: Literal["conflicting_revision", "exact_duplicates_only"]
    conflict_fields: list[ConflictField]
    row_count: int
    unique_variant_count: int
    exact_duplicate_rows: int
    row_indices: list[int]
    omitted_row_indices: int


class Output(OutputModel):
    input_rows: int
    revision_group_count: int
    conflicting_revision_groups: int
    rows_in_conflicting_groups: int
    exact_duplicate_rows: int
    exact_duplicate_only_groups: int
    conflict_field_counts: dict[str, int]
    conflict_free: bool
    unique_and_consistent: bool
    value_comparison: Literal["canonical_json_type_sensitive"] = "canonical_json_type_sensitive"
    finding_order: Literal["conflicts_first_then_exact_duplicates_by_group_key"] = (
        "conflicts_first_then_exact_duplicates_by_group_key"
    )
    findings: list[Finding]
    omitted_findings: int


def execute(request: Input, context: OperationContext) -> Output:
    groups: dict[tuple[str, datetime, str], list[int]] = defaultdict(list)
    for index, record in enumerate(request.records):
        groups[(record.security_id, record.event_time.astimezone(UTC), record.revision_id)].append(
            index
        )

    conflict_findings: list[Finding] = []
    duplicate_findings: list[Finding] = []
    conflict_count = conflict_rows = duplicate_rows = duplicate_only_groups = finding_count = 0
    field_counts: Counter[str] = Counter()
    for (security, event_time, revision_id), indices in sorted(groups.items()):
        if len(indices) == 1:
            continue
        sources: set[str] = set()
        availability_clocks: set[datetime] = set()
        payloads: set[bytes] = set()
        variants: dict[tuple[str, datetime, bytes], int] = {}
        for index in indices:
            record = request.records[index]
            available = record.available_time.astimezone(UTC)
            payload = canonical_json(record.values)
            sources.add(record.source)
            availability_clocks.add(available)
            payloads.add(payload)
            variants.setdefault((record.source, available, payload), index)
        fields: list[ConflictField] = []
        if len(sources) > 1:
            fields.append("source")
        if len(availability_clocks) > 1:
            fields.append("available_time")
        if len(payloads) > 1:
            fields.append("values")
        repeats = len(indices) - len(variants)
        duplicate_rows += repeats
        if fields:
            conflict_count += 1
            conflict_rows += len(indices)
            field_counts.update(fields)
        else:
            duplicate_only_groups += 1
        finding_count += 1
        destination = conflict_findings if fields else duplicate_findings
        if len(destination) < request.max_diagnostics:
            # Include differing variants before repeated rows, so a diagnostic
            # does not show only identical leading rows while hiding its conflict.
            chosen_indices = set(list(variants.values())[: request.max_indices_per_group])
            for index in indices:
                if len(chosen_indices) >= request.max_indices_per_group:
                    break
                chosen_indices.add(index)
            examples = sorted(chosen_indices)
            destination.append(
                Finding(
                    security_id=security,
                    event_time=event_time,
                    revision_id=revision_id,
                    status="conflicting_revision" if fields else "exact_duplicates_only",
                    conflict_fields=fields,
                    row_count=len(indices),
                    unique_variant_count=len(variants),
                    exact_duplicate_rows=repeats,
                    row_indices=examples,
                    omitted_row_indices=len(indices) - len(examples),
                )
            )

    findings = (conflict_findings + duplicate_findings)[: request.max_diagnostics]
    return Output(
        input_rows=len(request.records),
        revision_group_count=len(groups),
        conflicting_revision_groups=conflict_count,
        rows_in_conflicting_groups=conflict_rows,
        exact_duplicate_rows=duplicate_rows,
        exact_duplicate_only_groups=duplicate_only_groups,
        conflict_field_counts=dict(sorted(field_counts.items())),
        conflict_free=conflict_count == 0,
        unique_and_consistent=conflict_count == 0 and duplicate_rows == 0,
        findings=findings,
        omitted_findings=finding_count - len(findings),
    )


OPERATION = Operation(
    id="skills.audit_revision_conflicts",
    kind="skill",
    description=(
        "Audit records grouped by security/event/revision, distinguish exact repeats "
        "from contradictory values, availability, or source, and return exact aggregate "
        "counts with bounded row-index diagnostics. Timestamps compare as UTC instants."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
