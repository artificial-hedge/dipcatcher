"""Compare two explicit snapshots without interpreting omitted rows as tombstones.

Entity keys are unique within each snapshot. Field equality is type-sensitive
canonical JSON, so false, 0 and 0.0 are distinct. Availability regressions and
changes to source/availability/values under an unchanged revision ID are issues;
ordinary additions, removals and newly versioned changes are described.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel, canonical_json

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Cell = (
    Annotated[str, Field(strict=True, max_length=1024)]
    | StrictBool
    | StrictInt
    | StrictFloat
    | None
)


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


class SnapshotRow(InputModel):
    entity_key: Name
    revision_id: Name
    source: Name
    available_time: AwareDatetime
    values: dict[Name, Cell] = Field(min_length=1, max_length=32)

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    before_rows: list[SnapshotRow] = Field(max_length=10_000)
    after_rows: list[SnapshotRow] = Field(max_length=10_000)
    before_asof: AwareDatetime
    after_asof: AwareDatetime
    max_examples: int = Field(default=50, strict=True, ge=0, le=200)
    max_field_changes_per_example: int = Field(default=8, strict=True, ge=1, le=16)

    @field_validator("before_asof", "after_asof", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def ordered_unique_snapshots(self) -> Self:
        if self.after_asof < self.before_asof:
            raise ValueError("after_asof cannot precede before_asof")
        for name, rows in (("before", self.before_rows), ("after", self.after_rows)):
            if len({row.entity_key for row in rows}) != len(rows):
                raise ValueError(f"{name} snapshot entity keys must be unique")
        if sum(len(row.values) for row in (*self.before_rows, *self.after_rows)) > 250_000:
            raise ValueError("snapshots may contain at most 250000 value fields combined")
        return self


class FieldChange(OutputModel):
    field: str
    kind: Literal["added", "removed", "changed"]
    before_value: Cell
    after_value: Cell


class Change(OutputModel):
    entity_key: str
    kind: Literal["added", "removed", "modified"]
    before_row_index: int | None
    after_row_index: int | None
    revision_changed: bool
    source_changed: bool
    availability_changed: bool
    field_change_count: int
    field_changes: list[FieldChange]
    omitted_field_changes: int


class Finding(OutputModel):
    code: Literal[
        "not_observable_at_snapshot", "availability_regression", "revision_reuse_conflict"
    ]
    entity_key: str
    before_row_index: int | None = None
    after_row_index: int | None = None
    snapshot: Literal["before", "after"] | None = None


class Output(OutputModel):
    before_row_count: int
    after_row_count: int
    added_entities: int
    removed_entities: int
    modified_entities: int
    unchanged_entities: int
    fields_added: int
    fields_removed: int
    fields_changed: int
    availability_regressions: int
    revision_reuse_conflicts: int
    before_unobservable_rows: int
    after_unobservable_rows: int
    passed: bool
    removal_semantics: str = "absent_from_after_snapshot_not_an_asserted_delete"
    field_comparison: str = "canonical_json_type_sensitive"
    changes: list[Change]
    omitted_changed_entities: int
    issue_count: int
    findings: list[Finding]
    omitted_findings: int


def execute(request: Input, context: OperationContext) -> Output:
    before = {row.entity_key: index for index, row in enumerate(request.before_rows)}
    after = {row.entity_key: index for index, row in enumerate(request.after_rows)}
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    changes: list[Change] = []

    def issue(finding: Finding) -> None:
        issues[finding.code] += 1
        if len(findings) < request.max_examples:
            findings.append(finding)

    for side, rows, clock in (
        ("before", request.before_rows, request.before_asof),
        ("after", request.after_rows, request.after_asof),
    ):
        for index, row in enumerate(rows):
            if row.available_time > clock:
                counts[f"{side}_unobservable"] += 1
                issue(
                    Finding(
                        code="not_observable_at_snapshot",
                        entity_key=row.entity_key,
                        before_row_index=index if side == "before" else None,
                        after_row_index=index if side == "after" else None,
                        snapshot="before" if side == "before" else "after",
                    )
                )

    for key in sorted(before.keys() | after.keys()):
        before_index, after_index = before.get(key), after.get(key)
        prior = request.before_rows[before_index] if before_index is not None else None
        current = request.after_rows[after_index] if after_index is not None else None
        before_values = prior.values if prior is not None else {}
        after_values = current.values if current is not None else {}
        changed_fields: list[FieldChange] = []
        field_count = 0
        for field in sorted(before_values.keys() | after_values.keys()):
            field_kind: Literal["added", "removed", "changed"] | None = None
            if field not in before_values:
                field_kind = "added"
            elif field not in after_values:
                field_kind = "removed"
            elif canonical_json(before_values[field]) != canonical_json(after_values[field]):
                field_kind = "changed"
            if field_kind is not None:
                field_count += 1
                counts[f"field_{field_kind}"] += 1
                if len(changed_fields) < request.max_field_changes_per_example:
                    changed_fields.append(
                        FieldChange(
                            field=field,
                            kind=field_kind,
                            before_value=before_values.get(field),
                            after_value=after_values.get(field),
                        )
                    )
        revision_changed = (
            prior is not None and current is not None and prior.revision_id != current.revision_id
        )
        source_changed = (
            prior is not None and current is not None and prior.source != current.source
        )
        availability_changed = (
            prior is not None
            and current is not None
            and prior.available_time != current.available_time
        )
        if prior is None:
            kind: Literal["added", "removed", "modified"] = "added"
        elif current is None:
            kind = "removed"
        else:
            if current.available_time < prior.available_time:
                issue(
                    Finding(
                        code="availability_regression",
                        entity_key=key,
                        before_row_index=before_index,
                        after_row_index=after_index,
                    )
                )
            if not revision_changed and (field_count > 0 or source_changed or availability_changed):
                issue(
                    Finding(
                        code="revision_reuse_conflict",
                        entity_key=key,
                        before_row_index=before_index,
                        after_row_index=after_index,
                    )
                )
            if not (revision_changed or source_changed or availability_changed or field_count):
                counts["unchanged"] += 1
                continue
            kind = "modified"
        counts[kind] += 1
        if len(changes) < request.max_examples:
            changes.append(
                Change(
                    entity_key=key,
                    kind=kind,
                    before_row_index=before_index,
                    after_row_index=after_index,
                    revision_changed=revision_changed,
                    source_changed=source_changed,
                    availability_changed=availability_changed,
                    field_change_count=field_count,
                    field_changes=changed_fields,
                    omitted_field_changes=field_count - len(changed_fields),
                )
            )

    total_issues = sum(issues.values())
    return Output(
        before_row_count=len(before),
        after_row_count=len(after),
        added_entities=counts["added"],
        removed_entities=counts["removed"],
        modified_entities=counts["modified"],
        unchanged_entities=counts["unchanged"],
        fields_added=counts["field_added"],
        fields_removed=counts["field_removed"],
        fields_changed=counts["field_changed"],
        availability_regressions=issues["availability_regression"],
        revision_reuse_conflicts=issues["revision_reuse_conflict"],
        before_unobservable_rows=counts["before_unobservable"],
        after_unobservable_rows=counts["after_unobservable"],
        passed=total_issues == 0,
        changes=changes,
        omitted_changed_entities=counts["added"]
        + counts["removed"]
        + counts["modified"]
        - len(changes),
        issue_count=total_issues,
        findings=findings,
        omitted_findings=total_issues - len(findings),
    )


OPERATION = Operation(
    id="skills.audit_vintage_transitions",
    kind="skill",
    description="Compare explicit before/after snapshots for entity additions/removals and typed field changes, availability regressions, unobservable records, and conflicting reuse of revision IDs; retain bounded row lineage without inferring tombstones.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
