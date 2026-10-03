"""Audit event ordering and numeric monotonicity in supplied group order.

Rows are never sorted: each group preserves its order in the input. Numeric
comparisons use adjacent supplied rows even if event ordering is invalid.
Repeated clocks are counted across the entire group, including nonadjacent
duplicates; timezone-equivalent instants are the same clock. Singleton groups
have no comparable pair and are explicitly unassessed.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, StrictFloat, StrictInt, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

GroupName = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Numeric = StrictInt | StrictFloat
Code = Literal["event_time_reversal", "duplicate_event_time", "numeric_direction_violation"]


class Observation(InputModel):
    group: GroupName = "default"
    event_time: AwareDatetime
    value: Numeric

    @field_validator("event_time", mode="before")
    @classmethod
    def require_explicit_datetime(cls, value: object) -> object:
        if not isinstance(value, (str, datetime)):
            raise ValueError("event_time must be a timezone-aware datetime or ISO datetime string")
        if isinstance(value, str):
            if len(value) > 64:
                raise ValueError("event_time string exceeds 64 characters")
            if not re.match(r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}", value):
                raise ValueError("event_time must use an ISO calendar datetime, not epoch text")
        return value

    @field_validator("event_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        try:
            return value.astimezone(UTC)
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                "event_time cannot be normalized into the supported UTC range"
            ) from exc


class Input(InputModel):
    observations: list[Observation] = Field(default_factory=list, max_length=10_000)
    direction: Literal["increasing", "decreasing"] = "increasing"
    strict: bool = Field(default=False, strict=True)
    allow_duplicate_event_times: bool = Field(default=False, strict=True)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def bound_groups(self) -> Self:
        if len({row.group for row in self.observations}) > 256:
            raise ValueError("observations may contain at most 256 groups")
        return self


class Finding(OutputModel):
    group: str
    code: Code
    is_violation: bool
    row_index: int
    related_row_index: int
    event_time: AwareDatetime
    related_event_time: AwareDatetime
    value: Numeric
    related_value: Numeric


class GroupAudit(OutputModel):
    group: str
    row_count: int
    comparable_pairs: int
    event_time_reversals: int
    duplicate_event_rows: int
    numeric_direction_violations: int
    passed: bool | None


class Output(OutputModel):
    row_count: int
    group_count: int
    direction: str
    strict: bool
    allow_duplicate_event_times: bool
    assessment: Literal["passed", "failed", "no_comparable_pairs"]
    passed: bool | None
    comparable_pairs: int
    assessed_groups: int
    unassessed_groups: int
    event_time_reversals: int
    duplicate_event_rows: int
    numeric_direction_violations: int
    violation_count: int
    diagnostic_count: int
    groups: list[GroupAudit]
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    grouped: dict[str, list[int]] = {}
    for index, observation in enumerate(request.observations):
        grouped.setdefault(observation.group, []).append(index)
    diagnostics: list[Finding] = []
    diagnostic_count = 0

    def record(group: str, code: Code, index: int, related: int, *, violation: bool) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) >= request.max_diagnostics:
            return
        row, previous = request.observations[index], request.observations[related]
        diagnostics.append(
            Finding(
                group=group,
                code=code,
                is_violation=violation,
                row_index=index,
                related_row_index=related,
                event_time=row.event_time,
                related_event_time=previous.event_time,
                value=row.value,
                related_value=previous.value,
            )
        )

    summaries: list[GroupAudit] = []
    total_reversals = total_duplicates = total_numeric = pairs = 0
    for group, indices in grouped.items():
        clocks: dict[datetime, int] = {}
        reversals = duplicates = numeric = 0
        previous_index: int | None = None
        for index in indices:
            row = request.observations[index]
            if row.event_time in clocks:
                duplicates += 1
                record(
                    group,
                    "duplicate_event_time",
                    index,
                    clocks[row.event_time],
                    violation=not request.allow_duplicate_event_times,
                )
            else:
                clocks[row.event_time] = index
            if previous_index is not None:
                previous = request.observations[previous_index]
                if row.event_time < previous.event_time:
                    reversals += 1
                    record(group, "event_time_reversal", index, previous_index, violation=True)
                wrong_direction = (
                    row.value < previous.value
                    if request.direction == "increasing"
                    else row.value > previous.value
                )
                if wrong_direction or (request.strict and row.value == previous.value):
                    numeric += 1
                    record(
                        group, "numeric_direction_violation", index, previous_index, violation=True
                    )
            previous_index = index
        group_pairs = len(indices) - 1
        group_failed = (
            reversals > 0
            or numeric > 0
            or (duplicates > 0 and not request.allow_duplicate_event_times)
        )
        summaries.append(
            GroupAudit(
                group=group,
                row_count=len(indices),
                comparable_pairs=group_pairs,
                event_time_reversals=reversals,
                duplicate_event_rows=duplicates,
                numeric_direction_violations=numeric,
                passed=not group_failed if group_pairs else None,
            )
        )
        pairs += group_pairs
        total_reversals += reversals
        total_duplicates += duplicates
        total_numeric += numeric
    violation_count = (
        total_reversals
        + total_numeric
        + (0 if request.allow_duplicate_event_times else total_duplicates)
    )
    passed = False if violation_count else (True if pairs else None)
    assessed = sum(summary.comparable_pairs > 0 for summary in summaries)
    return Output(
        row_count=len(request.observations),
        group_count=len(grouped),
        direction=request.direction,
        strict=request.strict,
        allow_duplicate_event_times=request.allow_duplicate_event_times,
        assessment="failed" if passed is False else ("passed" if passed else "no_comparable_pairs"),
        passed=passed,
        comparable_pairs=pairs,
        assessed_groups=assessed,
        unassessed_groups=len(grouped) - assessed,
        event_time_reversals=total_reversals,
        duplicate_event_rows=total_duplicates,
        numeric_direction_violations=total_numeric,
        violation_count=violation_count,
        diagnostic_count=diagnostic_count,
        groups=summaries,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_monotonic_sequences",
    kind="skill",
    description=(
        "Audit timezone-aware event order and increasing/decreasing numeric sequences per "
        "group without sorting away errors. Supports strict value ordering, reports global "
        "duplicate clocks, bounds diagnostics, and leaves singleton groups unassessed."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
