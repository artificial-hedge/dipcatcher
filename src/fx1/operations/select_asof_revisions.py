"""Select one observable vintage per security/event while preserving input lineage.

Availability is the vintage clock. Equal-clock rows must agree on source and
payload; different revision labels then use lexicographically greatest label,
followed by earliest input row. This is an explicit tie rule, not an assertion
that revision labels sort chronologically. Announced future events are allowed
unless the caller explicitly requests completed events.
"""

from __future__ import annotations

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
    available_time: AwareDatetime
    revision_id: Name
    source: Name
    values: dict[Name, Cell] = Field(min_length=1, max_length=32)

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _aware_clock(value)


class Input(InputModel):
    records: list[Record] = Field(min_length=1, max_length=10_000)
    decision_time: AwareDatetime
    require_event_by_decision: bool = Field(default=False, strict=True)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=1000, strict=True, ge=1, le=1000)

    @field_validator("decision_time", mode="before")
    @classmethod
    def validate_decision_clock(cls, value: object) -> datetime:
        return _aware_clock(value)

    @field_validator("records")
    @classmethod
    def consistent_same_clock_records(cls, records: list[Record]) -> list[Record]:
        clocks: dict[tuple[str, datetime, datetime], tuple[bytes, int]] = {}
        for index, record in enumerate(records):
            key = (
                record.security_id,
                record.event_time.astimezone(UTC),
                record.available_time.astimezone(UTC),
            )
            fingerprint = canonical_json({"source": record.source, "values": record.values})
            existing = clocks.get(key)
            if existing is not None and existing[0] != fingerprint:
                raise ValueError(
                    f"conflicting source or values at the same event/availability clock "
                    f"in rows {existing[1]} and {index}"
                )
            if existing is None:
                clocks[key] = fingerprint, index
        return records


class SelectedRecord(OutputModel):
    input_row_index: int
    eligible_vintage_rows: int
    record: Record


class Output(OutputModel):
    input_rows: int
    eligible_rows: int
    unavailable_rows: int
    future_event_rows_excluded: int
    selected_event_count: int
    superseded_or_duplicate_rows: int
    decision_time: datetime
    require_event_by_decision: bool
    tie_policy: Literal["latest_availability_greatest_revision_id_earliest_input_row"] = (
        "latest_availability_greatest_revision_id_earliest_input_row"
    )
    ordering: Literal["security_id_then_event_time_ascending"] = (
        "security_id_then_event_time_ascending"
    )
    offset: int
    next_offset: int | None
    selected: list[SelectedRecord]


def execute(request: Input, context: OperationContext) -> Output:
    decision = request.decision_time.astimezone(UTC)
    selected: dict[tuple[str, datetime], tuple[datetime, str, int]] = {}
    row_counts: dict[tuple[str, datetime], int] = {}
    unavailable = future_events = eligible = 0

    for index, record in enumerate(request.records):
        event_time = record.event_time.astimezone(UTC)
        availability = record.available_time.astimezone(UTC)
        if availability > decision:
            unavailable += 1
            continue
        if request.require_event_by_decision and event_time > decision:
            future_events += 1
            continue
        eligible += 1
        key = record.security_id, event_time
        row_counts[key] = row_counts.get(key, 0) + 1
        # Negative index gives earliest input row priority for exact ties.
        priority = availability, record.revision_id, -index
        if key not in selected or priority > selected[key]:
            selected[key] = priority

    ordered_keys = sorted(selected)
    page_keys = ordered_keys[request.offset : request.offset + request.limit]
    page = [
        SelectedRecord(
            input_row_index=-selected[key][2],
            eligible_vintage_rows=row_counts[key],
            record=request.records[-selected[key][2]],
        )
        for key in page_keys
    ]
    stop = request.offset + len(page)
    return Output(
        input_rows=len(request.records),
        eligible_rows=eligible,
        unavailable_rows=unavailable,
        future_event_rows_excluded=future_events,
        selected_event_count=len(selected),
        superseded_or_duplicate_rows=eligible - len(selected),
        decision_time=decision,
        require_event_by_decision=request.require_event_by_decision,
        offset=request.offset,
        next_offset=stop if stop < len(selected) else None,
        selected=page,
    )


OPERATION = Operation(
    id="skills.select_asof_revisions",
    kind="skill",
    description=(
        "Select the latest observable vintage for each security/event from caller-supplied "
        "records, reject same-clock source/value conflicts, preserve input-row lineage, "
        "and page results. Future announced events are allowed unless explicitly excluded."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
