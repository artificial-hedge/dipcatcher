"""Construct trading-session targets by indexing explicitly supplied close calendars.

Each calendar is a caller-asserted complete ordered close sequence within the
inclusive coverage_start/coverage_end window. The entire calendar snapshot has
one available_time: a decision before that clock cannot use it. Future closes
can be known in advance; their scheduled close time is not their availability.
This operation does not infer holidays, validate calendar completeness, or revise
historical calendars. Supply separate calendar IDs for separate known snapshots.

session_offset=1 means the first qualifying close, offset=2 the second, etc.
strictly_after excludes a close equal to decision_time; at_or_after includes it.
Decisions outside asserted coverage and offsets beyond the supplied future close
sequence produce explicit unresolved outcomes. No elapsed-day arithmetic is used.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
SessionOffset = Annotated[int, Field(strict=True, ge=1, le=10_000)]
Status = Literal[
    "assigned",
    "missing_calendar",
    "calendar_not_available",
    "decision_before_calendar",
    "decision_after_calendar",
    "insufficient_future_calendar",
]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError(
            "clock must be an explicit timezone-aware ISO datetime of at most 64 chars"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class SessionClose(InputModel):
    session_id: Name
    close_time: AwareDatetime

    @field_validator("close_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Calendar(InputModel):
    calendar_id: Name
    available_time: AwareDatetime
    coverage_start: AwareDatetime
    coverage_end: AwareDatetime
    closes: list[SessionClose] = Field(max_length=10_000)

    @field_validator("available_time", "coverage_start", "coverage_end", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def ordered_coverage(self) -> Self:
        if self.coverage_end < self.coverage_start:
            raise ValueError("calendar coverage_end must not precede coverage_start")
        if len({row.session_id for row in self.closes}) != len(self.closes):
            raise ValueError("session IDs must be unique within each calendar snapshot")
        previous: datetime | None = None
        for row in self.closes:
            if not self.coverage_start <= row.close_time <= self.coverage_end:
                raise ValueError("session closes must lie inside the asserted coverage window")
            if previous is not None and row.close_time <= previous:
                raise ValueError("session closes must be supplied in strictly increasing order")
            previous = row.close_time
        return self


class Decision(InputModel):
    calendar_id: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    calendars: list[Calendar] = Field(max_length=64)
    decisions: list[Decision] = Field(max_length=10_000)
    session_offsets: list[SessionOffset] = Field(min_length=1, max_length=32)
    inclusion: Literal["strictly_after", "at_or_after"] = "strictly_after"
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=100, strict=True, ge=1, le=1_000)

    @model_validator(mode="after")
    def bounded_unique_inputs(self) -> Self:
        if len({row.calendar_id for row in self.calendars}) != len(self.calendars):
            raise ValueError("calendar IDs must be unique; snapshots need distinct IDs")
        if len(set(self.session_offsets)) != len(self.session_offsets):
            raise ValueError("session_offsets must be distinct")
        if sum(len(row.closes) for row in self.calendars) > 20_000:
            raise ValueError("calendar snapshots may contain at most 20000 closes combined")
        if len(self.decisions) * len(self.session_offsets) > 100_000:
            raise ValueError("at most 100000 decision/horizon assignments are supported")
        return self


class Assignment(OutputModel):
    assignment_index: int
    decision_row_index: int
    horizon_index: int
    session_offset: int
    status: Status
    calendar_row_index: int | None
    target_session_index: int | None
    target_session_id: str | None
    target_time: datetime | None
    target_strictly_after_decision: bool | None


class Output(OutputModel):
    calendar_count: int
    supplied_close_count: int
    decision_count: int
    horizon_count: int
    assignment_count: int
    assigned_count: int
    unresolved_count: int
    status_counts: dict[str, int]
    inclusion: str
    coverage_completeness: Literal["asserted_by_caller_not_inferred"] = (
        "asserted_by_caller_not_inferred"
    )
    assignment_order: Literal["decision_input_order_then_session_offsets_input_order"] = (
        "decision_input_order_then_session_offsets_input_order"
    )
    assignments: list[Assignment]
    next_offset: int | None


def execute(request: Input, context: OperationContext) -> Output:
    lookup = {row.calendar_id: index for index, row in enumerate(request.calendars)}
    clocks = [[close.close_time for close in calendar.closes] for calendar in request.calendars]
    counts: Counter[str] = Counter()
    assignments: list[Assignment] = []
    horizon_count = len(request.session_offsets)
    search = bisect_right if request.inclusion == "strictly_after" else bisect_left
    for decision_index, decision in enumerate(request.decisions):
        calendar_index = lookup.get(decision.calendar_id)
        calendar = request.calendars[calendar_index] if calendar_index is not None else None
        base_status: Status | None = None
        anchor_index = 0
        if calendar is None:
            base_status = "missing_calendar"
        elif calendar.available_time > decision.decision_time:
            base_status = "calendar_not_available"
        elif decision.decision_time < calendar.coverage_start:
            base_status = "decision_before_calendar"
        elif decision.decision_time > calendar.coverage_end:
            base_status = "decision_after_calendar"
        elif calendar_index is not None:
            anchor_index = search(clocks[calendar_index], decision.decision_time)
        for horizon_index, session_offset in enumerate(request.session_offsets):
            index = decision_index * horizon_count + horizon_index
            target_index = anchor_index + session_offset - 1
            target: SessionClose | None = None
            if base_status is not None:
                status = base_status
            elif calendar is None or target_index >= len(calendar.closes):
                status = "insufficient_future_calendar"
            else:
                status = "assigned"
                target = calendar.closes[target_index]
            counts[status] += 1
            if request.offset <= index < request.offset + request.limit:
                assignments.append(
                    Assignment(
                        assignment_index=index,
                        decision_row_index=decision_index,
                        horizon_index=horizon_index,
                        session_offset=session_offset,
                        status=status,
                        calendar_row_index=calendar_index,
                        target_session_index=target_index if target is not None else None,
                        target_session_id=target.session_id if target is not None else None,
                        target_time=target.close_time if target is not None else None,
                        target_strictly_after_decision=(
                            target.close_time > decision.decision_time
                            if target is not None
                            else None
                        ),
                    )
                )
    total = len(request.decisions) * horizon_count
    next_offset = request.offset + len(assignments)
    return Output(
        calendar_count=len(request.calendars),
        supplied_close_count=sum(len(row) for row in clocks),
        decision_count=len(request.decisions),
        horizon_count=horizon_count,
        assignment_count=total,
        assigned_count=counts["assigned"],
        unresolved_count=total - counts["assigned"],
        status_counts=dict(sorted(counts.items())),
        inclusion=request.inclusion,
        assignments=assignments,
        next_offset=next_offset if next_offset < total else None,
    )


OPERATION = Operation(
    id="skills.assign_horizon_targets",
    kind="skill",
    description=(
        "Assign trading-session target closes from observable, explicitly ordered calendar "
        "snapshots with caller-defined coverage, close-inclusion conventions, and unresolved outcomes."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
