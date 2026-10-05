"""Audit completed bar closes on caller-supplied session segments, including breaks.

Each segment has close support (open_time, close_time]. Bars close at open_time
plus positive integer multiples of interval_seconds. A final short interval is
explicitly rejected, included at close_time, or dropped. Breaks and boundaries
come solely from the supplied segments; no exchange calendar is inferred.

Only schedules and bars available by asof contribute to coverage. Bar availability
before its event close is invalid under this completed-bar contract. Expected
points stop at asof; unavailable bars cannot fill gaps. Missing points are counted
arithmetically and represented by compressed spans, never enumerated. An outside
segment finding describes supplied visible coverage, not whether an exchange was
closed. Duplicate event closes count once toward coverage.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from itertools import pairwise
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
        raise ValueError(
            "clock must be an explicit timezone-aware ISO datetime of at most 64 chars"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Segment(InputModel):
    open_time: AwareDatetime
    close_time: AwareDatetime

    @field_validator("open_time", "close_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.close_time <= self.open_time:
            raise ValueError("segment close_time must be after open_time")
        return self


class Session(InputModel):
    group_id: Name
    session_id: Name
    available_time: AwareDatetime
    segments: list[Segment] = Field(min_length=1, max_length=32)

    @field_validator("available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Bar(InputModel):
    group_id: Name
    event_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    sessions: list[Session] = Field(max_length=5_000)
    bars: list[Bar] = Field(max_length=10_000)
    asof: AwareDatetime
    interval_seconds: int = Field(strict=True, ge=1, le=31_536_000)
    partial_bar: Literal["reject", "include", "drop"] = "reject"
    offset: int = Field(default=0, strict=True, ge=0, le=5_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)

    @field_validator("asof", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def bounded_disjoint_segments(self) -> Self:
        if len({(row.group_id, row.session_id) for row in self.sessions}) != len(self.sessions):
            raise ValueError("session IDs must be unique within each group")
        if sum(len(row.segments) for row in self.sessions) > 10_000:
            raise ValueError("sessions may contain at most 10000 segments combined")
        groups: dict[str, list[Segment]] = defaultdict(list)
        interval = timedelta(seconds=self.interval_seconds)
        for session in self.sessions:
            previous: datetime | None = None
            for segment in session.segments:
                if previous is not None and segment.open_time < previous:
                    raise ValueError("session segments must be ordered and nonoverlapping")
                previous = segment.close_time
                if (
                    self.partial_bar == "reject"
                    and (segment.close_time - segment.open_time) % interval
                ):
                    raise ValueError(
                        "segment duration is not an exact multiple of interval_seconds"
                    )
                groups[session.group_id].append(segment)
        for segments in groups.values():
            ordered = sorted(segments, key=lambda row: row.open_time)
            if any(right.open_time < left.close_time for left, right in pairwise(ordered)):
                raise ValueError("segments from different sessions of one group must not overlap")
        return self


class Finding(OutputModel):
    code: Literal[
        "availability_before_close",
        "duplicate_close",
        "outside_supplied_visible_segments",
        "off_grid",
        "missing_span",
    ]
    bar_row_index: int | None = None
    session_row_index: int | None = None
    segment_index: int | None = None
    event_time: datetime | None = None
    first_missing_close: datetime | None = None
    last_missing_close: datetime | None = None
    missing_points: int = 0


class SessionSummary(OutputModel):
    session_row_index: int
    schedule_observable: bool
    due_expected_points: int
    observed_grid_points: int
    missing_grid_points: int


class Output(OutputModel):
    session_count: int
    visible_session_count: int
    segment_count: int
    bar_count: int
    unavailable_bar_rows: int
    premature_availability_rows: int
    duplicate_bar_rows: int
    outside_supplied_visible_segments_rows: int
    off_grid_rows: int
    due_expected_points: int
    observed_grid_points: int
    missing_grid_points: int
    asof: datetime
    interval_seconds: int
    partial_bar: str
    passed: bool
    sessions: list[SessionSummary]
    next_offset: int | None
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    interval = timedelta(seconds=request.interval_seconds)
    # Each flat segment retains both original indices. Sorted closes support
    # unambiguous binary-search lookup because all supplied segments are disjoint.
    flat: list[tuple[int, int, Segment, int, int]] = []
    grouped: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    expected = [0] * len(request.sessions)
    observed = [0] * len(request.sessions)
    for session_index, session in enumerate(request.sessions):
        if session.available_time > request.asof:
            continue
        for segment_index, segment in enumerate(session.segments):
            full_slots, residual = divmod(segment.close_time - segment.open_time, interval)
            stop = min(request.asof, segment.close_time)
            due_slots = max(0, (stop - segment.open_time) // interval)
            if request.partial_bar == "include" and residual and segment.close_time <= request.asof:
                due_slots += 1
            flat_index = len(flat)
            flat.append((session_index, segment_index, segment, full_slots, due_slots))
            grouped[session.group_id].append((segment.close_time, flat_index))
            expected[session_index] += due_slots
    close_lists: dict[str, list[datetime]] = {}
    for group, entries in grouped.items():
        entries.sort()
        close_lists[group] = [close for close, _ in entries]
    slots: dict[int, set[int]] = defaultdict(set)
    seen: set[tuple[str, datetime]] = set()
    diagnostics: list[Finding] = []
    diagnostic_count = unavailable = premature = duplicates = outside = off_grid = 0

    def report(finding: Finding) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    for bar_index, bar in enumerate(request.bars):
        unavailable += int(bar.available_time > request.asof)
        if bar.available_time < bar.event_time:
            premature += 1
            report(
                Finding(
                    code="availability_before_close",
                    bar_row_index=bar_index,
                    event_time=bar.event_time,
                )
            )
            continue
        if bar.available_time > request.asof:
            continue
        key = bar.group_id, bar.event_time
        if key in seen:
            duplicates += 1
            report(
                Finding(code="duplicate_close", bar_row_index=bar_index, event_time=bar.event_time)
            )
        seen.add(key)
        entries = grouped.get(bar.group_id, [])
        position = bisect_left(close_lists.get(bar.group_id, []), bar.event_time)
        if position == len(entries) or flat[entries[position][1]][2].open_time >= bar.event_time:
            outside += 1
            report(
                Finding(
                    code="outside_supplied_visible_segments",
                    bar_row_index=bar_index,
                    event_time=bar.event_time,
                )
            )
            continue
        flat_index = entries[position][1]
        session_index, segment_index, segment, full_slots, _ = flat[flat_index]
        slot, residual = divmod(bar.event_time - segment.open_time, interval)
        if residual:
            if request.partial_bar == "include" and bar.event_time == segment.close_time:
                slot = full_slots + 1
            else:
                off_grid += 1
                report(
                    Finding(
                        code="off_grid",
                        bar_row_index=bar_index,
                        session_row_index=session_index,
                        segment_index=segment_index,
                        event_time=bar.event_time,
                    )
                )
                continue
        slots[flat_index].add(slot)

    for flat_index, (session_index, segment_index, segment, full_slots, due_slots) in enumerate(
        flat
    ):
        aligned = sorted(slots[flat_index])
        observed[session_index] += len(aligned)
        previous_slot = 0
        for slot in [*aligned, due_slots + 1]:
            missing = slot - previous_slot - 1
            if missing:
                first_slot, last_slot = previous_slot + 1, slot - 1
                # The final partial close is the only possible nonuniform step.
                first = (
                    segment.close_time
                    if first_slot > full_slots
                    else segment.open_time + first_slot * interval
                )
                last = (
                    segment.close_time
                    if last_slot > full_slots
                    else segment.open_time + last_slot * interval
                )
                report(
                    Finding(
                        code="missing_span",
                        session_row_index=session_index,
                        segment_index=segment_index,
                        first_missing_close=first,
                        last_missing_close=last,
                        missing_points=missing,
                    )
                )
            previous_slot = slot
    summaries = [
        SessionSummary(
            session_row_index=index,
            schedule_observable=session.available_time <= request.asof,
            due_expected_points=expected[index],
            observed_grid_points=observed[index],
            missing_grid_points=expected[index] - observed[index],
        )
        for index, session in enumerate(request.sessions)
        if request.offset <= index < request.offset + request.limit
    ]
    next_offset = request.offset + len(summaries)
    return Output(
        session_count=len(request.sessions),
        visible_session_count=sum(row.available_time <= request.asof for row in request.sessions),
        segment_count=sum(len(row.segments) for row in request.sessions),
        bar_count=len(request.bars),
        unavailable_bar_rows=unavailable,
        premature_availability_rows=premature,
        duplicate_bar_rows=duplicates,
        outside_supplied_visible_segments_rows=outside,
        off_grid_rows=off_grid,
        due_expected_points=sum(expected),
        observed_grid_points=sum(observed),
        missing_grid_points=sum(expected) - sum(observed),
        asof=request.asof,
        interval_seconds=request.interval_seconds,
        partial_bar=request.partial_bar,
        passed=diagnostic_count == 0,
        sessions=summaries,
        next_offset=next_offset if next_offset < len(request.sessions) else None,
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_session_coverage",
    kind="skill",
    description=(
        "Audit observable bar closes against explicit session segments and breaks at an asof, "
        "with partial-bar policies, compressed missing spans, duplicate and outside-grid diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
