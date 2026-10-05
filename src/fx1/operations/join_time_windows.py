"""Match grouped event timestamps to explicit windows with bounded materialization.

Endpoints are independently open or closed, with [start, end) as the default.
Equal endpoints match only when both are closed. Range membership does not infer
event_time <= decision_time: a caller can deliberately select known future events.
The default availability filter still requires available_time <= decision_time.

Candidate work is counted before availability filtering across ALL left rows.
Per-left limits apply after that filter. Earliest/latest retain a prefix/suffix
of (event_time, original right-row index), returned in that chronological order.
Output pages contain left-row summaries and index pairs only. The output-pair
budget applies to the requested page; counts and overflow checks cover all rows.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
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
        raise ValueError(
            "clock must be an explicit timezone-aware ISO datetime of at most 64 chars"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Window(InputModel):
    group_id: Name
    start_time: AwareDatetime
    end_time: AwareDatetime
    decision_time: AwareDatetime

    @field_validator("start_time", "end_time", "decision_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def ordered_window(self) -> Self:
        if self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        return self


class Observation(InputModel):
    group_id: Name
    event_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    windows: list[Window] = Field(max_length=10_000)
    observations: list[Observation] = Field(max_length=10_000)
    start_closed: bool = Field(default=True, strict=True)
    end_closed: bool = Field(default=False, strict=True)
    require_available_at_decision: bool = Field(default=True, strict=True)
    max_matches_per_window: int = Field(default=100, strict=True, ge=0, le=1_000)
    retention: Literal["earliest", "latest"] = "earliest"
    overflow: Literal["truncate", "error"] = "truncate"
    max_candidate_pairs: int = Field(default=250_000, strict=True, ge=0, le=1_000_000)
    max_output_pairs: int = Field(default=5_000, strict=True, ge=0, le=10_000)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)


class Match(OutputModel):
    window_row_index: int
    observation_row_index: int


class WindowSummary(OutputModel):
    window_row_index: int
    candidate_pairs: int
    excluded_by_availability: int
    eligible_matches: int
    returned_matches: int
    omitted_by_per_window_limit: int
    eligible_max_available_time: datetime | None
    returned_max_available_time: datetime | None


class Output(OutputModel):
    window_count: int
    observation_count: int
    candidate_pairs: int
    excluded_by_availability: int
    eligible_pairs: int
    retained_pairs_all_windows: int
    omitted_by_per_window_limit: int
    windows_without_eligible_matches: int
    windows_with_truncated_matches: int
    start_closed: bool
    end_closed: bool
    require_available_at_decision: bool
    retention: str
    overflow: str
    max_matches_per_window: int
    max_candidate_pairs: int
    max_output_pairs: int
    window_summaries: list[WindowSummary]
    pairs: list[Match]
    returned_pairs: int
    retained_pairs_outside_page: int
    next_offset: int | None


def execute(request: Input, context: OperationContext) -> Output:
    grouped: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    for index, row in enumerate(request.observations):
        grouped[row.group_id].append((row.event_time, index))
    timestamps: dict[str, list[datetime]] = {}
    for group, entries in grouped.items():
        entries.sort()
        timestamps[group] = [timestamp for timestamp, _ in entries]

    # O((N + Q) log N) range lookup precedes the bounded candidate scan. No
    # availability predicate can hide a pathological N-by-Q range expansion.
    ranges: list[tuple[int, int]] = []
    candidate_pairs = 0
    for window in request.windows:
        clock_list = timestamps.get(window.group_id, [])
        lower = (bisect_left if request.start_closed else bisect_right)(
            clock_list, window.start_time
        )
        upper = (bisect_right if request.end_closed else bisect_left)(clock_list, window.end_time)
        upper = max(lower, upper)
        ranges.append((lower, upper))
        candidate_pairs += upper - lower
    if candidate_pairs > request.max_candidate_pairs:
        raise ValueError(
            f"window ranges require {candidate_pairs} candidate pairs, exceeding "
            f"max_candidate_pairs={request.max_candidate_pairs}; reduce windows or range widths"
        )

    summaries: list[WindowSummary] = []
    pairs: list[Match] = []
    eligible_pairs = excluded_total = retained_total = truncated_windows = unmatched_windows = 0
    for window_index, (window, (lower, upper)) in enumerate(
        zip(request.windows, ranges, strict=True)
    ):
        entries = grouped.get(window.group_id, [])
        eligible_indices: list[int] = []
        max_available: datetime | None = None
        for position in range(lower, upper):
            observation_index = entries[position][1]
            availability = request.observations[observation_index].available_time
            if request.require_available_at_decision and availability > window.decision_time:
                continue
            eligible_indices.append(observation_index)
            max_available = (
                availability if max_available is None else max(max_available, availability)
            )
        count = len(eligible_indices)
        excluded = upper - lower - count
        eligible_pairs += count
        excluded_total += excluded
        unmatched_windows += int(count == 0)
        truncated = count > request.max_matches_per_window
        truncated_windows += int(truncated)
        if truncated and request.overflow == "error":
            raise ValueError(
                f"window row {window_index} has {count} eligible matches, exceeding "
                f"max_matches_per_window={request.max_matches_per_window}"
            )
        retained = min(count, request.max_matches_per_window)
        retained_total += retained
        if not request.offset <= window_index < request.offset + request.limit:
            continue
        if len(pairs) + retained > request.max_output_pairs:
            raise ValueError(
                f"requested window page exceeds max_output_pairs={request.max_output_pairs}; "
                "reduce limit or max_matches_per_window, or raise the output-pair budget"
            )
        if retained == 0:
            chosen: list[int] = []
        elif request.retention == "earliest":
            chosen = eligible_indices[:retained]
        else:
            chosen = eligible_indices[-retained:]
        returned_max = max(
            (request.observations[index].available_time for index in chosen),
            default=None,
        )
        summaries.append(
            WindowSummary(
                window_row_index=window_index,
                candidate_pairs=upper - lower,
                excluded_by_availability=excluded,
                eligible_matches=count,
                returned_matches=retained,
                omitted_by_per_window_limit=count - retained,
                eligible_max_available_time=max_available,
                returned_max_available_time=returned_max,
            )
        )
        pairs.extend(
            Match(window_row_index=window_index, observation_row_index=index) for index in chosen
        )

    next_offset = request.offset + len(summaries)
    return Output(
        window_count=len(request.windows),
        observation_count=len(request.observations),
        candidate_pairs=candidate_pairs,
        excluded_by_availability=excluded_total,
        eligible_pairs=eligible_pairs,
        retained_pairs_all_windows=retained_total,
        omitted_by_per_window_limit=eligible_pairs - retained_total,
        windows_without_eligible_matches=unmatched_windows,
        windows_with_truncated_matches=truncated_windows,
        start_closed=request.start_closed,
        end_closed=request.end_closed,
        require_available_at_decision=request.require_available_at_decision,
        retention=request.retention,
        overflow=request.overflow,
        max_matches_per_window=request.max_matches_per_window,
        max_candidate_pairs=request.max_candidate_pairs,
        max_output_pairs=request.max_output_pairs,
        window_summaries=summaries,
        pairs=pairs,
        returned_pairs=len(pairs),
        retained_pairs_outside_page=retained_total - len(pairs),
        next_offset=next_offset if next_offset < len(request.windows) else None,
    )


OPERATION = Operation(
    id="skills.join_time_windows",
    kind="skill",
    description=(
        "Match grouped timestamps within explicit open/closed windows with optional decision-time "
        "availability, source-row lineage, exact counts, and bounded candidate/output pair budgets."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
