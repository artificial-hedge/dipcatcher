"""Audit half-open interval geometry using exact overlap counts and gap witnesses.

None at valid_from means negative infinity; None at valid_to means positive
infinity. Empty/inverted intervals are invalid and excluded from geometry.
Touching endpoints do not overlap. Only internal gaps in the observed union
are reported; no exterior horizon or market calendar is inferred.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import UTC, datetime
from math import fsum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
_MIN_CLOCK = datetime.min.replace(tzinfo=UTC)


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


class Interval(InputModel):
    group_id: Name
    valid_from: AwareDatetime | None
    valid_to: AwareDatetime | None

    @field_validator("valid_from", "valid_to", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime | None:
        return None if value is None else _clock(value)


class Input(InputModel):
    intervals: list[Interval] = Field(min_length=1, max_length=10_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)


class Finding(OutputModel):
    group_id: str
    code: Literal["inverted_interval", "empty_interval", "overlap", "gap"]
    row_index: int
    other_row_index: int | None = None
    span_start: datetime | None
    span_end: datetime | None
    prior_overlap_partners: int = 0
    gap_seconds: float | None = None


class Output(OutputModel):
    input_intervals: int
    group_count: int
    valid_intervals: int
    invalid_intervals: int
    open_start_intervals: int
    open_end_intervals: int
    overlap_pair_count: int
    intervals_with_prior_overlap: int
    internal_gap_count: int
    total_internal_gap_seconds: float
    passed: bool
    interval_policy: str = "half_open_with_none_as_unbounded_endpoint"
    overlap_diagnostics: str = "one_witness_per_interval_with_prior_overlap"
    finding_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    groups: dict[str, list[int]] = defaultdict(list)
    diagnostics: list[Finding] = []
    findings = invalid = open_starts = open_ends = 0

    def report(finding: Finding) -> None:
        nonlocal findings
        findings += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    for index, row in enumerate(request.intervals):
        open_starts += int(row.valid_from is None)
        open_ends += int(row.valid_to is None)
        groups.setdefault(row.group_id, [])
        if (
            row.valid_from is not None
            and row.valid_to is not None
            and row.valid_to <= row.valid_from
        ):
            invalid += 1
            report(
                Finding(
                    group_id=row.group_id,
                    code="empty_interval"
                    if row.valid_to == row.valid_from
                    else "inverted_interval",
                    row_index=index,
                    span_start=row.valid_from,
                    span_end=row.valid_to,
                )
            )
        else:
            groups[row.group_id].append(index)

    overlap_pairs = overlapping_intervals = gap_count = 0
    gap_durations: list[float] = []
    for group_id, indices in sorted(groups.items()):
        indices.sort(
            key=lambda index: (
                request.intervals[index].valid_from is not None,
                request.intervals[index].valid_from or _MIN_CLOCK,
                index,
            )
        )
        active: list[tuple[bool, datetime, int]] = []
        union_end: datetime | None = None
        union_end_index: int | None = None
        have_union = False
        for index in indices:
            row = request.intervals[index]
            start, end = row.valid_from, row.valid_to
            if start is not None:
                while active and not active[0][0] and active[0][1] <= start:
                    heapq.heappop(active)
            if active:
                overlapping_intervals += 1
                overlap_pairs += len(active)
                witness_index = active[0][2]
                witness_end = request.intervals[witness_index].valid_to
                overlap_end = (
                    end
                    if witness_end is None
                    else witness_end
                    if end is None
                    else min(end, witness_end)
                )
                report(
                    Finding(
                        group_id=group_id,
                        code="overlap",
                        row_index=index,
                        other_row_index=witness_index,
                        span_start=start,
                        span_end=overlap_end,
                        prior_overlap_partners=len(active),
                    )
                )
            if have_union and union_end is not None and start is not None and start > union_end:
                duration = (start - union_end).total_seconds()
                gap_count += 1
                gap_durations.append(duration)
                report(
                    Finding(
                        group_id=group_id,
                        code="gap",
                        row_index=index,
                        other_row_index=union_end_index,
                        span_start=union_end,
                        span_end=start,
                        gap_seconds=duration,
                    )
                )
            if not have_union or (union_end is not None and (end is None or end > union_end)):
                union_end = end
                union_end_index = index
            have_union = True
            heapq.heappush(active, (end is None, end or _MIN_CLOCK, index))

    return Output(
        input_intervals=len(request.intervals),
        group_count=len(groups),
        valid_intervals=len(request.intervals) - invalid,
        invalid_intervals=invalid,
        open_start_intervals=open_starts,
        open_end_intervals=open_ends,
        overlap_pair_count=overlap_pairs,
        intervals_with_prior_overlap=overlapping_intervals,
        internal_gap_count=gap_count,
        total_internal_gap_seconds=fsum(gap_durations),
        passed=findings == 0,
        finding_count=findings,
        diagnostics=diagnostics,
        omitted_diagnostics=findings - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_effective_intervals",
    kind="skill",
    description=(
        "Audit grouped half-open effective intervals for empty/inverted rows, exact "
        "overlap-pair counts, and internal union gaps using a heap sweep. Support unbounded "
        "endpoints and bounded witnesses without inferring calendars or exterior coverage."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
