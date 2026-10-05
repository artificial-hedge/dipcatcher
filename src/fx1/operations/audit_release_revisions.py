"""Audit initial publication timing and revised observations against a release schedule.

A release is identified by (series_id, release_id); expected_sources lists the
publications required for it. Its event_time is the economic reference clock,
which may precede or follow publication. available_time is the observation's
public availability, never an ingestion timestamp. Only schedules and observations
known by asof enter the comparison; schedule timing cannot be inferred from data.

Early/late tolerances apply to the FIRST supplied visible identity-matching
observation for each expected source. Omitted earlier publications cannot be
inferred. Later revisions are not automatically late initial
releases. Reusing a revision ID within one release/source for different values,
event or availability is a conflict; exact repeats have a separate explicit
policy. Different sources may legitimately use the same revision ID.

Coverage means at least one identity-matching visible row for every expected
source whose scheduled time plus late tolerance has elapsed. This coverage flag
does not certify value consistency or timely publication; passed also requires
those audits. Missing future observations are not fabricated or backfilled.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    Annotated[str, Field(strict=True, max_length=512)] | StrictBool | StrictInt | StrictFloat | None
)
Timing = Literal["not_due", "awaiting_within_tolerance", "missing", "early", "on_time", "late"]


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


def _microseconds(later: datetime, earlier: datetime) -> int:
    difference = later - earlier
    return (difference.days * 86_400 + difference.seconds) * 1_000_000 + difference.microseconds


class ScheduledRelease(InputModel):
    series_id: Name
    release_id: Name
    event_time: AwareDatetime
    scheduled_time: AwareDatetime
    available_time: AwareDatetime
    expected_sources: list[Name] = Field(min_length=1, max_length=16)

    @field_validator("event_time", "scheduled_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def distinct_sources(self) -> Self:
        if len(set(self.expected_sources)) != len(self.expected_sources):
            raise ValueError("expected_sources must be distinct")
        return self


class Observation(InputModel):
    series_id: Name
    release_id: Name
    event_time: AwareDatetime
    available_time: AwareDatetime
    source: Name
    revision_id: Name
    values: dict[Name, Cell] = Field(min_length=1, max_length=16)

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    schedule: list[ScheduledRelease] = Field(max_length=10_000)
    observations: list[Observation] = Field(max_length=10_000)
    asof: AwareDatetime
    early_tolerance_seconds: int = Field(default=0, strict=True, ge=0, le=31_536_000)
    late_tolerance_seconds: int = Field(default=0, strict=True, ge=0, le=31_536_000)
    reject_exact_duplicates: bool = Field(default=False, strict=True)
    offset: int = Field(default=0, strict=True, ge=0, le=20_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indices: int = Field(default=5, strict=True, ge=1, le=10)

    @field_validator("asof", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def bounded_schedule(self) -> Self:
        if len({(row.series_id, row.release_id) for row in self.schedule}) != len(self.schedule):
            raise ValueError("schedule must contain one row per series/release identity")
        if sum(len(row.expected_sources) for row in self.schedule) > 20_000:
            raise ValueError("schedule may declare at most 20000 release/source pairs")
        return self


class Finding(OutputModel):
    code: Literal[
        "unknown_release",
        "schedule_not_available",
        "unexpected_source",
        "event_time_mismatch",
        "revision_reuse_conflict",
        "exact_duplicate_rows",
        "ambiguous_initial_publication",
        "early",
        "late",
        "missing",
    ]
    is_violation: bool
    schedule_row_index: int | None = None
    source: str
    observation_row_indices: list[int]
    omitted_row_indices: int = 0


class SourceRelease(OutputModel):
    pair_index: int
    schedule_row_index: int
    expected_source_index: int
    source: str
    due_by_asof: bool
    timing: Timing
    visible_matching_rows: int
    revision_count: int
    first_available_time: datetime | None
    first_lag_microseconds: int | None
    first_observation_row_index: int | None
    initial_publication_ambiguous: bool


class Output(OutputModel):
    scheduled_release_count: int
    visible_scheduled_release_count: int
    unobservable_scheduled_release_count: int
    observation_count: int
    future_observation_rows: int
    observations_with_unobservable_schedule: int
    unknown_release_rows: int
    unexpected_source_rows: int
    event_time_mismatch_rows: int
    visible_expected_source_pairs: int
    due_expected_source_pairs: int
    missing_due_source_pairs: int
    revision_reuse_conflict_groups: int
    exact_duplicate_rows: int
    ambiguous_initial_publications: int
    timing_counts: dict[str, int]
    coverage_complete: bool
    passed: bool
    asof: datetime
    early_tolerance_seconds: int
    late_tolerance_seconds: int
    reject_exact_duplicates: bool
    timing_scope: Literal["earliest_supplied_visible_matching_publication"] = (
        "earliest_supplied_visible_matching_publication"
    )
    releases: list[SourceRelease]
    next_offset: int | None
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    schedule_index = {
        (row.series_id, row.release_id): index for index, row in enumerate(request.schedule)
    }
    expected_sources = [set(row.expected_sources) for row in request.schedule]
    pairs: dict[tuple[int, str], list[int]] = defaultdict(list)
    revisions: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    payloads: dict[int, bytes] = {}
    diagnostics: list[Finding] = []
    diagnostic_count = future_rows = unknown_rows = unexpected_rows = mismatched_rows = (
        unseen_schedule
    ) = 0

    def report(
        code: str,
        source: str,
        indices: list[int],
        schedule_row: int | None,
        *,
        violation: bool = True,
    ) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            # All code values are local literals; use model validation to keep
            # the public diagnostic vocabulary explicit in its output schema.
            diagnostics.append(
                Finding.model_validate(
                    {
                        "code": code,
                        "is_violation": violation,
                        "source": source,
                        "schedule_row_index": schedule_row,
                        "observation_row_indices": indices[: request.max_row_indices],
                        "omitted_row_indices": max(0, len(indices) - request.max_row_indices),
                    }
                )
            )

    for index, row in enumerate(request.observations):
        if row.available_time > request.asof:
            future_rows += 1
            continue
        scheduled_index = schedule_index.get((row.series_id, row.release_id))
        if scheduled_index is None:
            unknown_rows += 1
            report("unknown_release", row.source, [index], None)
            continue
        scheduled = request.schedule[scheduled_index]
        if scheduled.available_time > request.asof:
            unseen_schedule += 1
            report("schedule_not_available", row.source, [index], scheduled_index, violation=False)
            continue
        valid = True
        if row.source not in expected_sources[scheduled_index]:
            unexpected_rows += 1
            report("unexpected_source", row.source, [index], scheduled_index)
            valid = False
        if row.event_time != scheduled.event_time:
            mismatched_rows += 1
            report("event_time_mismatch", row.source, [index], scheduled_index)
            valid = False
        # Reuse checks also inspect wrong-event rows for an otherwise expected
        # source, so a revision cannot conceal a changed release reference clock.
        if row.source in expected_sources[scheduled_index]:
            revisions[(scheduled_index, row.source, row.revision_id)].append(index)
            payloads[index] = canonical_json(row.values)
        if valid:
            pairs[(scheduled_index, row.source)].append(index)

    conflicts = exact_duplicates = 0
    for (scheduled_index, source, _), indices in sorted(revisions.items()):
        variants: dict[tuple[datetime, datetime, bytes], int] = {}
        for index in indices:
            row = request.observations[index]
            variants.setdefault((row.event_time, row.available_time, payloads[index]), index)
        repeated = len(indices) - len(variants)
        exact_duplicates += repeated
        if repeated:
            report(
                "exact_duplicate_rows",
                source,
                indices,
                scheduled_index,
                violation=request.reject_exact_duplicates,
            )
        if len(variants) > 1:
            conflicts += 1
            report("revision_reuse_conflict", source, list(variants.values()), scheduled_index)

    timing_counts: Counter[str] = Counter()
    releases: list[SourceRelease] = []
    pair_index = due_count = missing_count = ambiguous_count = visible_schedules = 0
    early_us = request.early_tolerance_seconds * 1_000_000
    late_us = request.late_tolerance_seconds * 1_000_000
    for scheduled_index, scheduled in enumerate(request.schedule):
        if scheduled.available_time > request.asof:
            continue
        visible_schedules += 1
        elapsed = _microseconds(request.asof, scheduled.scheduled_time)
        due = elapsed >= late_us
        for source_index, source in enumerate(scheduled.expected_sources):
            indices = pairs.get((scheduled_index, source), [])
            first_time = min(
                (request.observations[index].available_time for index in indices), default=None
            )
            initial = [
                index
                for index in indices
                if request.observations[index].available_time == first_time
            ]
            initial_variants = {payloads[index] for index in initial}
            ambiguous = len(initial_variants) > 1
            ambiguous_count += int(ambiguous)
            if ambiguous:
                report("ambiguous_initial_publication", source, initial, scheduled_index)
            lag = (
                _microseconds(first_time, scheduled.scheduled_time)
                if first_time is not None
                else None
            )
            timing: Timing
            if lag is not None:
                timing = "early" if lag < -early_us else "late" if lag > late_us else "on_time"
            elif due:
                timing = "missing"
                missing_count += 1
            else:
                timing = "not_due" if elapsed < 0 else "awaiting_within_tolerance"
            due_count += int(due)
            timing_counts[timing] += 1
            if timing in ("early", "late", "missing"):
                report(timing, source, initial, scheduled_index)
            if request.offset <= pair_index < request.offset + request.limit:
                releases.append(
                    SourceRelease(
                        pair_index=pair_index,
                        schedule_row_index=scheduled_index,
                        expected_source_index=source_index,
                        source=source,
                        due_by_asof=due,
                        timing=timing,
                        visible_matching_rows=len(indices),
                        revision_count=len(
                            {request.observations[index].revision_id for index in indices}
                        ),
                        first_available_time=first_time,
                        first_lag_microseconds=lag,
                        first_observation_row_index=(
                            initial[0] if initial and not ambiguous else None
                        ),
                        initial_publication_ambiguous=ambiguous,
                    )
                )
            pair_index += 1
    passed = not (
        unknown_rows
        or unexpected_rows
        or mismatched_rows
        or conflicts
        or ambiguous_count
        or timing_counts["early"]
        or timing_counts["late"]
        or missing_count
        or (request.reject_exact_duplicates and exact_duplicates)
    )
    next_offset = request.offset + len(releases)
    return Output(
        scheduled_release_count=len(request.schedule),
        visible_scheduled_release_count=visible_schedules,
        unobservable_scheduled_release_count=len(request.schedule) - visible_schedules,
        observation_count=len(request.observations),
        future_observation_rows=future_rows,
        observations_with_unobservable_schedule=unseen_schedule,
        unknown_release_rows=unknown_rows,
        unexpected_source_rows=unexpected_rows,
        event_time_mismatch_rows=mismatched_rows,
        visible_expected_source_pairs=pair_index,
        due_expected_source_pairs=due_count,
        missing_due_source_pairs=missing_count,
        revision_reuse_conflict_groups=conflicts,
        exact_duplicate_rows=exact_duplicates,
        ambiguous_initial_publications=ambiguous_count,
        timing_counts=dict(sorted(timing_counts.items())),
        coverage_complete=missing_count == 0,
        passed=passed,
        asof=request.asof,
        early_tolerance_seconds=request.early_tolerance_seconds,
        late_tolerance_seconds=request.late_tolerance_seconds,
        reject_exact_duplicates=request.reject_exact_duplicates,
        releases=releases,
        next_offset=next_offset if next_offset < pair_index else None,
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_release_revisions",
    kind="skill",
    description=(
        "Audit expected source publications against explicit release schedules at an asof, "
        "distinguishing initial early/late timing, due coverage, revision conflicts and exact repeats."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
