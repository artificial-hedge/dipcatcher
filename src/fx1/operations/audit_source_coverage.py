"""Audit caller-declared source/security expectations at a decision clock.

Only completed and available observations are eligible. Select the latest event
per pair, then latest availability, then earliest input row. Optional staleness
uses the caller-selected event or availability age. Expected pairs are supplied
explicitly; this operation never invents an exchange calendar or source truth.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Seconds = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=315_537_897_600)]


def _clock(value: object) -> datetime:
    if isinstance(value, str):
        if len(value) > 64:
            raise ValueError("clock exceeds 64 characters")
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("clock must be a timezone-aware ISO datetime, not an epoch number")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must have a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ValueError("clock cannot be normalized to UTC") from exc


class Pair(InputModel):
    source: Name
    security_id: Name


class Observation(Pair):
    event_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    expected_pairs: list[Pair] = Field(default_factory=list, max_length=5000)
    observations: list[Observation] = Field(default_factory=list, max_length=10_000)
    decision_time: AwareDatetime
    max_age_seconds: Seconds | None = None
    staleness_clock: Literal["event_time", "available_time"] = "event_time"
    offset: int = Field(default=0, strict=True, ge=0, le=5000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)
    max_unexpected_examples: int = Field(default=25, strict=True, ge=0, le=100)

    @field_validator("decision_time", mode="before")
    @classmethod
    def validate_decision_clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def unique_expectations(self) -> Self:
        keys = {(pair.source, pair.security_id) for pair in self.expected_pairs}
        if len(keys) != len(self.expected_pairs):
            raise ValueError("expected source/security pairs must be unique")
        return self


class PairCoverage(OutputModel):
    source: str
    security_id: str
    status: Literal["missing", "stale", "covered"]
    input_records: int
    eligible_records: int
    selected_row_index: int | None
    selected_event_time: AwareDatetime | None
    selected_available_time: AwareDatetime | None
    event_age_seconds: float | None
    availability_age_seconds: float | None


class UnexpectedPair(OutputModel):
    source: str
    security_id: str
    eligible_records: int
    selected_row_index: int


class Output(OutputModel):
    decision_time: AwareDatetime
    selection_policy: Literal["latest_event_then_availability_then_earliest_input_row"] = (
        "latest_event_then_availability_then_earliest_input_row"
    )
    staleness_clock: str
    max_age_seconds: float | None
    assessment: Literal["covered", "incomplete", "no_expected_pairs"]
    complete_expected_coverage: bool | None
    expected_pair_count: int
    covered_expected_pairs: int
    stale_expected_pairs: int
    missing_expected_pairs: int
    input_record_count: int
    eligible_record_count: int
    unavailable_records: int
    future_event_records_excluded: int
    eligible_expected_records: int
    eligible_unexpected_records: int
    unexpected_input_pair_count: int
    unexpected_eligible_pair_count: int
    unexpected_eligible_pairs: list[UnexpectedPair]
    omitted_unexpected_examples: int
    offset: int
    next_offset: int | None
    expected_pairs: list[PairCoverage]


def execute(request: Input, context: OperationContext) -> Output:
    expected = {(pair.source, pair.security_id) for pair in request.expected_pairs}
    input_counts: Counter[tuple[str, str]] = Counter()
    eligible_counts: Counter[tuple[str, str]] = Counter()
    selected: dict[tuple[str, str], tuple[datetime, datetime, int]] = {}
    unavailable = future_events = eligible = 0
    for index, row in enumerate(request.observations):
        key = row.source, row.security_id
        input_counts[key] += 1
        if row.available_time > request.decision_time:
            unavailable += 1
            continue
        if row.event_time > request.decision_time:
            future_events += 1
            continue
        eligible += 1
        eligible_counts[key] += 1
        priority = row.event_time, row.available_time, -index
        if key not in selected or priority > selected[key]:
            selected[key] = priority

    covered = missing = stale = 0
    page: list[PairCoverage] = []
    ordered_expected = sorted(expected)
    for pair_index, key in enumerate(ordered_expected):
        chosen = selected.get(key)
        event_age: float | None = None
        availability_age: float | None = None
        selected_index: int | None = None
        event_time: datetime | None = None
        available_time: datetime | None = None
        status: Literal["missing", "stale", "covered"]
        if chosen is None:
            missing += 1
            status = "missing"
        else:
            event_time, available_time, negative_index = chosen
            selected_index = -negative_index
            event_age = (request.decision_time - event_time).total_seconds()
            availability_age = (request.decision_time - available_time).total_seconds()
            age = event_age if request.staleness_clock == "event_time" else availability_age
            if request.max_age_seconds is not None and age > request.max_age_seconds:
                stale += 1
                status = "stale"
            else:
                covered += 1
                status = "covered"
        if request.offset <= pair_index < request.offset + request.limit:
            page.append(
                PairCoverage(
                    source=key[0],
                    security_id=key[1],
                    status=status,
                    input_records=input_counts[key],
                    eligible_records=eligible_counts[key],
                    selected_row_index=selected_index,
                    selected_event_time=event_time,
                    selected_available_time=available_time,
                    event_age_seconds=event_age,
                    availability_age_seconds=availability_age,
                )
            )
    unexpected_keys = sorted(eligible_counts.keys() - expected)
    unexpected = [
        UnexpectedPair(
            source=key[0],
            security_id=key[1],
            eligible_records=eligible_counts[key],
            selected_row_index=-selected[key][2],
        )
        for key in unexpected_keys[: request.max_unexpected_examples]
    ]
    stop = request.offset + len(page)
    return Output(
        decision_time=request.decision_time,
        staleness_clock=request.staleness_clock,
        max_age_seconds=request.max_age_seconds,
        assessment=("incomplete" if missing or stale else "covered") if expected else "no_expected_pairs",
        complete_expected_coverage=missing == stale == 0 if expected else None,
        expected_pair_count=len(expected),
        covered_expected_pairs=covered,
        stale_expected_pairs=stale,
        missing_expected_pairs=missing,
        input_record_count=len(request.observations),
        eligible_record_count=eligible,
        unavailable_records=unavailable,
        future_event_records_excluded=future_events,
        eligible_expected_records=sum(eligible_counts[key] for key in expected),
        eligible_unexpected_records=sum(eligible_counts[key] for key in unexpected_keys),
        unexpected_input_pair_count=len(input_counts.keys() - expected),
        unexpected_eligible_pair_count=len(unexpected_keys),
        unexpected_eligible_pairs=unexpected,
        omitted_unexpected_examples=len(unexpected_keys) - len(unexpected),
        offset=request.offset,
        next_offset=stop if stop < len(ordered_expected) else None,
        expected_pairs=page,
    )


OPERATION = Operation(
    id="skills.audit_source_coverage",
    kind="skill",
    description=(
        "Audit caller-supplied source/security expectations using only completed observations "
        "available by a decision clock. Counts missing/stale coverage and unexpected pairs, "
        "preserves latest-row lineage, and pages details without assuming a market calendar."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
