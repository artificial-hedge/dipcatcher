"""Integrate a causal held-last-value process across an explicit time window.

An observation can change the held state only after both its event and
availability timestamps. Among observable rows, the latest event wins; an old
event arriving late cannot overwrite a newer event or rewrite past intervals.
Values are held on (activation_time, next_activation_time], a left-continuous
convention. Point endpoints have zero integration duration. A row activated
exactly at the query clock contributes zero weight. Holding has no implicit
staleness cutoff. Missing initial coverage fails closed instead of backfilling.
"""

from datetime import UTC, datetime, timedelta
from math import fsum
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


def _aware_clock(value: object) -> datetime:
    """Accept explicit aware ISO datetimes, excluding implicit Unix timestamps."""
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
    except OverflowError as exc:
        raise ValueError("clock cannot be represented in UTC") from exc


class Observation(InputModel):
    event_time: AwareDatetime
    available_time: AwareDatetime
    value: Value

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def validate_clock(cls, value: object) -> datetime:
        return _aware_clock(value)


class Input(InputModel):
    observations: list[Observation] = Field(min_length=1, max_length=10_000)
    query_time: AwareDatetime
    lookback_seconds: int = Field(strict=True, ge=1, le=31_536_000)

    @field_validator("query_time", mode="before")
    @classmethod
    def validate_query_clock(cls, value: object) -> datetime:
        return _aware_clock(value)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        try:
            self.query_time - timedelta(seconds=self.lookback_seconds)
        except OverflowError as exc:
            raise ValueError("lookback extends before the supported datetime range") from exc
        event_times = [row.event_time for row in self.observations]
        if len(set(event_times)) != len(event_times):
            raise ValueError("event_time must be unique; ambiguous revisions are unsupported")
        return self


class Output(OutputModel):
    window_start: AwareDatetime
    query_time: AwareDatetime
    time_weighted_mean: Finite
    integral_value_seconds: Finite
    covered_seconds: float = Field(gt=0, allow_inf_nan=False)
    requested_seconds: int
    coverage_fraction: float = Field(ge=0, le=1, allow_inf_nan=False)
    positive_duration_segments: int
    initial_observation_index: int
    contributing_observation_indices: list[int] = Field(max_length=10_000)
    rows_not_activated_before_query: int
    max_contributor_available_time: AwareDatetime
    hold_policy: str = "latest_observable_event_without_staleness_cutoff"


def execute(request: Input, context: OperationContext) -> Output:
    """Integrate only state transitions observable when each interval begins."""
    end = request.query_time
    start = end - timedelta(seconds=request.lookback_seconds)
    activated = [
        (max(row.event_time, row.available_time), row.event_time, index, row)
        for index, row in enumerate(request.observations)
        if max(row.event_time, row.available_time) < end
    ]
    activated.sort(key=lambda item: (item[0], item[1]))
    initial = [item for item in activated if item[0] <= start]
    if not initial:
        raise ValueError(
            "window has no observable initial value: an observation must have both event_time "
            "and available_time at or before window_start; partial coverage is unsupported"
        )
    _, current_event, current_index, current_row = max(initial, key=lambda item: item[1])
    initial_index = current_index
    segment_start = start
    contributions: list[float] = []
    contributors: list[int] = []
    contributor_availability: list[datetime] = []
    for activation, event_time, index, row in activated:
        if activation <= start or event_time <= current_event:
            continue
        duration = (activation - segment_start).total_seconds()
        if duration > 0.0:
            contributions.append(current_row.value * duration)
            contributors.append(current_index)
            contributor_availability.append(current_row.available_time)
        current_event = event_time
        current_index = index
        current_row = row
        segment_start = activation
    final_duration = (end - segment_start).total_seconds()
    contributions.append(current_row.value * final_duration)
    contributors.append(current_index)
    contributor_availability.append(current_row.available_time)
    integral = fsum(contributions)
    return Output(
        window_start=start,
        query_time=end,
        time_weighted_mean=integral / request.lookback_seconds,
        integral_value_seconds=integral,
        covered_seconds=float(request.lookback_seconds),
        requested_seconds=request.lookback_seconds,
        coverage_fraction=1.0,
        positive_duration_segments=len(contributions),
        initial_observation_index=initial_index,
        contributing_observation_indices=contributors,
        rows_not_activated_before_query=len(request.observations) - len(activated),
        max_contributor_available_time=max(contributor_availability),
    )


OPERATION = Operation(
    id="features.time_weighted_mean",
    kind="feature",
    description=(
        "Integrate the left-continuous latest-observable-event value over an explicit "
        "lookback and aware query clock. Both event and availability govern activation; "
        "late rows never rewrite history. Missing initial coverage and duplicate events fail."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
