"""Join decisions to the latest completed and observable event in each security.

An availability/event activation sweep handles all decisions together. Each row
becomes usable at max(event_time, available_time); the best active row per
security is ordered by event, availability, revision label, and original index.
Runtime is O((N+Q) log(N+Q)), with O(N+Q) storage and no query-by-row scan.
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


class Observation(InputModel):
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


class Query(InputModel):
    security_id: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def validate_decision_clock(cls, value: object) -> datetime:
        return _aware_clock(value)


class Input(InputModel):
    observations: list[Observation] = Field(max_length=10_000)
    queries: list[Query] = Field(min_length=1, max_length=10_000)
    max_event_age_seconds: Annotated[float, Field(strict=True, ge=0, le=315_537_897_599)] | None = (
        None
    )
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=1000, strict=True, ge=1, le=1000)

    @field_validator("observations")
    @classmethod
    def consistent_same_clock_records(cls, rows: list[Observation]) -> list[Observation]:
        fingerprints: dict[tuple[str, datetime, datetime], tuple[bytes, int]] = {}
        for index, row in enumerate(rows):
            key = (
                row.security_id,
                row.event_time.astimezone(UTC),
                row.available_time.astimezone(UTC),
            )
            fingerprint = canonical_json({"source": row.source, "values": row.values})
            prior = fingerprints.get(key)
            if prior is not None and prior[0] != fingerprint:
                raise ValueError(
                    f"conflicting source or values at the same event/availability clock "
                    f"in rows {prior[1]} and {index}"
                )
            if prior is None:
                fingerprints[key] = fingerprint, index
        return rows


class JoinedDecision(OutputModel):
    query_row_index: int
    security_id: str
    decision_time: datetime
    matched: bool
    unmatched_reason: Literal["no_eligible_observation", "outside_lookback"] | None
    observation_row_index: int | None
    event_age_seconds: float | None
    max_source_available_time: datetime | None


class SelectedObservation(OutputModel):
    observation_row_index: int
    observation: Observation


class Output(OutputModel):
    observation_count: int
    query_count: int
    matched_count: int
    no_eligible_observation_count: int
    outside_lookback_count: int
    max_event_age_seconds: float | None
    max_source_available_time_used: datetime | None
    selection_policy: Literal[
        "latest_event_latest_availability_greatest_revision_id_earliest_input_row"
    ] = "latest_event_latest_availability_greatest_revision_id_earliest_input_row"
    ordering: Literal["original_query_order"] = "original_query_order"
    offset: int
    next_offset: int | None
    joined: list[JoinedDecision]
    selected_observations: list[SelectedObservation]


def execute(request: Input, context: OperationContext) -> Output:
    activations = sorted(
        (
            max(row.event_time.astimezone(UTC), row.available_time.astimezone(UTC)),
            index,
        )
        for index, row in enumerate(request.observations)
    )
    query_order = sorted(
        (query.decision_time.astimezone(UTC), index) for index, query in enumerate(request.queries)
    )
    best: dict[str, tuple[datetime, datetime, str, int]] = {}
    cursor = matched = no_eligible = outside_lookback = 0
    max_availability: datetime | None = None
    page: dict[int, JoinedDecision] = {}
    page_source_indices: set[int] = set()
    page_stop = min(len(request.queries), request.offset + request.limit)

    for decision, query_index in query_order:
        while cursor < len(activations) and activations[cursor][0] <= decision:
            source_index = activations[cursor][1]
            row = request.observations[source_index]
            priority = (
                row.event_time.astimezone(UTC),
                row.available_time.astimezone(UTC),
                row.revision_id,
                -source_index,
            )
            if row.security_id not in best or priority > best[row.security_id]:
                best[row.security_id] = priority
            cursor += 1

        query = request.queries[query_index]
        candidate = best.get(query.security_id)
        reason: Literal["no_eligible_observation", "outside_lookback"] | None = None
        source_index_or_none: int | None = None
        event_age: float | None = None
        availability: datetime | None = None
        if candidate is None:
            no_eligible += 1
            reason = "no_eligible_observation"
        else:
            age = decision - candidate[0]
            event_age = age.total_seconds()
            if (
                request.max_event_age_seconds is not None
                and event_age > request.max_event_age_seconds
            ):
                outside_lookback += 1
                reason = "outside_lookback"
            else:
                matched += 1
                source_index_or_none = -candidate[3]
                availability = candidate[1]
                if max_availability is None or availability > max_availability:
                    max_availability = availability

        if request.offset <= query_index < page_stop:
            if source_index_or_none is not None:
                page_source_indices.add(source_index_or_none)
            page[query_index] = JoinedDecision(
                query_row_index=query_index,
                security_id=query.security_id,
                decision_time=decision,
                matched=reason is None,
                unmatched_reason=reason,
                observation_row_index=source_index_or_none,
                event_age_seconds=event_age,
                max_source_available_time=availability,
            )

    return Output(
        observation_count=len(request.observations),
        query_count=len(request.queries),
        matched_count=matched,
        no_eligible_observation_count=no_eligible,
        outside_lookback_count=outside_lookback,
        max_event_age_seconds=request.max_event_age_seconds,
        max_source_available_time_used=max_availability,
        offset=request.offset,
        next_offset=page_stop if page_stop < len(request.queries) else None,
        joined=[page[index] for index in sorted(page)],
        selected_observations=[
            SelectedObservation(
                observation_row_index=index, observation=request.observations[index]
            )
            for index in sorted(page_source_indices)
        ],
    )


OPERATION = Operation(
    id="skills.join_asof_observations",
    kind="skill",
    description=(
        "Join caller-supplied security/decision queries to the latest event and revision "
        "whose event and availability clocks are both observable. Use an ordered sweep, "
        "optional elapsed-time lookback, explicit unmatched reasons, and input-row lineage."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
