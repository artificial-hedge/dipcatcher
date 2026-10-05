"""Measure completed-event frontiers using supplied publication and ingestion clocks.

Publication readiness is max(event_time, available_time); local readiness also
requires ingested_time. Thus an announced future event is not yet completed, and
an ingestion clock before publication cannot make the record observable early.
Each frontier selects the greatest completed event time. Ties retain the earliest
readiness clock, then the earliest input row, rather than selecting a revision.

Counts describe supplied records, including duplicates. A frontier does not prove
that earlier events are complete or that absent events ever existed. Two sorted
activation sweeps answer queries in O((records + queries) log(records + queries)).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Convention = Literal["publication", "publication_and_ingestion"]
Status = Literal["available", "stale", "no_completed_observable_event"]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
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


def _microseconds(delta: timedelta) -> int:
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class Observation(InputModel):
    stream_id: Name
    event_time: AwareDatetime
    available_time: AwareDatetime
    ingested_time: AwareDatetime

    @field_validator("event_time", "available_time", "ingested_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class Decision(InputModel):
    stream_id: Name
    decision_time: AwareDatetime

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    observations: list[Observation] = Field(max_length=10_000)
    decisions: list[Decision] = Field(max_length=10_000)
    convention: Convention = "publication_and_ingestion"
    max_event_age_microseconds: int | None = Field(default=None, strict=True, ge=0, le=10**18)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)


class Frontier(OutputModel):
    decision_row_index: int
    status: Status
    publication_frontier_row_index: int | None
    ingestion_frontier_row_index: int | None
    selected_row_index: int | None
    event_time: datetime | None
    available_time: datetime | None
    ingested_time: datetime | None
    ready_time: datetime | None
    event_age_microseconds: int | None
    publication_lag_microseconds: int | None
    ingestion_lag_microseconds: int | None
    readiness_age_microseconds: int | None
    publication_ingestion_event_gap_microseconds: int | None
    completed_published_record_count: int
    completed_published_ingested_record_count: int
    pending_ingestion_record_count: int


class Output(OutputModel):
    observation_count: int
    decision_count: int
    convention: Convention
    max_event_age_microseconds: int | None
    status_counts: dict[str, int]
    negative_ingestion_lag_record_count: int
    tie_policy: Literal["latest_event_then_earliest_readiness_then_input_row"] = (
        "latest_event_then_earliest_readiness_then_input_row"
    )
    completeness: Literal["not_inferred_from_supplied_records"] = (
        "not_inferred_from_supplied_records"
    )
    record_counts_include_duplicates: Literal[True] = True
    offset: int
    next_offset: int | None
    frontiers: list[Frontier]


def execute(request: Input, context: OperationContext) -> Output:
    publication = sorted(
        (max(row.event_time, row.available_time), index)
        for index, row in enumerate(request.observations)
    )
    ingestion = sorted(
        (max(row.event_time, row.available_time, row.ingested_time), index)
        for index, row in enumerate(request.observations)
    )
    published_counts: Counter[str] = Counter()
    ingested_counts: Counter[str] = Counter()
    published_best: dict[str, int] = {}
    ingested_best: dict[str, int] = {}
    publication_cursor = 0
    ingestion_cursor = 0
    status_counts: Counter[str] = Counter()
    page: dict[int, Frontier] = {}
    page_end = min(len(request.decisions), request.offset + request.limit)

    for query_index in sorted(
        range(len(request.decisions)), key=lambda index: request.decisions[index].decision_time
    ):
        query = request.decisions[query_index]
        while (
            publication_cursor < len(publication)
            and publication[publication_cursor][0] <= query.decision_time
        ):
            row_index = publication[publication_cursor][1]
            row = request.observations[row_index]
            published_counts[row.stream_id] += 1
            previous = published_best.get(row.stream_id)
            if previous is None or row.event_time > request.observations[previous].event_time:
                published_best[row.stream_id] = row_index
            publication_cursor += 1
        while (
            ingestion_cursor < len(ingestion)
            and ingestion[ingestion_cursor][0] <= query.decision_time
        ):
            row_index = ingestion[ingestion_cursor][1]
            row = request.observations[row_index]
            ingested_counts[row.stream_id] += 1
            previous = ingested_best.get(row.stream_id)
            if previous is None or row.event_time > request.observations[previous].event_time:
                ingested_best[row.stream_id] = row_index
            ingestion_cursor += 1

        published_index = published_best.get(query.stream_id)
        ingested_index = ingested_best.get(query.stream_id)
        selected_index = published_index if request.convention == "publication" else ingested_index
        selected = request.observations[selected_index] if selected_index is not None else None
        age = _microseconds(query.decision_time - selected.event_time) if selected else None
        status: Status = "available" if selected else "no_completed_observable_event"
        if (
            age is not None
            and request.max_event_age_microseconds is not None
            and age > request.max_event_age_microseconds
        ):
            status = "stale"
        status_counts[status] += 1
        if not request.offset <= query_index < page_end:
            continue

        ready: datetime | None = None
        if selected:
            ready = max(selected.event_time, selected.available_time)
            if request.convention == "publication_and_ingestion":
                ready = max(ready, selected.ingested_time)
        event_gap = None
        if published_index is not None and ingested_index is not None:
            event_gap = _microseconds(
                request.observations[published_index].event_time
                - request.observations[ingested_index].event_time
            )
        page[query_index] = Frontier(
            decision_row_index=query_index,
            status=status,
            publication_frontier_row_index=published_index,
            ingestion_frontier_row_index=ingested_index,
            selected_row_index=selected_index,
            event_time=selected.event_time if selected else None,
            available_time=selected.available_time if selected else None,
            ingested_time=selected.ingested_time if selected else None,
            ready_time=ready,
            event_age_microseconds=age,
            publication_lag_microseconds=(
                _microseconds(selected.available_time - selected.event_time) if selected else None
            ),
            ingestion_lag_microseconds=(
                _microseconds(selected.ingested_time - selected.available_time)
                if selected
                else None
            ),
            readiness_age_microseconds=(
                _microseconds(query.decision_time - ready) if ready else None
            ),
            publication_ingestion_event_gap_microseconds=event_gap,
            completed_published_record_count=published_counts[query.stream_id],
            completed_published_ingested_record_count=ingested_counts[query.stream_id],
            pending_ingestion_record_count=(
                published_counts[query.stream_id] - ingested_counts[query.stream_id]
            ),
        )
    return Output(
        observation_count=len(request.observations),
        decision_count=len(request.decisions),
        convention=request.convention,
        max_event_age_microseconds=request.max_event_age_microseconds,
        status_counts=dict(sorted(status_counts.items())),
        negative_ingestion_lag_record_count=sum(
            row.ingested_time < row.available_time for row in request.observations
        ),
        offset=request.offset,
        next_offset=page_end if page_end < len(request.decisions) else None,
        frontiers=[page[index] for index in sorted(page)],
    )


OPERATION = Operation(
    id="skills.build_availability_frontier",
    kind="skill",
    description=(
        "Sweep completed-event publication and ingestion frontiers with exact staleness, "
        "backlog counts, and source lineage; supplied rows do not establish completeness."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
