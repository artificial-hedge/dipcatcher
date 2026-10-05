"""Resolve observable upsert/delete streams without treating absence as deletion.

Each entity key is an independent stream. Eligible events have revision_time
and available_time no later than the decision. Greatest revision_time wins,
then greatest availability. Compatible same-clock records use greatest lexical
revision ID and earliest input row; contradictory records remain ambiguous or
raise according to the explicit policy. A later upsert can resurrect a key.
"""

from __future__ import annotations

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
    Annotated[str, Field(strict=True, max_length=1024)]
    | StrictBool
    | StrictInt
    | StrictFloat
    | None
)


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


class RevisionEvent(InputModel):
    entity_key: Name
    action: Literal["upsert", "delete"]
    revision_time: AwareDatetime
    available_time: AwareDatetime
    revision_id: Name
    source: Name
    values: dict[Name, Cell] | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("revision_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def action_payload(self) -> Self:
        if self.action == "upsert" and self.values is None:
            raise ValueError("upsert events require values")
        if self.action == "delete" and self.values is not None:
            raise ValueError("delete tombstones cannot carry values")
        return self


class Input(InputModel):
    events: list[RevisionEvent] = Field(max_length=10_000)
    decision_time: AwareDatetime
    entity_keys: list[Name] | None = Field(default=None, min_length=1, max_length=10_000)
    same_clock_policy: Literal["ambiguous", "error"] = "ambiguous"
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=200, strict=True, ge=1, le=500)
    max_witnesses: int = Field(default=5, strict=True, ge=2, le=20)

    @field_validator("decision_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @field_validator("entity_keys")
    @classmethod
    def unique_keys(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("entity_keys must be distinct")
        return value


class Selection(OutputModel):
    entity_key: str
    status: Literal["present", "deleted", "missing", "ambiguous"]
    selected_event_row_index: int | None
    eligible_event_rows: int
    winning_clock_rows: int
    conflicting_variants: int
    witness_row_indices: list[int]
    omitted_winning_clock_rows: int


class EventLineage(OutputModel):
    event_row_index: int
    event: RevisionEvent


class Output(OutputModel):
    decision_time: datetime
    input_event_count: int
    eligible_event_count: int
    unavailable_event_count: int
    not_yet_effective_event_count: int
    entity_count: int
    present_count: int
    deleted_count: int
    missing_count: int
    ambiguous_count: int
    same_clock_policy: str
    ordering_policy: str = "revision_time_then_availability_then_compatible_revision_id"
    absence_policy: str = "no_eligible_event_is_missing_never_implicit_delete"
    offset: int
    next_offset: int | None
    selections: list[Selection]
    event_lineage: list[EventLineage]


def execute(request: Input, context: OperationContext) -> Output:
    winning: dict[str, list[int]] = {}
    eligible_counts: dict[str, int] = {}
    unavailable = future = eligible = 0
    for index, event in enumerate(request.events):
        if event.available_time > request.decision_time:
            unavailable += 1
            continue
        if event.revision_time > request.decision_time:
            future += 1
            continue
        eligible += 1
        eligible_counts[event.entity_key] = eligible_counts.get(event.entity_key, 0) + 1
        previous = winning.get(event.entity_key)
        clock = event.revision_time, event.available_time
        if previous is None:
            winning[event.entity_key] = [index]
        else:
            prior = request.events[previous[0]]
            prior_clock = prior.revision_time, prior.available_time
            if clock > prior_clock:
                winning[event.entity_key] = [index]
            elif clock == prior_clock:
                previous.append(index)

    keys = sorted(
        request.entity_keys
        if request.entity_keys is not None
        else {event.entity_key for event in request.events}
    )
    counts = {"present": 0, "deleted": 0, "missing": 0, "ambiguous": 0}
    selections: list[Selection] = []
    lineage: set[int] = set()
    for position, key in enumerate(keys):
        indices = winning.get(key, [])
        variants: dict[bytes, int] = {}
        for index in indices:
            event = request.events[index]
            signature = canonical_json(
                {"action": event.action, "source": event.source, "values": event.values}
            )
            variants.setdefault(signature, index)
        selected: int | None = None
        status: Literal["present", "deleted", "missing", "ambiguous"]
        if not indices:
            status = "missing"
        elif len(variants) > 1:
            if request.same_clock_policy == "error":
                raise ValueError(
                    f"conflicting winning-clock revision variants for entity key {key!r}"
                )
            status = "ambiguous"
        else:
            selected = max(indices, key=lambda index: (request.events[index].revision_id, -index))
            status = "present" if request.events[selected].action == "upsert" else "deleted"
        counts[status] += 1
        if request.offset <= position < request.offset + request.limit:
            priority = (
                ([selected] if selected is not None else []) + list(variants.values()) + indices
            )
            witnesses = list(dict.fromkeys(priority))[: request.max_witnesses]
            lineage.update(witnesses)
            selections.append(
                Selection(
                    entity_key=key,
                    status=status,
                    selected_event_row_index=selected,
                    eligible_event_rows=eligible_counts.get(key, 0),
                    winning_clock_rows=len(indices),
                    conflicting_variants=len(variants) if len(variants) > 1 else 0,
                    witness_row_indices=witnesses,
                    omitted_winning_clock_rows=len(indices) - len(witnesses),
                )
            )
    stop = request.offset + len(selections)
    return Output(
        decision_time=request.decision_time,
        input_event_count=len(request.events),
        eligible_event_count=eligible,
        unavailable_event_count=unavailable,
        not_yet_effective_event_count=future,
        entity_count=len(keys),
        present_count=counts["present"],
        deleted_count=counts["deleted"],
        missing_count=counts["missing"],
        ambiguous_count=counts["ambiguous"],
        same_clock_policy=request.same_clock_policy,
        offset=request.offset,
        next_offset=stop if stop < len(keys) else None,
        selections=selections,
        event_lineage=[
            EventLineage(event_row_index=index, event=request.events[index])
            for index in sorted(lineage)
        ],
    )


OPERATION = Operation(
    id="skills.select_revision_tombstones",
    kind="skill",
    description="Resolve observable upsert/delete revision streams at a decision clock, preserving explicit tombstones and row lineage, allowing later resurrection, and reporting same-clock contradictions without treating absent rows as deletions.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
