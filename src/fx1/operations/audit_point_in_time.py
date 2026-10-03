"""Audit observability, with optional completed-event and ingestion constraints.

Announcements may precede their economic event, including future corporate
actions. Completed-event checks apply only when explicitly requested for data
such as closed bars; availability by the decision clock is always required.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC
from typing import Literal

from pydantic import AwareDatetime, Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

FailureCode = Literal[
    "available_before_event",
    "event_after_decision",
    "available_after_decision",
    "missing_ingestion",
    "ingested_before_availability",
    "ingested_after_decision",
]


class Observation(InputModel):
    event_time: AwareDatetime
    available_time: AwareDatetime
    ingested_time: AwareDatetime | None = None


class Input(InputModel):
    observations: list[Observation] = Field(min_length=1, max_length=10_000)
    decision_time: AwareDatetime
    require_completed_events: bool = Field(default=False, strict=True)
    require_ingestion: bool = Field(default=False, strict=True)
    require_ingestion_by_decision: bool = Field(default=False, strict=True)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)


class Finding(OutputModel):
    row_index: int
    code: FailureCode


class Output(OutputModel):
    row_count: int
    valid_rows: int
    invalid_rows: int
    passed: bool
    violation_count: int
    violation_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    completed_events_required: bool
    ingestion_required: bool
    ingestion_by_decision_required: bool


def execute(request: Input, context: OperationContext) -> Output:
    counts: Counter[str] = Counter()
    diagnostics: list[Finding] = []
    invalid_rows = 0
    ingestion_required = request.require_ingestion or request.require_ingestion_by_decision
    decision_time = request.decision_time.astimezone(UTC)

    for row_index, observation in enumerate(request.observations):
        codes: list[FailureCode] = []
        event_time = observation.event_time.astimezone(UTC)
        available_time = observation.available_time.astimezone(UTC)
        if request.require_completed_events and available_time < event_time:
            codes.append("available_before_event")
        if request.require_completed_events and event_time > decision_time:
            codes.append("event_after_decision")
        if available_time > decision_time:
            codes.append("available_after_decision")
        if observation.ingested_time is None:
            if ingestion_required:
                codes.append("missing_ingestion")
        else:
            ingested_time = observation.ingested_time.astimezone(UTC)
            if ingested_time < available_time:
                codes.append("ingested_before_availability")
            if request.require_ingestion_by_decision and ingested_time > decision_time:
                codes.append("ingested_after_decision")
        if codes:
            invalid_rows += 1
        for code in codes:
            counts[code] += 1
            if len(diagnostics) < request.max_diagnostics:
                diagnostics.append(Finding(row_index=row_index, code=code))

    violation_count = sum(counts.values())
    return Output(
        row_count=len(request.observations),
        valid_rows=len(request.observations) - invalid_rows,
        invalid_rows=invalid_rows,
        passed=invalid_rows == 0,
        violation_count=violation_count,
        violation_counts=dict(sorted(counts.items())),
        diagnostics=diagnostics,
        omitted_diagnostics=violation_count - len(diagnostics),
        completed_events_required=request.require_completed_events,
        ingestion_required=ingestion_required,
        ingestion_by_decision_required=request.require_ingestion_by_decision,
    )


OPERATION = Operation(
    id="skills.audit_point_in_time",
    kind="skill",
    description=(
        "Audit timezone-aware availability by the decision clock, optional completed-event "
        "ordering, and ingestion lineage. Future announced events are allowed unless "
        "completed events are required; ingestion by decision is an optional constraint."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
