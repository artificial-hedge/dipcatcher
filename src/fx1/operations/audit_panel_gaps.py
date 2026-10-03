"""Find gaps on an explicit elapsed-time grid independently for each security.

This is a continuous UTC elapsed-time grid anchored at each security's earliest
observation. It does not infer exchange sessions, holidays, or market closures.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Observation(InputModel):
    security_id: str = Field(min_length=1, max_length=128, pattern=r"\S")
    event_time: AwareDatetime


class Input(InputModel):
    observations: list[Observation] = Field(min_length=1, max_length=10_000)
    interval_seconds: int = Field(strict=True, ge=1, le=31_536_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)


class Finding(OutputModel):
    security_id: str
    code: Literal["missing_grid_span", "duplicate_time", "out_of_order", "off_grid"]
    row_index: int | None = None
    event_time: datetime | None = None
    first_missing_time: datetime | None = None
    last_missing_time: datetime | None = None
    missing_points: int = 0


class Output(OutputModel):
    row_count: int
    security_count: int
    interval_seconds: int
    grid_policy: Literal["continuous_elapsed_utc_per_security_first_observation"] = (
        "continuous_elapsed_utc_per_security_first_observation"
    )
    passed: bool
    expected_grid_points: int
    observed_grid_points: int
    missing_grid_points: int
    duplicate_rows: int
    out_of_order_rows: int
    off_grid_rows: int
    finding_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    panels: dict[str, list[tuple[int, datetime]]] = defaultdict(list)
    for row_index, observation in enumerate(request.observations):
        panels[observation.security_id].append((row_index, observation.event_time.astimezone(UTC)))

    diagnostics: list[Finding] = []
    finding_count = 0
    expected_points = observed_points = missing_points = 0
    duplicate_rows = out_of_order_rows = off_grid_rows = 0
    interval = timedelta(seconds=request.interval_seconds)

    def report(finding: Finding) -> None:
        nonlocal finding_count
        finding_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    for security_id, rows in sorted(panels.items()):
        anchor = min(timestamp for _, timestamp in rows)
        last = max(timestamp for _, timestamp in rows)
        last_slot = (last - anchor) // interval
        expected_points += last_slot + 1
        seen: set[datetime] = set()
        aligned_slots: set[int] = set()
        previous: datetime | None = None
        for row_index, timestamp in rows:
            if previous is not None and timestamp < previous:
                out_of_order_rows += 1
                report(
                    Finding(
                        security_id=security_id,
                        code="out_of_order",
                        row_index=row_index,
                        event_time=timestamp,
                    )
                )
            previous = timestamp
            if timestamp in seen:
                duplicate_rows += 1
                report(
                    Finding(
                        security_id=security_id,
                        code="duplicate_time",
                        row_index=row_index,
                        event_time=timestamp,
                    )
                )
            seen.add(timestamp)
            elapsed = timestamp - anchor
            if elapsed % interval:
                off_grid_rows += 1
                report(
                    Finding(
                        security_id=security_id,
                        code="off_grid",
                        row_index=row_index,
                        event_time=timestamp,
                    )
                )
            else:
                aligned_slots.add(elapsed // interval)

        observed_points += len(aligned_slots)
        missing_points += last_slot + 1 - len(aligned_slots)
        previous_slot = -1
        # A sentinel counts missing grid points before an off-grid final row.
        # Spans are compressed: very long gaps never allocate every timestamp.
        for slot in [*sorted(aligned_slots), last_slot + 1]:
            missing = slot - previous_slot - 1
            if missing:
                report(
                    Finding(
                        security_id=security_id,
                        code="missing_grid_span",
                        first_missing_time=anchor + (previous_slot + 1) * interval,
                        last_missing_time=anchor + (slot - 1) * interval,
                        missing_points=missing,
                    )
                )
            previous_slot = slot

    return Output(
        row_count=len(request.observations),
        security_count=len(panels),
        interval_seconds=request.interval_seconds,
        passed=finding_count == 0,
        expected_grid_points=expected_points,
        observed_grid_points=observed_points,
        missing_grid_points=missing_points,
        duplicate_rows=duplicate_rows,
        out_of_order_rows=out_of_order_rows,
        off_grid_rows=off_grid_rows,
        finding_count=finding_count,
        diagnostics=diagnostics,
        omitted_diagnostics=finding_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_panel_gaps",
    kind="skill",
    description=(
        "Audit per-security observations on an explicit continuous elapsed-time grid; "
        "report missing spans, duplicates, input time reversals, and off-grid rows. "
        "This does not infer trading sessions or fill missing observations."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
