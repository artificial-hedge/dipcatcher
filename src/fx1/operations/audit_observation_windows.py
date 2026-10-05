"""Audit declared continuous exposure and terminal-event/right-censoring windows.

Each window declares risk_start <= observation_start <= observation_end <=
planned_end, with risk_start < planned_end. Exposure measure is the half-open
interval [observation_start, observation_end). Point events at either endpoint
are eligible terminal observations. Delayed entry and the planned tail after
observation_end are reported separately; neither is called an internal gap.
Zero-duration observed windows are explicit and do not fabricate exposure.

Exposure observations are completed intervals: available_time must be at least
end_time and no later than the window's decision. Event observations have equal
start/end times with the same availability rule. These are declared completed
observation semantics, not a universal rule for announced future events.
Usable exposure intervals are clipped to the observed window; out-of-window
portions remain errors. Coverage is their exact integer-microsecond union, with
overlap duration and multiplicity excess computed separately. No regular grid,
calendar, interpolation, or matched point-pair materialization is used.

Window.available_time publishes the finalized window/outcome declaration, not
just its prospective schedule. It must follow observation_end and every used
source observation's availability. The decision must follow that publication.

An event outcome requires exactly its named visible event at observation_end and
no other visible event inside the observed window. Right censoring requires a
supplied reason and no such event. This validates supplied declarations, not the
absence of unreported events or a statistical independent-censoring assumption.

Bounds: 5000 windows, 10000 explicitly linked observations, O(N log N) interval
sorting, 200 findings with at most two source-row witnesses, and 200 window results
per page. All rows are audited before pagination. Duplicate IDs are ambiguous.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
        raise ValueError("clock must be an explicit timezone-aware ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


def _duration(start: datetime, end: datetime) -> int:
    delta = end - start
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class Window(InputModel):
    window_id: Name
    subject_id: Name
    risk_start: AwareDatetime
    planned_end: AwareDatetime
    observation_start: AwareDatetime
    observation_end: AwareDatetime
    available_time: AwareDatetime
    decision_time: AwareDatetime
    outcome: Literal["event", "right_censored"]
    terminal_observation_id: Name | None = None
    censoring_reason: Name | None = None

    @field_validator(
        "risk_start",
        "planned_end",
        "observation_start",
        "observation_end",
        "available_time",
        "decision_time",
        mode="before",
    )
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def outcome_fields(self) -> Self:
        if self.outcome == "event":
            if self.terminal_observation_id is None or self.censoring_reason is not None:
                raise ValueError("event windows require a terminal ID and no censoring reason")
        elif self.terminal_observation_id is not None or self.censoring_reason is None:
            raise ValueError("right-censored windows require a reason and no terminal ID")
        return self


class Observation(InputModel):
    observation_id: Name
    window_id: Name
    subject_id: Name
    kind: Literal["exposure", "event"]
    start_time: AwareDatetime
    end_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("start_time", "end_time", "available_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    windows: list[Window] = Field(min_length=1, max_length=5000)
    observations: list[Observation] = Field(max_length=10_000)
    overlap_policy: Literal["reject", "allow"] = "reject"
    max_diagnostics: int = Field(default=100, strict=True, ge=1, le=200)
    offset: int = Field(default=0, strict=True, ge=0, le=5000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)


class Finding(OutputModel):
    code: str
    window_row_index: int | None = None
    observation_row_indexes: list[int] = Field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None


class WindowResult(OutputModel):
    window_row_index: int
    window_id: str
    subject_id: str
    outcome: Literal["event", "right_censored"]
    coverage_assessment: Literal["complete", "gapped", "zero_duration", "unassessable"]
    declared_observation_count: int
    usable_exposure_count: int
    usable_event_count: int
    delayed_entry_microseconds: int | None
    unobserved_tail_microseconds: int | None
    observed_window_microseconds: int | None
    covered_microseconds: int | None
    missing_microseconds: int | None
    overlapping_microseconds: int | None
    multiplicity_excess_microseconds: int | None
    maximum_exposure_multiplicity: int | None
    gap_count: int | None
    terminal_observation_row_index: int | None
    terminal_declaration_consistent: bool | None
    maximum_used_available_time: datetime | None
    passed: bool


class Output(OutputModel):
    passed: bool
    window_count: int
    observation_count: int
    overlap_policy: Literal["reject", "allow"]
    complete_or_zero_duration_window_count: int
    internally_gapped_window_count: int
    unassessable_window_count: int
    issue_counts: dict[str, int]
    diagnostics: list[Finding]
    omitted_diagnostics: int
    windows: list[WindowResult]
    offset: int
    has_more: bool
    unreported_event_absence_verified: Literal[False] = False
    independent_censoring_verified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    issues: Counter[str] = Counter()
    findings: list[Finding] = []
    invalid_windows: set[int] = set()

    def report(
        code: str,
        window: int | None = None,
        observations: list[int] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        issues[code] += 1
        if window is not None:
            invalid_windows.add(window)
        if len(findings) < request.max_diagnostics:
            findings.append(
                Finding(
                    code=code,
                    window_row_index=window,
                    observation_row_indexes=observations or [],
                    start_time=start,
                    end_time=end,
                )
            )

    window_rows: dict[str, list[int]] = defaultdict(list)
    observation_rows: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(request.windows):
        window_rows[window.window_id].append(index)
    for index, observation in enumerate(request.observations):
        observation_rows[observation.observation_id].append(index)
    unique_windows = {name: rows[0] for name, rows in window_rows.items() if len(rows) == 1}
    unique_observations = {
        name: rows[0] for name, rows in observation_rows.items() if len(rows) == 1
    }
    assessable: list[bool] = []
    for index, window in enumerate(request.windows):
        valid = True
        if len(window_rows[window.window_id]) != 1:
            report("ambiguous_window_id", window=index)
            valid = False
        if not (
            window.risk_start < window.planned_end
            and window.risk_start
            <= window.observation_start
            <= window.observation_end
            <= window.planned_end
        ):
            report("invalid_window_geometry", window=index)
            valid = False
        if window.available_time < window.observation_end:
            report("window_declaration_precedes_observation_end", window=index)
            valid = False
        if (
            window.available_time > window.decision_time
            or window.observation_end > window.decision_time
        ):
            report("window_not_observable_at_decision", window=index)
            valid = False
        assessable.append(valid)

    linked_counts = [0] * len(request.windows)
    intervals: list[list[tuple[datetime, datetime, int]]] = [[] for _ in request.windows]
    events: list[list[int]] = [[] for _ in request.windows]
    used_clocks: list[datetime | None] = [None] * len(request.windows)
    for index, observation in enumerate(request.observations):
        position = unique_windows.get(observation.window_id)
        if position is not None:
            linked_counts[position] += 1
        if len(observation_rows[observation.observation_id]) != 1:
            report("ambiguous_observation_id", window=position, observations=[index])
            continue
        if position is None:
            report("unresolved_observation_window", observations=[index])
            continue
        window = request.windows[position]
        if observation.subject_id != window.subject_id:
            report("observation_subject_mismatch", window=position, observations=[index])
            continue
        if (observation.kind == "event" and observation.start_time != observation.end_time) or (
            observation.kind == "exposure" and observation.start_time >= observation.end_time
        ):
            report("invalid_observation_geometry", window=position, observations=[index])
            continue
        if observation.available_time < observation.end_time:
            report(
                "availability_precedes_completed_observation", window=position, observations=[index]
            )
            continue
        if observation.available_time > window.decision_time:
            report("observation_after_decision", window=position, observations=[index])
            continue
        if not assessable[position]:
            continue
        if (
            observation.start_time < window.observation_start
            or observation.end_time > window.observation_end
        ):
            report(
                "observation_outside_observed_window",
                window=position,
                observations=[index],
                start=observation.start_time,
                end=observation.end_time,
            )
        if observation.kind == "event":
            if window.observation_start <= observation.start_time <= window.observation_end:
                events[position].append(index)
            else:
                continue
        else:
            start, end = (
                max(window.observation_start, observation.start_time),
                min(window.observation_end, observation.end_time),
            )
            if start >= end:
                continue
            intervals[position].append((start, end, index))
        previous = used_clocks[position]
        used_clocks[position] = (
            observation.available_time
            if previous is None
            else max(previous, observation.available_time)
        )

    for index, used_available in enumerate(used_clocks):
        if used_available is not None and used_available > request.windows[index].available_time:
            report("window_declaration_precedes_used_evidence", window=index)

    results: list[WindowResult] = []
    complete_count = gapped_count = unassessable_count = 0
    for index, window in enumerate(request.windows):
        terminal = unique_observations.get(window.terminal_observation_id or "")
        assessment: Literal["complete", "gapped", "zero_duration", "unassessable"] = "unassessable"
        delayed = tail = duration = covered = missing = overlap = excess = multiplicity = gaps = (
            None
        )
        outcome_consistent: bool | None = None
        if not assessable[index]:
            unassessable_count += 1
        else:
            delayed = _duration(window.risk_start, window.observation_start)
            tail = _duration(window.observation_end, window.planned_end)
            duration = _duration(window.observation_start, window.observation_end)
            changes: Counter[datetime] = Counter()
            ordered = sorted(intervals[index])
            cursor = window.observation_start
            ending_row: int | None = None
            gaps = 0
            for start, end, row in ordered:
                changes[start] += 1
                changes[end] -= 1
                if start > cursor:
                    gaps += 1
                    witnesses = [row] if ending_row is None else [ending_row, row]
                    report(
                        "internal_exposure_gap",
                        window=index,
                        observations=witnesses,
                        start=cursor,
                        end=start,
                    )
                elif start < cursor and request.overlap_policy == "reject":
                    report(
                        "overlapping_exposure",
                        window=index,
                        observations=[ending_row, row] if ending_row is not None else [row],
                        start=start,
                        end=min(cursor, end),
                    )
                if end > cursor:
                    cursor, ending_row = end, row
            if cursor < window.observation_end:
                gaps += 1
                report(
                    "internal_exposure_gap",
                    window=index,
                    observations=[] if ending_row is None else [ending_row],
                    start=cursor,
                    end=window.observation_end,
                )
            covered = overlap = excess = multiplicity = active = 0
            previous_time = window.observation_start
            for time in sorted(changes):
                span = _duration(previous_time, time)
                if active:
                    covered += span
                    excess += span * (active - 1)
                    if active > 1:
                        overlap += span
                active += changes[time]
                multiplicity = max(multiplicity, active)
                previous_time = time
            missing = duration - covered
            assessment = (
                "zero_duration" if duration == 0 else "complete" if missing == 0 else "gapped"
            )
            if missing:
                gapped_count += 1
            else:
                complete_count += 1
            if window.outcome == "right_censored":
                outcome_consistent = not events[index]
                if not outcome_consistent:
                    report(
                        "event_contradicts_right_censoring",
                        window=index,
                        observations=events[index][:2],
                    )
            else:
                outcome_consistent = (
                    terminal is not None
                    and events[index] == [terminal]
                    and request.observations[terminal].start_time == window.observation_end
                )
                if not outcome_consistent:
                    report(
                        "terminal_event_declaration_mismatch",
                        window=index,
                        observations=events[index][:2],
                    )
        if request.offset <= index < request.offset + request.limit:
            results.append(
                WindowResult(
                    window_row_index=index,
                    window_id=window.window_id,
                    subject_id=window.subject_id,
                    outcome=window.outcome,
                    coverage_assessment=assessment,
                    declared_observation_count=linked_counts[index],
                    usable_exposure_count=len(intervals[index]),
                    usable_event_count=len(events[index]),
                    delayed_entry_microseconds=delayed,
                    unobserved_tail_microseconds=tail,
                    observed_window_microseconds=duration,
                    covered_microseconds=covered,
                    missing_microseconds=missing,
                    overlapping_microseconds=overlap,
                    multiplicity_excess_microseconds=excess,
                    maximum_exposure_multiplicity=multiplicity,
                    gap_count=gaps,
                    terminal_observation_row_index=terminal,
                    terminal_declaration_consistent=outcome_consistent,
                    maximum_used_available_time=used_clocks[index],
                    passed=index not in invalid_windows,
                )
            )
    return Output(
        passed=not issues,
        window_count=len(request.windows),
        observation_count=len(request.observations),
        overlap_policy=request.overlap_policy,
        complete_or_zero_duration_window_count=complete_count,
        internally_gapped_window_count=gapped_count,
        unassessable_window_count=unassessable_count,
        issue_counts=dict(sorted(issues.items())),
        diagnostics=findings,
        omitted_diagnostics=sum(issues.values()) - len(findings),
        windows=results,
        offset=request.offset,
        has_more=request.offset + len(results) < len(request.windows),
    )


OPERATION = Operation(
    id="skills.audit_observation_windows",
    kind="skill",
    description="Audit continuous observation exposure, delayed entry, internal coverage and explicit terminal-event/right-censoring declarations with source-row lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
