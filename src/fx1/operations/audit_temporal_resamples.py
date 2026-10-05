"""Audit supplied temporal resample positions and declared block boundaries.

Each replicate references one caller-ordered source stream and an explicit
decision origin. Every referenced row must have available_time <= origin.
Requiring event completion is independent and configurable: announced future
events may be observable but are not completed. Availability before event time
is not itself an error. Source input row indexes define continuation order.

Output positions cover [0, resample_length). Duplicate positions are invalid and
excluded from adjacency checks, while repeated source rows at distinct output
positions are allowed. Block IDs must form contiguous runs in the observed
unambiguous positions. Within a block, continuation can be unchecked, consecutive
or circular. Circular event resets are exempt only from within-block event-order
checks when explicitly allowed; whole-replicate ordering remains independent.

No RNG, stationary-source assumption, fixed block length or statistical inference
is validated. Missing positions are compressed into spans, never materialized.
The entire supplied source catalog is checked under its declared order policy,
including streams unused by a replicate. Point checks include duplicate and
out-of-range membership rows; these rows cannot silently avoid availability rules.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Order = Literal["unchecked", "nondecreasing", "strictly_increasing"]


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


class SourceRow(InputModel):
    event_time: AwareDatetime
    available_time: AwareDatetime

    @field_validator("event_time", "available_time", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)


class SourceStream(InputModel):
    stream_id: Name
    rows: list[SourceRow] = Field(max_length=10_000)


class Replicate(InputModel):
    replicate_id: Name
    stream_id: Name
    decision_origin: AwareDatetime
    resample_length: int = Field(strict=True, ge=0, le=10_000)

    @field_validator("decision_origin", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Membership(InputModel):
    replicate_id: Name
    output_position: int = Field(strict=True, ge=0, le=9_999)
    block_id: Name
    source_row_index: int = Field(strict=True, ge=0, le=9_999)


class Input(InputModel):
    streams: list[SourceStream] = Field(max_length=64)
    replicates: list[Replicate] = Field(max_length=500)
    memberships: list[Membership] = Field(max_length=20_000)
    source_event_order: Order = "nondecreasing"
    within_block_event_order: Order = "nondecreasing"
    replicate_event_order: Order = "unchecked"
    block_continuation_policy: Literal["unchecked", "consecutive", "circular"] = "consecutive"
    allow_circular_event_reset: bool = Field(default=True, strict=True)
    require_completed_events: bool = Field(default=True, strict=True)
    offset: int = Field(default=0, strict=True, ge=0, le=500)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=3, le=10)

    @model_validator(mode="after")
    def unique_bounded_catalog(self) -> Self:
        if len({row.stream_id for row in self.streams}) != len(self.streams):
            raise ValueError("stream IDs must be distinct")
        if len({row.replicate_id for row in self.replicates}) != len(self.replicates):
            raise ValueError("replicate IDs must be distinct")
        if sum(len(stream.rows) for stream in self.streams) > 10_000:
            raise ValueError("source streams may contain at most 10000 rows combined")
        return self


class Diagnostic(OutputModel):
    code: str
    is_violation: bool
    replicate_id: str | None
    replicate_row_index: int | None
    stream_row_index: int | None
    first_output_position: int | None
    last_output_position: int | None
    affected_count: int
    membership_row_indexes: list[int]
    source_row_indexes: list[int]
    omitted_membership_row_indexes: int


class ReplicateSummary(OutputModel):
    replicate_id: str
    replicate_row_index: int
    stream_row_index: int | None
    membership_row_count: int
    expected_position_count: int
    observed_expected_positions: int
    missing_position_count: int
    invalid_source_reference_rows: int
    unavailable_source_rows: int
    future_event_rows: int
    unambiguous_observed_block_count: int
    audited_adjacent_position_pairs: int
    unassessed_expected_adjacencies: int
    observed_circular_continuation_wraps: int
    source_event_order_passed: bool | None
    issue_counts: dict[str, int]
    violation_count: int
    passed: bool


class Output(OutputModel):
    passed: bool
    stream_count: int
    source_row_count: int
    replicate_count: int
    membership_row_count: int
    source_event_order: Order
    within_block_event_order: Order
    replicate_event_order: Order
    block_continuation_policy: str
    allow_circular_event_reset: bool
    require_completed_events: bool
    repeated_source_inclusion_allowed: Literal[True] = True
    inference_validity_established: Literal[False] = False
    issue_counts: dict[str, int]
    violation_count: int
    diagnostic_count: int
    omitted_diagnostics: int
    offset: int
    next_offset: int | None
    replicates: list[ReplicateSummary]
    diagnostics: list[Diagnostic]


def _out_of_order(previous: datetime, current: datetime, policy: Order) -> bool:
    return policy != "unchecked" and (
        current < previous or (policy == "strictly_increasing" and current == previous)
    )


def execute(request: Input, context: OperationContext) -> Output:
    stream_indexes = {stream.stream_id: index for index, stream in enumerate(request.streams)}
    replicate_indexes = {row.replicate_id: index for index, row in enumerate(request.replicates)}
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(request.memberships):
        grouped[row.replicate_id].append(index)
    diagnostics: list[Diagnostic] = []
    issues: Counter[str] = Counter()
    per_replicate: dict[int, Counter[str]] = defaultdict(Counter)
    violations: Counter[int] = Counter()
    total_violations = 0
    source_passes = [True] * len(request.streams)

    def report(
        code: str,
        replicate_id: str | None,
        stream_index: int | None,
        membership_rows: list[int],
        source_rows: list[int],
        count: int = 1,
        first: int | None = None,
        last: int | None = None,
        violation: bool = True,
    ) -> None:
        nonlocal total_violations
        replicate_index = replicate_indexes.get(replicate_id) if replicate_id is not None else None
        issues[code] += 1
        total_violations += violation
        if replicate_index is not None:
            per_replicate[replicate_index][code] += 1
            violations[replicate_index] += violation
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Diagnostic(
                    code=code,
                    is_violation=violation,
                    replicate_id=replicate_id,
                    replicate_row_index=replicate_index,
                    stream_row_index=stream_index,
                    first_output_position=first,
                    last_output_position=last,
                    affected_count=count,
                    membership_row_indexes=membership_rows[: request.max_row_indexes],
                    source_row_indexes=source_rows[: request.max_row_indexes],
                    omitted_membership_row_indexes=max(
                        0, len(membership_rows) - request.max_row_indexes
                    ),
                )
            )

    for catalog_stream_index, stream in enumerate(request.streams):
        for index in range(1, len(stream.rows)):
            if _out_of_order(
                stream.rows[index - 1].event_time,
                stream.rows[index].event_time,
                request.source_event_order,
            ):
                source_passes[catalog_stream_index] = False
                report("source_event_order", None, catalog_stream_index, [], [index - 1, index])
    for replicate_id, rows in grouped.items():
        if replicate_id not in replicate_indexes:
            report("unknown_replicate", replicate_id, None, rows, [], len(rows))

    summaries: list[ReplicateSummary] = []
    for replicate_index, replicate in enumerate(request.replicates):
        rows = grouped.get(replicate.replicate_id, [])
        stream_index = stream_indexes.get(replicate.stream_id)
        source = request.streams[stream_index] if stream_index is not None else None
        if source is None:
            report("unknown_source_stream", replicate.replicate_id, None, rows, [], len(rows))
        by_position: dict[int, list[int]] = defaultdict(list)
        valid_references: set[int] = set()
        invalid_references = unavailable = future_events = 0
        for index in rows:
            draw = request.memberships[index]
            by_position[draw.output_position].append(index)
            if draw.output_position >= replicate.resample_length:
                report(
                    "position_outside_replicate",
                    replicate.replicate_id,
                    stream_index,
                    [index],
                    [],
                    first=draw.output_position,
                    last=draw.output_position,
                )
            if source is None or draw.source_row_index >= len(source.rows):
                invalid_references += 1
                if source is not None:
                    report(
                        "source_row_outside_stream",
                        replicate.replicate_id,
                        stream_index,
                        [index],
                        [draw.source_row_index],
                    )
                continue
            valid_references.add(index)
            observation = source.rows[draw.source_row_index]
            if observation.available_time > replicate.decision_origin:
                unavailable += 1
                report(
                    "source_not_available_at_origin",
                    replicate.replicate_id,
                    stream_index,
                    [index],
                    [draw.source_row_index],
                )
            if observation.event_time > replicate.decision_origin:
                future_events += 1
                report(
                    "event_after_origin",
                    replicate.replicate_id,
                    stream_index,
                    [index],
                    [draw.source_row_index],
                    violation=request.require_completed_events,
                )
        cursor = covered = 0
        ordered: list[int] = []
        for position, position_rows in sorted(by_position.items()):
            if len(position_rows) > 1:
                report(
                    "duplicate_output_position",
                    replicate.replicate_id,
                    stream_index,
                    position_rows,
                    [],
                    len(position_rows),
                    position,
                    position,
                )
            if position >= replicate.resample_length:
                continue
            covered += 1
            if position > cursor:
                report(
                    "missing_position_span",
                    replicate.replicate_id,
                    stream_index,
                    [position_rows[0]],
                    [],
                    position - cursor,
                    cursor,
                    position - 1,
                )
            cursor = position + 1
            if len(position_rows) == 1:
                ordered.append(position_rows[0])
        if cursor < replicate.resample_length:
            report(
                "missing_position_span",
                replicate.replicate_id,
                stream_index,
                [],
                [],
                replicate.resample_length - cursor,
                cursor,
                replicate.resample_length - 1,
            )

        first_block_rows: dict[str, int] = {}
        previous_index: int | None = None
        audited_pairs = wraps = 0
        for index in ordered:
            draw = request.memberships[index]
            previous = request.memberships[previous_index] if previous_index is not None else None
            if (
                draw.block_id in first_block_rows
                and previous is not None
                and previous.block_id != draw.block_id
            ):
                assert previous_index is not None
                report(
                    "noncontiguous_block_id",
                    replicate.replicate_id,
                    stream_index,
                    [first_block_rows[draw.block_id], previous_index, index],
                    [],
                )
            first_block_rows.setdefault(draw.block_id, index)
            if (
                previous is not None
                and previous_index in valid_references
                and index in valid_references
                and draw.output_position == previous.output_position + 1
                and source is not None
            ):
                assert previous_index is not None
                audited_pairs += 1
                first_observation = source.rows[previous.source_row_index]
                observation = source.rows[draw.source_row_index]
                source_rows = [previous.source_row_index, draw.source_row_index]
                if _out_of_order(
                    first_observation.event_time,
                    observation.event_time,
                    request.replicate_event_order,
                ):
                    report(
                        "replicate_event_order",
                        replicate.replicate_id,
                        stream_index,
                        [previous_index, index],
                        source_rows,
                    )
                if previous.block_id == draw.block_id:
                    circular_wrap = (
                        request.block_continuation_policy == "circular"
                        and previous.source_row_index == len(source.rows) - 1
                        and draw.source_row_index == 0
                    )
                    wraps += circular_wrap
                    if request.block_continuation_policy != "unchecked":
                        expected = previous.source_row_index + 1
                        if request.block_continuation_policy == "circular":
                            expected %= len(source.rows)
                        if draw.source_row_index != expected:
                            report(
                                "block_continuation",
                                replicate.replicate_id,
                                stream_index,
                                [previous_index, index],
                                source_rows,
                            )
                    if not (circular_wrap and request.allow_circular_event_reset) and _out_of_order(
                        first_observation.event_time,
                        observation.event_time,
                        request.within_block_event_order,
                    ):
                        report(
                            "within_block_event_order",
                            replicate.replicate_id,
                            stream_index,
                            [previous_index, index],
                            source_rows,
                        )
            previous_index = index
        if request.offset <= replicate_index < request.offset + request.limit:
            source_valid = source_passes[stream_index] if stream_index is not None else None
            summaries.append(
                ReplicateSummary(
                    replicate_id=replicate.replicate_id,
                    replicate_row_index=replicate_index,
                    stream_row_index=stream_index,
                    membership_row_count=len(rows),
                    expected_position_count=replicate.resample_length,
                    observed_expected_positions=covered,
                    missing_position_count=replicate.resample_length - covered,
                    invalid_source_reference_rows=invalid_references,
                    unavailable_source_rows=unavailable,
                    future_event_rows=future_events,
                    unambiguous_observed_block_count=len(first_block_rows),
                    audited_adjacent_position_pairs=audited_pairs,
                    unassessed_expected_adjacencies=max(0, replicate.resample_length - 1)
                    - audited_pairs,
                    observed_circular_continuation_wraps=wraps,
                    source_event_order_passed=source_valid,
                    issue_counts=dict(sorted(per_replicate[replicate_index].items())),
                    violation_count=violations[replicate_index],
                    passed=violations[replicate_index] == 0 and source_valid is True,
                )
            )
    diagnostic_count = sum(issues.values())
    end = min(len(request.replicates), request.offset + request.limit)
    return Output(
        passed=total_violations == 0,
        stream_count=len(request.streams),
        source_row_count=sum(len(stream.rows) for stream in request.streams),
        replicate_count=len(request.replicates),
        membership_row_count=len(request.memberships),
        source_event_order=request.source_event_order,
        within_block_event_order=request.within_block_event_order,
        replicate_event_order=request.replicate_event_order,
        block_continuation_policy=request.block_continuation_policy,
        allow_circular_event_reset=request.allow_circular_event_reset,
        require_completed_events=request.require_completed_events,
        issue_counts=dict(sorted(issues.items())),
        violation_count=total_violations,
        diagnostic_count=diagnostic_count,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
        offset=request.offset,
        next_offset=end if end < len(request.replicates) else None,
        replicates=summaries,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_temporal_resamples",
    kind="skill",
    description=(
        "Audit supplied resample positions and blocks for origin-time availability, "
        "event ordering, missing/duplicate positions, source lineage and continuation policies."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
