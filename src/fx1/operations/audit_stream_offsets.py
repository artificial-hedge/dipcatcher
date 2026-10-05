"""Audit caller-declared inclusive offset ranges without materializing missing rows.

Identity is (source_id, stream_id, partition_id). Offsets are nonnegative signed
64-bit integers and missing ranges are compressed with integer arithmetic. The
same offset in different identities is unrelated. Hashes are caller-supplied
SHA-256 labels: this operation does not read or authenticate payload bytes.

Duplicate records have identical event clocks and payload hashes. Reusing an
offset with different clocks or hashes is always a conflict. Optional ordering
rules inspect adjacent input rows, and adjacent distinct offsets after sorting.
The latter compares the earlier offset's latest event with the later offset's
earliest event, so conflicting rows cannot conceal a reversal. Diagnostics carry
actual input indexes and have a separate bounded page-independent limit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Offset = Annotated[int, Field(strict=True, ge=0, le=2**63 - 1)]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
StreamKey = tuple[str, str, str]


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


class StreamIdentity(InputModel):
    source_id: Name
    stream_id: Name
    partition_id: Name

    def key(self) -> StreamKey:
        return (self.source_id, self.stream_id, self.partition_id)


class ExpectedRange(StreamIdentity):
    start_offset: Offset
    end_offset: Offset

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end_offset < self.start_offset:
            raise ValueError("expected end_offset must be at least start_offset")
        return self


class Record(StreamIdentity):
    offset: Offset
    event_time: AwareDatetime
    payload_sha256: Digest

    @field_validator("event_time", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _clock(value)


class Input(InputModel):
    expected_ranges: list[ExpectedRange] = Field(max_length=2_048)
    records: list[Record] = Field(max_length=10_000)
    duplicate_policy: Literal["reject", "allow_identical"] = "reject"
    require_input_offset_order: bool = Field(default=False, strict=True)
    require_event_time_order: bool = Field(default=False, strict=True)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=2, le=10)
    offset: int = Field(default=0, strict=True, ge=0, le=2_048)
    limit: int = Field(default=100, strict=True, ge=1, le=500)

    @model_validator(mode="after")
    def unique_bounded_streams(self) -> Self:
        keys = {row.key() for row in self.expected_ranges}
        if len(keys) != len(self.expected_ranges):
            raise ValueError("each source/stream/partition may have one expected range")
        keys.update(row.key() for row in self.records)
        if len(keys) > 2_048:
            raise ValueError("at most 2048 distinct source/stream/partition identities are allowed")
        return self


class Diagnostic(OutputModel):
    code: str
    is_violation: bool
    source_id: str
    stream_id: str
    partition_id: str
    start_offset: int | None
    end_offset: int | None
    affected_count: int
    row_indexes: list[int]
    row_indexes_truncated: bool


class StreamSummary(OutputModel):
    source_id: str
    stream_id: str
    partition_id: str
    expected_range_row_index: int | None
    expected_start_offset: int | None
    expected_end_offset: int | None
    expected_offset_count: int | None
    record_count: int
    distinct_offset_count: int
    observed_min_offset: int | None
    observed_max_offset: int | None
    observed_expected_offset_count: int | None
    missing_offset_count: int | None
    missing_range_count: int | None
    out_of_range_record_count: int | None
    duplicate_row_count: int
    exact_duplicate_row_count: int
    payload_conflict_offset_count: int
    event_clock_conflict_offset_count: int
    input_offset_reversal_count: int
    adjacent_offset_event_reversal_count: int
    coverage_complete: bool | None
    passed: bool


class Output(OutputModel):
    passed: bool
    stream_count: int
    record_count: int
    expected_range_count: int
    duplicate_policy: str
    require_input_offset_order: bool
    require_event_time_order: bool
    issue_counts: dict[str, int]
    violation_count: int
    diagnostic_count: int
    diagnostics_truncated: bool
    hash_scope: Literal["supplied_labels_not_verified_payload_bytes"] = (
        "supplied_labels_not_verified_payload_bytes"
    )
    range_boundary: Literal["inclusive_start_and_end"] = "inclusive_start_and_end"
    offset: int
    next_offset: int | None
    streams: list[StreamSummary]
    diagnostics: list[Diagnostic]


def execute(request: Input, context: OperationContext) -> Output:
    expected = {row.key(): index for index, row in enumerate(request.expected_ranges)}
    grouped: dict[StreamKey, list[int]] = defaultdict(list)
    for index, row in enumerate(request.records):
        grouped[row.key()].append(index)
    keys = sorted(set(expected) | set(grouped))
    diagnostics: list[Diagnostic] = []
    issues: Counter[str] = Counter()
    violation_count = 0
    streams: list[StreamSummary] = []

    def report(
        key: StreamKey,
        code: str,
        is_violation: bool,
        low: int | None,
        high: int | None,
        affected_count: int,
        indexes: list[int],
    ) -> None:
        nonlocal violation_count
        issues[code] += 1
        violation_count += is_violation
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Diagnostic(
                    code=code,
                    is_violation=is_violation,
                    source_id=key[0],
                    stream_id=key[1],
                    partition_id=key[2],
                    start_offset=low,
                    end_offset=high,
                    affected_count=affected_count,
                    row_indexes=indexes[: request.max_row_indexes],
                    row_indexes_truncated=len(indexes) > request.max_row_indexes,
                )
            )

    for stream_index, key in enumerate(keys):
        before_violations = violation_count
        indexes = grouped.get(key, [])
        range_index = expected.get(key)
        interval = request.expected_ranges[range_index] if range_index is not None else None
        offsets: dict[int, list[int]] = defaultdict(list)
        input_reversals = 0
        previous_index: int | None = None
        for index in indexes:
            record = request.records[index]
            offsets[record.offset].append(index)
            if previous_index is not None:
                previous = request.records[previous_index]
                if record.offset < previous.offset:
                    input_reversals += 1
                    report(
                        key,
                        "input_offset_reversal",
                        request.require_input_offset_order,
                        record.offset,
                        previous.offset,
                        2,
                        [previous_index, index],
                    )
            previous_index = index
        if interval is None:
            report(key, "undeclared_stream", True, None, None, len(indexes), indexes)

        ordered_offsets = sorted(offsets)
        duplicate_rows = 0
        exact_duplicate_rows = 0
        payload_conflicts = 0
        clock_conflicts = 0
        event_reversals = 0
        out_of_range_rows = 0
        observed_expected = 0
        missing_count = 0
        missing_ranges = 0
        next_expected = interval.start_offset if interval else 0
        previous_max_index: int | None = None
        for offset in ordered_offsets:
            offset_indexes = offsets[offset]
            duplicate_rows += len(offset_indexes) - 1
            variants = {
                (request.records[index].event_time, request.records[index].payload_sha256)
                for index in offset_indexes
            }
            exact_duplicate_rows += len(offset_indexes) - len(variants)
            if len(offset_indexes) > 1:
                report(
                    key,
                    "duplicate_offset",
                    request.duplicate_policy == "reject",
                    offset,
                    offset,
                    len(offset_indexes) - 1,
                    offset_indexes,
                )
            hash_indexes: dict[str, int] = {}
            clock_indexes: dict[datetime, int] = {}
            for index in offset_indexes:
                record = request.records[index]
                hash_indexes.setdefault(record.payload_sha256, index)
                clock_indexes.setdefault(record.event_time, index)
            if len(hash_indexes) > 1:
                payload_conflicts += 1
                report(
                    key,
                    "payload_hash_conflict",
                    True,
                    offset,
                    offset,
                    len(offset_indexes),
                    list(hash_indexes.values()),
                )
            if len(clock_indexes) > 1:
                clock_conflicts += 1
                report(
                    key,
                    "event_clock_conflict",
                    True,
                    offset,
                    offset,
                    len(offset_indexes),
                    list(clock_indexes.values()),
                )
            minimum_index = min(offset_indexes, key=lambda index: request.records[index].event_time)
            maximum_index = max(offset_indexes, key=lambda index: request.records[index].event_time)
            if (
                previous_max_index is not None
                and request.records[minimum_index].event_time
                < request.records[previous_max_index].event_time
            ):
                event_reversals += 1
                report(
                    key,
                    "adjacent_offset_event_reversal",
                    request.require_event_time_order,
                    request.records[previous_max_index].offset,
                    offset,
                    2,
                    [previous_max_index, minimum_index],
                )
            previous_max_index = maximum_index
            if interval is None:
                continue
            if not interval.start_offset <= offset <= interval.end_offset:
                out_of_range_rows += len(offset_indexes)
                report(
                    key,
                    "offset_outside_expected_range",
                    True,
                    offset,
                    offset,
                    len(offset_indexes),
                    offset_indexes,
                )
                continue
            observed_expected += 1
            if offset > next_expected:
                missing_count += offset - next_expected
                missing_ranges += 1
                report(
                    key,
                    "missing_offset_range",
                    True,
                    next_expected,
                    offset - 1,
                    offset - next_expected,
                    [],
                )
            next_expected = offset + 1
        if interval is not None and next_expected <= interval.end_offset:
            gap_size = interval.end_offset - next_expected + 1
            missing_count += gap_size
            missing_ranges += 1
            report(
                key,
                "missing_offset_range",
                True,
                next_expected,
                interval.end_offset,
                gap_size,
                [],
            )
        if not request.offset <= stream_index < request.offset + request.limit:
            continue
        streams.append(
            StreamSummary(
                source_id=key[0],
                stream_id=key[1],
                partition_id=key[2],
                expected_range_row_index=range_index,
                expected_start_offset=interval.start_offset if interval else None,
                expected_end_offset=interval.end_offset if interval else None,
                expected_offset_count=(
                    interval.end_offset - interval.start_offset + 1 if interval else None
                ),
                record_count=len(indexes),
                distinct_offset_count=len(offsets),
                observed_min_offset=ordered_offsets[0] if ordered_offsets else None,
                observed_max_offset=ordered_offsets[-1] if ordered_offsets else None,
                observed_expected_offset_count=observed_expected if interval else None,
                missing_offset_count=missing_count if interval else None,
                missing_range_count=missing_ranges if interval else None,
                out_of_range_record_count=out_of_range_rows if interval else None,
                duplicate_row_count=duplicate_rows,
                exact_duplicate_row_count=exact_duplicate_rows,
                payload_conflict_offset_count=payload_conflicts,
                event_clock_conflict_offset_count=clock_conflicts,
                input_offset_reversal_count=input_reversals,
                adjacent_offset_event_reversal_count=event_reversals,
                coverage_complete=missing_count == 0 if interval else None,
                passed=violation_count == before_violations,
            )
        )
    page_end = min(len(keys), request.offset + request.limit)
    diagnostic_count = sum(issues.values())
    return Output(
        passed=violation_count == 0,
        stream_count=len(keys),
        record_count=len(request.records),
        expected_range_count=len(request.expected_ranges),
        duplicate_policy=request.duplicate_policy,
        require_input_offset_order=request.require_input_offset_order,
        require_event_time_order=request.require_event_time_order,
        issue_counts=dict(sorted(issues.items())),
        violation_count=violation_count,
        diagnostic_count=diagnostic_count,
        diagnostics_truncated=diagnostic_count > len(diagnostics),
        offset=request.offset,
        next_offset=page_end if page_end < len(keys) else None,
        streams=streams,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_stream_offsets",
    kind="skill",
    description=(
        "Audit source/stream/partition offset coverage using inclusive expected ranges, "
        "compressed gaps, supplied hash conflicts, duplicate and explicit ordering policies."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
