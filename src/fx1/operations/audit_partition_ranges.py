"""Audit integer partition geometry against explicit half-open expected coverage.

Coverage identity is (dataset_id, version_id, key). Partition IDs must be unique
within a dataset/version, including across keys. Empty or inverted partition
ranges are diagnosed and excluded from geometry; duplicate IDs remain in the
geometry so overlapping supplied rows are still counted. Expected ranges may be
empty. Every observed identity requires an explicitly supplied expected range.

Coverage and excess sizes are measures of interval unions, not sums of partition
lengths. Overlap counts are exact unordered row-pair counts with one source-pair
witness per row having prior overlaps. Gaps are maximal uncovered expected spans.
No integer universe or Cartesian list of overlapping pairs is materialized.
Ranges are caller declarations and do not authenticate stored bytes or records.
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Endpoint = Annotated[int, Field(strict=True, ge=-(2**63), le=2**63 - 1)]
CoverageKey = tuple[str, str, str]


class Identity(InputModel):
    dataset_id: Name
    version_id: Name
    key: Name

    def identity(self) -> CoverageKey:
        return self.dataset_id, self.version_id, self.key


class ExpectedCoverage(Identity):
    start: Endpoint
    end: Endpoint

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end < self.start:
            raise ValueError("expected coverage end must be at least start")
        return self


class Partition(Identity):
    partition_id: Name
    start: Endpoint
    end: Endpoint


class Input(InputModel):
    expected_coverage: list[ExpectedCoverage] = Field(min_length=1, max_length=2_048)
    partitions: list[Partition] = Field(max_length=10_000)
    offset: int = Field(default=0, strict=True, ge=0, le=2_048)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=2, le=10)

    @model_validator(mode="after")
    def unique_bounded_coverage(self) -> Self:
        identities = {row.identity() for row in self.expected_coverage}
        if len(identities) != len(self.expected_coverage):
            raise ValueError("each dataset/version/key must have one expected coverage range")
        identities.update(row.identity() for row in self.partitions)
        if len(identities) > 2_048:
            raise ValueError("at most 2048 dataset/version/key identities are supported")
        return self


class Diagnostic(OutputModel):
    code: str
    dataset_id: str
    version_id: str
    key: str | None
    partition_id: str | None = None
    expected_coverage_row_index: int | None = None
    start: int | None = None
    end: int | None = None
    affected_count: int
    partition_row_indexes: list[int]
    omitted_partition_row_indexes: int


class CoverageSummary(OutputModel):
    dataset_id: str
    version_id: str
    key: str
    expected_coverage_row_index: int | None
    expected_start: int | None
    expected_end: int | None
    expected_integer_count: int | None
    partition_row_count: int
    valid_range_count: int
    invalid_range_count: int
    reused_partition_id_count: int
    observed_union_integer_count: int
    covered_expected_integer_count: int | None
    missing_expected_integer_count: int | None
    excess_union_integer_count: int | None
    missing_range_count: int | None
    outside_expected_range_rows: int | None
    overlap_pair_count: int
    maximum_concurrent_ranges: int
    coverage_complete: bool | None
    passed: bool


class Output(OutputModel):
    passed: bool
    partition_row_count: int
    expected_coverage_count: int
    coverage_identity_count: int
    reused_partition_id_count: int
    overlap_pair_count: int
    missing_expected_integer_count: int
    issue_counts: dict[str, int]
    diagnostic_count: int
    omitted_diagnostics: int
    range_policy: Literal["half_open_integer_ranges"] = "half_open_integer_ranges"
    partition_id_scope: Literal["dataset_and_version_across_keys"] = (
        "dataset_and_version_across_keys"
    )
    evidence_scope: Literal["declared_ranges_not_byte_or_record_authenticity"] = (
        "declared_ranges_not_byte_or_record_authenticity"
    )
    offset: int
    next_offset: int | None
    coverage: list[CoverageSummary]
    diagnostics: list[Diagnostic]


def execute(request: Input, context: OperationContext) -> Output:
    expected = {row.identity(): index for index, row in enumerate(request.expected_coverage)}
    grouped: dict[CoverageKey, list[int]] = defaultdict(list)
    partition_ids: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(request.partitions):
        grouped[row.identity()].append(index)
        partition_ids[(row.dataset_id, row.version_id, row.partition_id)].append(index)
    diagnostics: list[Diagnostic] = []
    issues: Counter[str] = Counter()
    bad_identities: set[CoverageKey] = set()
    duplicate_counts: Counter[CoverageKey] = Counter()

    def report(
        identity: CoverageKey,
        code: str,
        indexes: list[int],
        count: int,
        start: int | None = None,
        end: int | None = None,
        partition_id: str | None = None,
        multiple_keys: bool = False,
    ) -> None:
        issues[code] += 1
        bad_identities.add(identity)
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(
                Diagnostic(
                    code=code,
                    dataset_id=identity[0],
                    version_id=identity[1],
                    key=None if multiple_keys else identity[2],
                    partition_id=partition_id,
                    expected_coverage_row_index=None if multiple_keys else expected.get(identity),
                    start=start,
                    end=end,
                    affected_count=count,
                    partition_row_indexes=indexes[: request.max_row_indexes],
                    omitted_partition_row_indexes=max(0, len(indexes) - request.max_row_indexes),
                )
            )

    reused_ids = 0
    for (_, _, partition_id), indexes in sorted(partition_ids.items()):
        if len(indexes) <= 1:
            continue
        reused_ids += 1
        identities = {request.partitions[index].identity() for index in indexes}
        for identity in identities:
            duplicate_counts[identity] += 1
        bad_identities.update(identities)
        report(
            request.partitions[indexes[0]].identity(),
            "reused_partition_id",
            indexes,
            len(indexes),
            partition_id=partition_id,
            multiple_keys=len(identities) > 1,
        )

    keys = sorted(set(expected) | set(grouped))
    summaries: list[CoverageSummary] = []
    total_overlap_pairs = 0
    total_missing = 0
    for position, identity in enumerate(keys):
        indexes = grouped.get(identity, [])
        expected_index = expected.get(identity)
        coverage = request.expected_coverage[expected_index] if expected_index is not None else None
        if coverage is None:
            report(identity, "missing_expected_coverage", indexes, len(indexes))
        valid: list[int] = []
        invalid_count = 0
        for index in indexes:
            row = request.partitions[index]
            if row.end <= row.start:
                invalid_count += 1
                report(
                    identity,
                    "empty_range" if row.end == row.start else "inverted_range",
                    [index],
                    1,
                    row.start,
                    row.end,
                )
            else:
                valid.append(index)
        valid.sort(key=lambda index: (request.partitions[index].start, index))
        active: list[tuple[int, int]] = []
        overlap_pairs = 0
        maximum_concurrent = 0
        union_end: int | None = None
        union_size = 0
        covered_size = 0
        missing_ranges = 0
        outside_rows = 0
        cursor = coverage.start if coverage else 0
        previous_coverage_index: int | None = None
        for index in valid:
            row = request.partitions[index]
            while active and active[0][0] <= row.start:
                heapq.heappop(active)
            if active:
                overlap_pairs += len(active)
                witness_end, witness_index = active[0]
                report(
                    identity,
                    "overlap",
                    [witness_index, index],
                    len(active),
                    row.start,
                    min(row.end, witness_end),
                )
            heapq.heappush(active, (row.end, index))
            maximum_concurrent = max(maximum_concurrent, len(active))
            union_size += (
                row.end - row.start
                if union_end is None
                else max(0, row.end - max(row.start, union_end))
            )
            union_end = row.end if union_end is None else max(union_end, row.end)
            if coverage is None:
                continue
            if row.start < coverage.start or row.end > coverage.end:
                outside_rows += 1
                if row.start < coverage.start:
                    end = min(row.end, coverage.start)
                    report(
                        identity,
                        "outside_expected_coverage",
                        [index],
                        end - row.start,
                        row.start,
                        end,
                    )
                if row.end > coverage.end:
                    start = max(row.start, coverage.end)
                    report(
                        identity,
                        "outside_expected_coverage",
                        [index],
                        row.end - start,
                        start,
                        row.end,
                    )
            clipped_start, clipped_end = max(row.start, coverage.start), min(row.end, coverage.end)
            if clipped_end <= clipped_start:
                continue
            if clipped_start > cursor:
                missing_ranges += 1
                witnesses = (
                    [index] if previous_coverage_index is None else [previous_coverage_index, index]
                )
                report(
                    identity,
                    "missing_expected_range",
                    witnesses,
                    clipped_start - cursor,
                    cursor,
                    clipped_start,
                )
            covered_size += max(0, clipped_end - max(clipped_start, cursor))
            if clipped_end > cursor:
                cursor = clipped_end
                previous_coverage_index = index
        if coverage is not None and cursor < coverage.end:
            missing_ranges += 1
            report(
                identity,
                "missing_expected_range",
                [] if previous_coverage_index is None else [previous_coverage_index],
                coverage.end - cursor,
                cursor,
                coverage.end,
            )
        expected_size = coverage.end - coverage.start if coverage else None
        missing_size = expected_size - covered_size if expected_size is not None else None
        total_overlap_pairs += overlap_pairs
        total_missing += missing_size or 0
        if not request.offset <= position < request.offset + request.limit:
            continue
        summaries.append(
            CoverageSummary(
                dataset_id=identity[0],
                version_id=identity[1],
                key=identity[2],
                expected_coverage_row_index=expected_index,
                expected_start=coverage.start if coverage else None,
                expected_end=coverage.end if coverage else None,
                expected_integer_count=expected_size,
                partition_row_count=len(indexes),
                valid_range_count=len(valid),
                invalid_range_count=invalid_count,
                reused_partition_id_count=duplicate_counts[identity],
                observed_union_integer_count=union_size,
                covered_expected_integer_count=covered_size if coverage else None,
                missing_expected_integer_count=missing_size,
                excess_union_integer_count=union_size - covered_size if coverage else None,
                missing_range_count=missing_ranges if coverage else None,
                outside_expected_range_rows=outside_rows if coverage else None,
                overlap_pair_count=overlap_pairs,
                maximum_concurrent_ranges=maximum_concurrent,
                coverage_complete=missing_size == 0 if coverage else None,
                passed=identity not in bad_identities,
            )
        )
    diagnostic_count = sum(issues.values())
    page_end = min(len(keys), request.offset + request.limit)
    return Output(
        passed=not bad_identities,
        partition_row_count=len(request.partitions),
        expected_coverage_count=len(request.expected_coverage),
        coverage_identity_count=len(keys),
        reused_partition_id_count=reused_ids,
        overlap_pair_count=total_overlap_pairs,
        missing_expected_integer_count=total_missing,
        issue_counts=dict(sorted(issues.items())),
        diagnostic_count=diagnostic_count,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
        offset=request.offset,
        next_offset=page_end if page_end < len(keys) else None,
        coverage=summaries,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_partition_ranges",
    kind="skill",
    description=(
        "Audit declared half-open integer partition coverage by dataset/version/key, "
        "with exact union sizes, compressed gaps, overlap witnesses, and reused IDs."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
