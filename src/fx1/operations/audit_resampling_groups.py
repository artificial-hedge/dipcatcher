"""Audit caller-supplied resample draws against explicit partition constraints.

Each sample belongs to exactly one group in a unique sample catalog. Resamples
are independent: reuse across different resample_id values is permitted. Within
a resample, supplied disjoint partition pairs prohibit both the same sample and
different samples of the same group from appearing on opposite sides.

draw_count is a compressed count of repeated inclusions; repeated membership
rows add to it. Duplicate policy applies within (resample, sample, partition),
independently of cross-partition integrity. With allow it describes bootstrap
repetition without treating that repetition as a violation. Undeclared partition
pairs have no implied constraint. No resample is constructed, and no temporal
purge, independence guarantee, statistical significance, or market claim is made.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
UnitKey = tuple[str, str]


class Sample(InputModel):
    sample_id: Name
    group_id: Name


class Membership(InputModel):
    resample_id: Name
    sample_id: Name
    partition: Name
    draw_count: int = Field(default=1, strict=True, ge=1, le=1_000_000)


class DisjointPair(InputModel):
    first: Name
    second: Name

    @model_validator(mode="after")
    def different_partitions(self) -> Self:
        if self.first == self.second:
            raise ValueError("a disjoint pair must name two different partitions")
        return self


class Input(InputModel):
    samples: list[Sample] = Field(max_length=10_000)
    memberships: list[Membership] = Field(max_length=10_000)
    disjoint_pairs: list[DisjointPair] = Field(min_length=1, max_length=128)
    duplicate_inclusion_policy: Literal["allow", "reject"] = "reject"
    max_partition_comparisons: int = Field(default=250_000, strict=True, ge=0, le=1_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indices: int = Field(default=5, strict=True, ge=1, le=10)

    @model_validator(mode="after")
    def unique_catalog_and_pairs(self) -> Self:
        if len({row.sample_id for row in self.samples}) != len(self.samples):
            raise ValueError(
                "sample catalog must contain exactly one group assignment per sample_id"
            )
        pairs = {tuple(sorted((pair.first, pair.second))) for pair in self.disjoint_pairs}
        if len(pairs) != len(self.disjoint_pairs):
            raise ValueError("disjoint_pairs must not repeat an unordered partition pair")
        return self


class Finding(OutputModel):
    code: Literal[
        "unknown_sample",
        "duplicate_inclusion",
        "sample_across_disjoint_partitions",
        "group_across_disjoint_partitions",
    ]
    is_violation: bool
    resample_id: str
    sample_id: str | None = None
    group_id: str | None = None
    first_partition: str
    second_partition: str | None = None
    first_membership_indices: list[int]
    second_membership_indices: list[int] = Field(default_factory=list)
    omitted_first_indices: int = 0
    omitted_second_indices: int = 0
    draw_count: int | None = None


class ResampleSummary(OutputModel):
    resample_id: str
    membership_rows: int
    draw_count: int
    distinct_referenced_samples: int
    distinct_known_groups: int
    partition_count: int
    unknown_sample_rows: int
    duplicate_inclusion_groups: int
    sample_partition_conflicts: int
    group_partition_conflicts: int
    passed: bool


class Output(OutputModel):
    catalog_samples: int
    catalog_groups: int
    unreferenced_catalog_samples: int
    membership_rows: int
    draw_count: int
    resample_count: int
    unknown_sample_rows: int
    unknown_sample_draws: int
    duplicate_inclusion_groups: int
    excess_draws_within_partition: int
    sample_partition_conflicts: int
    group_partition_conflicts: int
    partition_comparisons: int
    disjoint_pair_count: int
    duplicate_inclusion_policy: str
    cross_resample_reuse_allowed: bool = True
    passed: bool
    resamples: list[ResampleSummary]
    next_offset: int | None
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def execute(request: Input, context: OperationContext) -> Output:
    catalog = {row.sample_id: row.group_id for row in request.samples}
    by_sample: dict[UnitKey, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_group: dict[UnitKey, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    resample_rows: dict[str, list[int]] = defaultdict(list)
    sample_sets: dict[str, set[str]] = defaultdict(set)
    group_sets: dict[str, set[str]] = defaultdict(set)
    partition_sets: dict[str, set[str]] = defaultdict(set)
    draws: dict[tuple[str, str, str], int] = defaultdict(int)
    resample_draws: dict[str, int] = defaultdict(int)
    unknown_counts: dict[str, int] = defaultdict(int)
    duplicate_counts: dict[str, int] = defaultdict(int)
    sample_conflict_counts: dict[str, int] = defaultdict(int)
    group_conflict_counts: dict[str, int] = defaultdict(int)
    referenced_known: set[str] = set()
    unknown_draws = 0
    diagnostics: list[Finding] = []
    diagnostic_count = 0

    def report(finding: Finding) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    for index, row in enumerate(request.memberships):
        resample_rows[row.resample_id].append(index)
        sample_sets[row.resample_id].add(row.sample_id)
        partition_sets[row.resample_id].add(row.partition)
        by_sample[(row.resample_id, row.sample_id)][row.partition].append(index)
        draws[(row.resample_id, row.sample_id, row.partition)] += row.draw_count
        resample_draws[row.resample_id] += row.draw_count
        if row.sample_id not in catalog:
            unknown_counts[row.resample_id] += 1
            unknown_draws += row.draw_count
            report(
                Finding(
                    code="unknown_sample",
                    is_violation=True,
                    resample_id=row.resample_id,
                    sample_id=row.sample_id,
                    first_partition=row.partition,
                    first_membership_indices=[index],
                    draw_count=row.draw_count,
                )
            )
            continue
        group = catalog[row.sample_id]
        referenced_known.add(row.sample_id)
        group_sets[row.resample_id].add(group)
        by_group[(row.resample_id, group)][row.partition].append(index)

    # Each declared edge is visited once from its lexically smaller endpoint.
    # Budget both sample-level and group-level adjacency probes before scanning.
    adjacency: dict[str, list[str]] = defaultdict(list)
    for pair in request.disjoint_pairs:
        first, second = sorted((pair.first, pair.second))
        adjacency[first].append(second)
    for neighbors in adjacency.values():
        neighbors.sort()
    comparisons = sum(
        len(adjacency.get(partition, []))
        for units in (by_sample, by_group)
        for partitions in units.values()
        for partition in partitions
    )
    if comparisons > request.max_partition_comparisons:
        raise ValueError(
            f"partition constraints require {comparisons} probes, exceeding "
            f"max_partition_comparisons={request.max_partition_comparisons}; "
            "partition the request by complete resample_id values"
        )

    excess_draws = 0
    for (resample, sample), partitions in sorted(by_sample.items()):
        for partition, indices in sorted(partitions.items()):
            count = draws[(resample, sample, partition)]
            if count > 1:
                duplicate_counts[resample] += 1
                excess_draws += count - 1
                report(
                    Finding(
                        code="duplicate_inclusion",
                        is_violation=request.duplicate_inclusion_policy == "reject",
                        resample_id=resample,
                        sample_id=sample,
                        group_id=catalog.get(sample),
                        first_partition=partition,
                        first_membership_indices=indices[: request.max_row_indices],
                        omitted_first_indices=max(0, len(indices) - request.max_row_indices),
                        draw_count=count,
                    )
                )

    for group_level, units in ((False, by_sample), (True, by_group)):
        for (resample, identifier), partitions in sorted(units.items()):
            for first, first_indices in sorted(partitions.items()):
                for second in adjacency.get(first, []):
                    if second not in partitions:
                        continue
                    second_indices = partitions[second]
                    if group_level:
                        group_conflict_counts[resample] += 1
                    else:
                        sample_conflict_counts[resample] += 1
                    report(
                        Finding(
                            code=(
                                "group_across_disjoint_partitions"
                                if group_level
                                else "sample_across_disjoint_partitions"
                            ),
                            is_violation=True,
                            resample_id=resample,
                            sample_id=None if group_level else identifier,
                            group_id=identifier if group_level else catalog.get(identifier),
                            first_partition=first,
                            second_partition=second,
                            first_membership_indices=first_indices[: request.max_row_indices],
                            second_membership_indices=second_indices[: request.max_row_indices],
                            omitted_first_indices=max(
                                0, len(first_indices) - request.max_row_indices
                            ),
                            omitted_second_indices=max(
                                0, len(second_indices) - request.max_row_indices
                            ),
                        )
                    )

    def passes(resample: str) -> bool:
        return not (
            unknown_counts[resample]
            or sample_conflict_counts[resample]
            or group_conflict_counts[resample]
            or (request.duplicate_inclusion_policy == "reject" and duplicate_counts[resample])
        )

    resample_ids = sorted(resample_rows)
    summaries = [
        ResampleSummary(
            resample_id=resample,
            membership_rows=len(resample_rows[resample]),
            draw_count=resample_draws[resample],
            distinct_referenced_samples=len(sample_sets[resample]),
            distinct_known_groups=len(group_sets[resample]),
            partition_count=len(partition_sets[resample]),
            unknown_sample_rows=unknown_counts[resample],
            duplicate_inclusion_groups=duplicate_counts[resample],
            sample_partition_conflicts=sample_conflict_counts[resample],
            group_partition_conflicts=group_conflict_counts[resample],
            passed=passes(resample),
        )
        for resample in resample_ids[request.offset : request.offset + request.limit]
    ]
    next_offset = request.offset + len(summaries)
    return Output(
        catalog_samples=len(catalog),
        catalog_groups=len(set(catalog.values())),
        unreferenced_catalog_samples=len(catalog) - len(referenced_known),
        membership_rows=len(request.memberships),
        draw_count=sum(resample_draws.values()),
        resample_count=len(resample_ids),
        unknown_sample_rows=sum(unknown_counts.values()),
        unknown_sample_draws=unknown_draws,
        duplicate_inclusion_groups=sum(duplicate_counts.values()),
        excess_draws_within_partition=excess_draws,
        sample_partition_conflicts=sum(sample_conflict_counts.values()),
        group_partition_conflicts=sum(group_conflict_counts.values()),
        partition_comparisons=comparisons,
        disjoint_pair_count=len(request.disjoint_pairs),
        duplicate_inclusion_policy=request.duplicate_inclusion_policy,
        passed=all(passes(resample) for resample in resample_ids),
        resamples=summaries,
        next_offset=next_offset if next_offset < len(resample_ids) else None,
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_resampling_groups",
    kind="skill",
    description=(
        "Audit supplied resample draws for sample and group separation under explicit disjoint "
        "partition pairs, unknown samples, and caller-declared repeated-inclusion semantics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
