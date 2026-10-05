"""Audit explicitly supplied nested folds and parent/child selection boundaries.

Outer folds contain train/test memberships; declared inner folds belong to one
outer fold and contain train/validation memberships. Sample IDs always separate
the two partitions at each level. Group separation is independently configurable
for the outer and inner boundaries. Outer test samples can never enter an inner
fold; with outer group separation enabled, their groups cannot enter either.

Inner coverage is measured against known outer training samples excluding outer
test samples. This avoids requiring a child to include an already-invalid parent
overlap. Outer coverage is measured against the entire supplied sample catalog.
Missing assignments, empty partitions, missing inner folds, and duplicate rows
have explicit policies. Reuse across different outer folds or different inner
folds is allowed; this audit does not construct folds or check temporal leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Unit = tuple[str, str | None]


class Sample(InputModel):
    sample_id: Name
    group_id: Name


class InnerFold(InputModel):
    outer_fold_id: Name
    inner_fold_id: Name


class OuterMembership(InputModel):
    outer_fold_id: Name
    sample_id: Name
    partition: Literal["train", "test"]


class InnerMembership(InputModel):
    outer_fold_id: Name
    inner_fold_id: Name
    sample_id: Name
    partition: Literal["train", "validation"]


class Input(InputModel):
    samples: list[Sample] = Field(max_length=10_000)
    outer_fold_ids: list[Name] = Field(min_length=1, max_length=128)
    inner_folds: list[InnerFold] = Field(max_length=1_024)
    outer_memberships: list[OuterMembership] = Field(max_length=10_000)
    inner_memberships: list[InnerMembership] = Field(max_length=10_000)
    duplicate_membership_policy: Literal["reject", "allow"] = "reject"
    outer_unassigned_policy: Literal["allow", "reject"] = "allow"
    inner_unassigned_policy: Literal["allow", "reject"] = "allow"
    require_outer_group_disjoint: bool = Field(default=True, strict=True)
    require_inner_group_disjoint: bool = Field(default=True, strict=True)
    require_nonempty_partitions: bool = Field(default=True, strict=True)
    require_inner_folds: bool = Field(default=True, strict=True)
    max_coverage_checks: int = Field(default=1_000_000, strict=True, ge=0, le=5_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=1_152)
    limit: int = Field(default=100, strict=True, ge=1, le=500)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indexes: int = Field(default=5, strict=True, ge=2, le=10)

    @model_validator(mode="after")
    def unique_definitions(self) -> Self:
        if len({row.sample_id for row in self.samples}) != len(self.samples):
            raise ValueError("the catalog must have one group assignment per sample_id")
        outer = set(self.outer_fold_ids)
        if len(outer) != len(self.outer_fold_ids):
            raise ValueError("outer_fold_ids must be distinct")
        inner = {(row.outer_fold_id, row.inner_fold_id) for row in self.inner_folds}
        if len(inner) != len(self.inner_folds):
            raise ValueError("inner fold IDs must be distinct within each outer fold")
        if any(row.outer_fold_id not in outer for row in self.inner_folds):
            raise ValueError("every inner fold definition must reference a declared outer fold")
        return self


@dataclass
class _Partition:
    samples: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    groups: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    def add(self, sample: str, group: str | None, index: int) -> None:
        self.samples[sample].append(index)
        if group is not None:
            self.groups[group].append(index)


@dataclass
class _Fold:
    declaration_index: int
    train: _Partition = field(default_factory=_Partition)
    evaluation: _Partition = field(default_factory=_Partition)


class Finding(OutputModel):
    code: str
    is_violation: bool
    outer_fold_id: str
    inner_fold_id: str | None
    partition: str | None
    sample_id: str | None
    group_id: str | None
    affected_count: int
    catalog_row_indexes: list[int]
    outer_membership_row_indexes: list[int]
    inner_membership_row_indexes: list[int]
    omitted_catalog_row_indexes: int
    omitted_outer_membership_row_indexes: int
    omitted_inner_membership_row_indexes: int


class FoldSummary(OutputModel):
    level: Literal["outer", "inner"]
    declaration_row_index: int
    outer_fold_id: str
    inner_fold_id: str | None
    evaluation_partition: Literal["test", "validation"]
    membership_rows: int
    distinct_referenced_samples: int
    distinct_known_samples: int
    train_known_samples: int
    evaluation_known_samples: int
    expected_coverage_samples: int
    unassigned_expected_samples: int
    diagnostic_count: int
    violation_count: int
    issue_counts: dict[str, int]
    passed: bool


class Output(OutputModel):
    passed: bool
    catalog_sample_count: int
    catalog_group_count: int
    declared_outer_fold_count: int
    declared_inner_fold_count: int
    outer_membership_row_count: int
    inner_membership_row_count: int
    coverage_checks: int
    duplicate_membership_policy: str
    outer_unassigned_policy: str
    inner_unassigned_policy: str
    require_outer_group_disjoint: bool
    require_inner_group_disjoint: bool
    require_nonempty_partitions: bool
    require_inner_folds: bool
    inner_coverage_scope: Literal["known_outer_train_excluding_outer_test"] = (
        "known_outer_train_excluding_outer_test"
    )
    cross_fold_reuse: Literal["allowed"] = "allowed"
    issue_counts: dict[str, int]
    violation_count: int
    diagnostic_count: int
    omitted_diagnostics: int
    offset: int
    next_offset: int | None
    folds: list[FoldSummary]
    diagnostics: list[Finding]


def execute(request: Input, context: OperationContext) -> Output:
    catalog = {row.sample_id: (row.group_id, index) for index, row in enumerate(request.samples)}
    catalog_ids = set(catalog)
    outer = {fold_id: _Fold(index) for index, fold_id in enumerate(request.outer_fold_ids)}
    inner = {
        (row.outer_fold_id, row.inner_fold_id): _Fold(index)
        for index, row in enumerate(request.inner_folds)
    }
    diagnostics: list[Finding] = []
    issues: Counter[str] = Counter()
    unit_issues: dict[Unit, Counter[str]] = defaultdict(Counter)
    violations: Counter[Unit] = Counter()

    def report(
        unit: Unit,
        code: str,
        is_violation: bool = True,
        affected_count: int = 1,
        partition: str | None = None,
        sample_id: str | None = None,
        group_id: str | None = None,
        catalog_rows: list[int] | None = None,
        outer_rows: list[int] | None = None,
        inner_rows: list[int] | None = None,
    ) -> None:
        issues[code] += 1
        unit_issues[unit][code] += 1
        violations[unit] += is_violation
        if len(diagnostics) >= request.max_diagnostics:
            return
        catalog_rows, outer_rows, inner_rows = (
            catalog_rows or [],
            outer_rows or [],
            inner_rows or [],
        )
        bound = request.max_row_indexes
        diagnostics.append(
            Finding(
                code=code,
                is_violation=is_violation,
                outer_fold_id=unit[0],
                inner_fold_id=unit[1],
                partition=partition,
                sample_id=sample_id,
                group_id=group_id,
                affected_count=affected_count,
                catalog_row_indexes=catalog_rows[:bound],
                outer_membership_row_indexes=outer_rows[:bound],
                inner_membership_row_indexes=inner_rows[:bound],
                omitted_catalog_row_indexes=max(0, len(catalog_rows) - bound),
                omitted_outer_membership_row_indexes=max(0, len(outer_rows) - bound),
                omitted_inner_membership_row_indexes=max(0, len(inner_rows) - bound),
            )
        )

    for index, membership in enumerate(request.outer_memberships):
        unit: Unit = membership.outer_fold_id, None
        fold = outer.get(membership.outer_fold_id)
        if fold is None:
            report(unit, "unknown_outer_fold", outer_rows=[index])
            continue
        sample = catalog.get(membership.sample_id)
        bucket = fold.train if membership.partition == "train" else fold.evaluation
        bucket.add(membership.sample_id, sample[0] if sample else None, index)
        if sample is None:
            report(unit, "unknown_sample", sample_id=membership.sample_id, outer_rows=[index])
    for index, inner_membership in enumerate(request.inner_memberships):
        inner_key = inner_membership.outer_fold_id, inner_membership.inner_fold_id
        child = inner.get(inner_key)
        if child is None:
            report(inner_key, "unknown_inner_fold", inner_rows=[index])
            continue
        sample = catalog.get(inner_membership.sample_id)
        bucket = child.train if inner_membership.partition == "train" else child.evaluation
        bucket.add(inner_membership.sample_id, sample[0] if sample else None, index)
        if sample is None:
            report(
                inner_key,
                "unknown_sample",
                sample_id=inner_membership.sample_id,
                inner_rows=[index],
            )

    parent_training = {
        fold_id: (set(fold.train.samples) & catalog_ids) - set(fold.evaluation.samples)
        for fold_id, fold in outer.items()
    }
    parent_test_groups = {fold_id: set(fold.evaluation.groups) for fold_id, fold in outer.items()}
    coverage_checks = len(catalog) * len(outer) + sum(len(parent_training[key[0]]) for key in inner)
    if coverage_checks > request.max_coverage_checks:
        raise ValueError("catalog/parent-training coverage scans exceed max_coverage_checks")
    children_by_parent: Counter[str] = Counter(key[0] for key in inner)
    for fold_id in request.outer_fold_ids:
        if not children_by_parent[fold_id]:
            report((fold_id, None), "no_inner_folds", request.require_inner_folds)

    children: dict[str, list[tuple[Unit, _Fold]]] = defaultdict(list)
    for key, child in inner.items():
        children[key[0]].append((key, child))
    fold_units: list[tuple[Unit, _Fold]] = []
    for fold_id in request.outer_fold_ids:
        fold_units.append(((fold_id, None), outer[fold_id]))
        fold_units.extend(children[fold_id])
    summaries: list[FoldSummary] = []
    for position, (unit, fold) in enumerate(fold_units):
        is_inner = unit[1] is not None
        evaluation_name = "validation" if is_inner else "test"
        memberships = set(fold.train.samples) | set(fold.evaluation.samples)
        known_memberships = memberships & catalog_ids
        for partition, bucket in (("train", fold.train), (evaluation_name, fold.evaluation)):
            if not set(bucket.samples) & catalog_ids:
                report(
                    unit,
                    "empty_known_partition",
                    request.require_nonempty_partitions,
                    affected_count=0,
                    partition=partition,
                )
            for sample_id, rows in sorted(bucket.samples.items()):
                if len(rows) > 1:
                    report(
                        unit,
                        "duplicate_membership",
                        request.duplicate_membership_policy == "reject",
                        affected_count=len(rows) - 1,
                        partition=partition,
                        sample_id=sample_id,
                        inner_rows=rows if is_inner else None,
                        outer_rows=None if is_inner else rows,
                    )
        for sample_id in sorted(set(fold.train.samples) & set(fold.evaluation.samples)):
            # Keep one concrete witness from each side even when a side has many duplicate rows.
            rows = [fold.train.samples[sample_id][0], fold.evaluation.samples[sample_id][0]]
            report(
                unit,
                "sample_across_partitions",
                sample_id=sample_id,
                outer_rows=None if is_inner else rows,
                inner_rows=rows if is_inner else None,
            )
        require_groups = (
            request.require_inner_group_disjoint
            if is_inner
            else request.require_outer_group_disjoint
        )
        for group_id in sorted(set(fold.train.groups) & set(fold.evaluation.groups)):
            rows = [fold.train.groups[group_id][0], fold.evaluation.groups[group_id][0]]
            report(
                unit,
                "group_across_partitions",
                require_groups,
                group_id=group_id,
                outer_rows=None if is_inner else rows,
                inner_rows=rows if is_inner else None,
            )

        if is_inner:
            parent = outer[unit[0]]
            for sample_id in sorted(memberships):
                rows = fold.train.samples.get(sample_id, []) + fold.evaluation.samples.get(
                    sample_id, []
                )
                if sample_id not in parent.train.samples or sample_id not in catalog:
                    report(
                        unit,
                        "inner_sample_not_in_outer_training",
                        sample_id=sample_id,
                        affected_count=len(rows),
                        inner_rows=rows,
                    )
                if sample_id in parent.evaluation.samples:
                    report(
                        unit,
                        "outer_test_sample_in_inner",
                        sample_id=sample_id,
                        affected_count=len(rows),
                        inner_rows=rows,
                        outer_rows=parent.evaluation.samples[sample_id],
                    )
            child_groups = set(fold.train.groups) | set(fold.evaluation.groups)
            for group_id in sorted(child_groups & parent_test_groups[unit[0]]):
                rows = fold.train.groups.get(group_id, []) + fold.evaluation.groups.get(
                    group_id, []
                )
                report(
                    unit,
                    "outer_test_group_in_inner",
                    request.require_outer_group_disjoint,
                    group_id=group_id,
                    affected_count=len(rows),
                    inner_rows=rows,
                    outer_rows=parent.evaluation.groups[group_id],
                )
            expected_samples = parent_training[unit[0]]
            reject_unassigned = request.inner_unassigned_policy == "reject"
        else:
            expected_samples = catalog_ids
            reject_unassigned = request.outer_unassigned_policy == "reject"
        unassigned = expected_samples - known_memberships
        if unassigned:
            report(
                unit,
                "unassigned_expected_samples",
                reject_unassigned,
                affected_count=len(unassigned),
                catalog_rows=[catalog[name][1] for name in sorted(unassigned)],
            )
        if not request.offset <= position < request.offset + request.limit:
            continue
        summaries.append(
            FoldSummary(
                level="inner" if is_inner else "outer",
                declaration_row_index=fold.declaration_index,
                outer_fold_id=unit[0],
                inner_fold_id=unit[1],
                evaluation_partition="validation" if is_inner else "test",
                membership_rows=sum(
                    len(rows)
                    for bucket in (fold.train, fold.evaluation)
                    for rows in bucket.samples.values()
                ),
                distinct_referenced_samples=len(memberships),
                distinct_known_samples=len(known_memberships),
                train_known_samples=len(set(fold.train.samples) & catalog_ids),
                evaluation_known_samples=len(set(fold.evaluation.samples) & catalog_ids),
                expected_coverage_samples=len(expected_samples),
                unassigned_expected_samples=len(unassigned),
                diagnostic_count=sum(unit_issues[unit].values()),
                violation_count=violations[unit],
                issue_counts=dict(sorted(unit_issues[unit].items())),
                passed=violations[unit] == 0,
            )
        )
    diagnostic_count = sum(issues.values())
    page_end = min(len(fold_units), request.offset + request.limit)
    return Output(
        passed=sum(violations.values()) == 0,
        catalog_sample_count=len(catalog),
        catalog_group_count=len({row.group_id for row in request.samples}),
        declared_outer_fold_count=len(outer),
        declared_inner_fold_count=len(inner),
        outer_membership_row_count=len(request.outer_memberships),
        inner_membership_row_count=len(request.inner_memberships),
        coverage_checks=coverage_checks,
        duplicate_membership_policy=request.duplicate_membership_policy,
        outer_unassigned_policy=request.outer_unassigned_policy,
        inner_unassigned_policy=request.inner_unassigned_policy,
        require_outer_group_disjoint=request.require_outer_group_disjoint,
        require_inner_group_disjoint=request.require_inner_group_disjoint,
        require_nonempty_partitions=request.require_nonempty_partitions,
        require_inner_folds=request.require_inner_folds,
        issue_counts=dict(sorted(issues.items())),
        violation_count=sum(violations.values()),
        diagnostic_count=diagnostic_count,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
        offset=request.offset,
        next_offset=page_end if page_end < len(fold_units) else None,
        folds=summaries,
        diagnostics=diagnostics,
    )


OPERATION = Operation(
    id="skills.audit_nested_folds",
    kind="skill",
    description=(
        "Audit explicit nested train/test and train/validation memberships for parent "
        "containment, held-out leakage, group separation, duplicates, and coverage."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
