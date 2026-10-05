"""Audit supplied split memberships and information intervals without constructing splits.

Observation and label intervals are separate half-open ranges; their intervening
gap is not assumed informative. Cross-split overlaps compare only within group_id.
Sample IDs are global. Optional pre-purge/post-embargo spans surround each
validation/test sample's complete information extent and block train intervals.
The explicit work budget counts cross-split sample pairs, not emitted findings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Split = Literal["train", "validation", "test"]


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


class Sample(InputModel):
    sample_id: Name
    group_id: Name = "global"
    split: Split
    observation_start: AwareDatetime
    observation_end: AwareDatetime
    label_start: AwareDatetime
    label_end: AwareDatetime

    @field_validator(
        "observation_start", "observation_end", "label_start", "label_end", mode="before"
    )
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def nonempty_intervals(self) -> Self:
        if self.observation_end <= self.observation_start or self.label_end <= self.label_start:
            raise ValueError("observation and label intervals must each be nonempty and increasing")
        return self


class Input(InputModel):
    samples: list[Sample] = Field(min_length=1, max_length=10_000)
    purge_before_seconds: int = Field(default=0, strict=True, ge=0, le=31_536_000)
    embargo_after_seconds: int = Field(default=0, strict=True, ge=0, le=31_536_000)
    pair_work_budget: int = Field(default=250_000, strict=True, ge=1, le=1_000_000)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def representable_guard_windows(self) -> Self:
        for row in self.samples:
            if row.split != "train":
                try:
                    min(row.observation_start, row.label_start) - timedelta(
                        seconds=self.purge_before_seconds
                    )
                    max(row.observation_end, row.label_end) + timedelta(
                        seconds=self.embargo_after_seconds
                    )
                except OverflowError as error:
                    raise ValueError(
                        "purge/embargo extends outside the supported UTC datetime range"
                    ) from error
        return self


class DuplicateId(OutputModel):
    sample_id: str
    split_counts: dict[str, int]
    cross_split_pairs: int
    first_row_indices_by_split: dict[str, int]


class PairFinding(OutputModel):
    group_id: str
    first_row_index: int
    second_row_index: int
    first_split: Split
    second_split: Split
    information_overlap: bool
    purge_overlap: bool
    embargo_overlap: bool
    first_witness_kind: str
    overlap_start: datetime
    overlap_end: datetime


class Output(OutputModel):
    sample_count: int
    group_count: int
    compared_sample_pairs: int
    duplicate_sample_id_groups: int
    cross_split_duplicate_id_pairs: int
    information_overlap_pairs: int
    purge_overlap_pairs: int
    embargo_overlap_pairs: int
    leaking_interval_pairs: int
    affected_interval_rows: int
    passed: bool
    purge_before_seconds: int
    embargo_after_seconds: int
    interval_policy: str = "half_open_with_separate_observation_and_label_intervals"
    duplicate_id_examples: list[DuplicateId]
    omitted_duplicate_id_groups: int
    pair_examples: list[PairFinding]
    omitted_interval_pairs: int


def _intervals(row: Sample) -> tuple[tuple[str, datetime, datetime], ...]:
    return (
        ("observation", row.observation_start, row.observation_end),
        ("label", row.label_start, row.label_end),
    )


def execute(request: Input, context: OperationContext) -> Output:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    ids: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(request.samples):
        grouped[row.group_id][row.split].append(index)
        ids[row.sample_id][row.split].append(index)
    work = sum(
        len(left) * len(right)
        for splits in grouped.values()
        for left, right in combinations(splits.values(), 2)
    )
    if work > request.pair_work_budget:
        raise ValueError(
            f"{work} cross-split sample pairs exceed pair_work_budget; smaller batches must "
            "retain every required cross-split comparison"
        )
    duplicate_groups = duplicate_pairs = 0
    id_examples: list[DuplicateId] = []
    for sample_id, splits in sorted(ids.items()):
        if len(splits) < 2:
            continue
        duplicate_groups += 1
        pair_count = sum(len(left) * len(right) for left, right in combinations(splits.values(), 2))
        duplicate_pairs += pair_count
        if len(id_examples) < request.max_diagnostics:
            id_examples.append(
                DuplicateId(
                    sample_id=sample_id,
                    split_counts={split: len(indices) for split, indices in sorted(splits.items())},
                    cross_split_pairs=pair_count,
                    first_row_indices_by_split={
                        split: indices[0] for split, indices in sorted(splits.items())
                    },
                )
            )

    counts: Counter[str] = Counter()
    affected: set[int] = set()
    examples: list[PairFinding] = []
    before = timedelta(seconds=request.purge_before_seconds)
    after = timedelta(seconds=request.embargo_after_seconds)
    for group_id, splits in sorted(grouped.items()):
        for first_split, second_split in combinations(sorted(splits), 2):
            for first_index in splits[first_split]:
                first = request.samples[first_index]
                for second_index in splits[second_split]:
                    second = request.samples[second_index]
                    witness: tuple[str, datetime, datetime] | None = None
                    overlap = purge = embargo = False
                    for first_kind, first_start, first_end in _intervals(first):
                        for second_kind, second_start, second_end in _intervals(second):
                            start, end = max(first_start, second_start), min(first_end, second_end)
                            if start < end:
                                overlap = True
                                if witness is None:
                                    witness = f"{first_kind}/{second_kind}", start, end
                    if first.split == "train" or second.split == "train":
                        training, protected = (
                            (first, second) if first.split == "train" else (second, first)
                        )
                        first_information = min(protected.observation_start, protected.label_start)
                        last_information = max(protected.observation_end, protected.label_end)
                        guards = (
                            ("purge", first_information - before, first_information),
                            ("embargo", last_information, last_information + after),
                        )
                        for guard_kind, guard_start, guard_end in guards:
                            for _, sample_start, sample_end in _intervals(training):
                                start, end = (
                                    max(guard_start, sample_start),
                                    min(guard_end, sample_end),
                                )
                                if start < end:
                                    if guard_kind == "purge":
                                        purge = True
                                    else:
                                        embargo = True
                                    if witness is None:
                                        witness = guard_kind, start, end
                    if witness is None:
                        continue
                    counts["any"] += 1
                    counts["information"] += overlap
                    counts["purge"] += purge
                    counts["embargo"] += embargo
                    affected.update((first_index, second_index))
                    if len(examples) < request.max_diagnostics:
                        examples.append(
                            PairFinding(
                                group_id=group_id,
                                first_row_index=first_index,
                                second_row_index=second_index,
                                first_split=first.split,
                                second_split=second.split,
                                information_overlap=overlap,
                                purge_overlap=purge,
                                embargo_overlap=embargo,
                                first_witness_kind=witness[0],
                                overlap_start=witness[1],
                                overlap_end=witness[2],
                            )
                        )

    return Output(
        sample_count=len(request.samples),
        group_count=len(grouped),
        compared_sample_pairs=work,
        duplicate_sample_id_groups=duplicate_groups,
        cross_split_duplicate_id_pairs=duplicate_pairs,
        information_overlap_pairs=counts["information"],
        purge_overlap_pairs=counts["purge"],
        embargo_overlap_pairs=counts["embargo"],
        leaking_interval_pairs=counts["any"],
        affected_interval_rows=len(affected),
        passed=duplicate_groups == 0 and counts["any"] == 0,
        purge_before_seconds=request.purge_before_seconds,
        embargo_after_seconds=request.embargo_after_seconds,
        duplicate_id_examples=id_examples,
        omitted_duplicate_id_groups=duplicate_groups - len(id_examples),
        pair_examples=examples,
        omitted_interval_pairs=counts["any"] - len(examples),
    )


OPERATION = Operation(
    id="skills.audit_split_leakage",
    kind="skill",
    description="Audit declared train/validation/test sample IDs and separate half-open observation/label windows for cross-split leakage, with explicit pre-purge/post-embargo guards, group isolation, bounded pair work, and no split construction.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
