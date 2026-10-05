"""Construct caller-scheduled chronological research folds with observable labels.

Training decisions lie in [train_start, test_start-gap). Their label intervals
must end by that cutoff and their labels must be available by test_start.
Test decisions lie in [test_start, test_end). Test-label availability is not
required at prediction time. This constructs indices, not fitted models or
statistical evidence. Fold schedules and sample information clocks are supplied.
"""

from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]


def _clock(value: object) -> datetime:
    if isinstance(value, str) and len(value) <= 64:
        clock = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        clock = value
    else:
        raise ValueError("timestamps must be timezone-aware ISO strings or datetimes")
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("timestamps require an explicit timezone")
    try:
        return clock.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise ValueError("timestamp cannot be represented in UTC") from exc


class Sample(InputModel):
    sample_id: Name
    decision_time: AwareDatetime
    max_source_available_time: AwareDatetime
    label_end_time: AwareDatetime
    label_available_time: AwareDatetime

    @field_validator(
        "decision_time",
        "max_source_available_time",
        "label_end_time",
        "label_available_time",
        mode="before",
    )
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def coherent_information(self) -> Self:
        if self.max_source_available_time > self.decision_time:
            raise ValueError("a sample uses a source unavailable at its decision time")
        if self.label_end_time < self.decision_time:
            raise ValueError("label_end_time cannot precede decision_time")
        if self.label_available_time < self.label_end_time:
            raise ValueError("label_available_time cannot precede label_end_time")
        return self


class Schedule(InputModel):
    fold_id: Name
    train_start: AwareDatetime
    test_start: AwareDatetime
    test_end: AwareDatetime

    @field_validator("train_start", "test_start", "test_end", mode="before")
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def ordered_boundaries(self) -> Self:
        if not self.train_start < self.test_start < self.test_end:
            raise ValueError("each fold requires train_start < test_start < test_end")
        return self


class Input(InputModel):
    samples: list[Sample] = Field(min_length=1, max_length=5000)
    schedules: list[Schedule] = Field(min_length=1, max_length=100)
    gap_microseconds: int = Field(default=0, strict=True, ge=0, le=315_537_897_600_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=100)
    limit: int = Field(default=5, strict=True, ge=1, le=5)

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        if len({row.sample_id for row in self.samples}) != len(self.samples):
            raise ValueError("sample IDs must be globally unique")
        if len({fold.fold_id for fold in self.schedules}) != len(self.schedules):
            raise ValueError("fold IDs must be unique")
        ordered = sorted(self.schedules, key=lambda fold: fold.test_start)
        if any(left.test_end > right.test_start for left, right in pairwise(ordered)):
            raise ValueError("test decision intervals must not overlap across folds")
        gap = timedelta(microseconds=self.gap_microseconds)
        for fold in self.schedules:
            try:
                cutoff = fold.test_start - gap
            except OverflowError as exc:
                raise ValueError("gap places the training cutoff outside datetime range") from exc
            if cutoff <= fold.train_start:
                raise ValueError("gap must leave a nonempty training decision interval")
        return self


class Fold(OutputModel):
    fold_id: str
    fit_time: AwareDatetime
    train_start: AwareDatetime
    training_cutoff: AwareDatetime
    test_end: AwareDatetime
    candidate_training_samples: int
    excluded_gap_decisions: int
    excluded_crossing_labels: int
    excluded_unavailable_labels: int
    training_samples: int
    test_samples: int
    usable: bool
    train_indices: list[int]
    test_indices: list[int]


class Output(OutputModel):
    sample_count: int
    fold_count: int
    usable_folds: int
    total_training_assignments: int
    total_test_assignments: int
    gap_microseconds: int
    folds: list[Fold]
    offset: int
    next_offset: int | None


def execute(request: Input, context: OperationContext) -> Output:
    """Evaluate every schedule; paginate folds only after complete accounting."""
    gap = timedelta(microseconds=request.gap_microseconds)
    page: list[Fold] = []
    usable = train_total = test_total = 0
    for position, fold in enumerate(request.schedules):
        cutoff = fold.test_start - gap
        train: list[int] = []
        test: list[int] = []
        candidates = gap_count = crossing = unavailable = 0
        for index, sample in enumerate(request.samples):
            if fold.test_start <= sample.decision_time < fold.test_end:
                test.append(index)
            if not fold.train_start <= sample.decision_time < fold.test_start:
                continue
            candidates += 1
            if sample.decision_time >= cutoff:
                gap_count += 1
            elif sample.label_end_time > cutoff:
                crossing += 1
            elif sample.label_available_time > fold.test_start:
                unavailable += 1
            else:
                train.append(index)
        train_total += len(train)
        test_total += len(test)
        usable += int(bool(train and test))
        if request.offset <= position < request.offset + request.limit:
            page.append(
                Fold(
                    fold_id=fold.fold_id,
                    fit_time=fold.test_start,
                    train_start=fold.train_start,
                    training_cutoff=cutoff,
                    test_end=fold.test_end,
                    candidate_training_samples=candidates,
                    excluded_gap_decisions=gap_count,
                    excluded_crossing_labels=crossing,
                    excluded_unavailable_labels=unavailable,
                    training_samples=len(train),
                    test_samples=len(test),
                    usable=bool(train and test),
                    train_indices=train,
                    test_indices=test,
                )
            )
    stop = request.offset + len(page)
    return Output(
        sample_count=len(request.samples),
        fold_count=len(request.schedules),
        usable_folds=usable,
        total_training_assignments=train_total,
        total_test_assignments=test_total,
        gap_microseconds=request.gap_microseconds,
        folds=page,
        offset=request.offset,
        next_offset=stop if stop < len(request.schedules) else None,
    )


OPERATION = Operation(
    id="skills.build_walk_forward_folds",
    kind="skill",
    description=(
        "Build scheduled chronological train/test index sets with a pre-test gap, "
        "label-end purging and label-availability checks at fit time. Enforces PIT "
        "feature clocks and nonoverlapping test intervals; reports all exclusion counts."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
