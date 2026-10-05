"""Seeded classification fold allocation with per-class and total count balance.

Classes are processed in lexicographic label order. Within each class, sample
IDs are sorted then shuffled with a fresh local random.Random(seed). Each fold
receives floor(class_count/k) rows. Remainders go to currently smallest folds,
with ties broken by a separately shuffled fold order. Assigning shuffled samples
to those quotas in numeric fold order guarantees per-class and total fold-size
ranges of at most one. Each supplied sample is assigned to exactly one fold.

Assignments by sample ID are invariant to input row reordering, but renaming
IDs or labels can change them. This is an independent allocation algorithm, not
a reproduction of scikit-learn's splitter. Repeatability is scoped to the
reported Python implementation/version and its Random.shuffle algorithm.

The default policy rejects any class smaller than the fold count. The explicit
allow_missing policy spreads its rows over distinct folds and reports missing
classes, including folds whose training complement would omit a singleton.
Training rows for fold f are the complement of its validation assignments.
No temporal, entity-group, availability or outcome-window isolation is enforced;
the caller must establish appropriate sample selection and split validity.

Stratification and fold-size balance objectives: scikit-learn StratifiedKFold:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
RNG scope: https://docs.python.org/3/library/random.html
"""

import platform
from collections import defaultdict
from random import Random
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
SmallPolicy = Literal["error", "allow_missing"]


class Sample(InputModel):
    sample_id: Name
    label: Name


class Input(InputModel):
    samples: list[Sample] = Field(min_length=2, max_length=2_048)
    fold_count: int = Field(default=5, strict=True, ge=2, le=20)
    seed: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    small_stratum_policy: SmallPolicy = "error"

    @model_validator(mode="after")
    def validate_strata(self) -> Self:
        if self.fold_count > len(self.samples):
            raise ValueError("fold_count cannot exceed the sample count")
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id values must be unique")
        counts: dict[str, int] = defaultdict(int)
        for sample in self.samples:
            counts[sample.label] += 1
        if len(counts) > 128:
            raise ValueError("at most 128 distinct classes are supported")
        if self.small_stratum_policy == "error" and any(
            count < self.fold_count for count in counts.values()
        ):
            raise ValueError(
                "every class must contain at least fold_count samples under error policy"
            )
        return self


class Assignment(OutputModel):
    source_index: int
    sample_id: str
    validation_fold: int


class ClassBalance(OutputModel):
    label: str
    sample_count: int
    fold_counts: list[int] = Field(min_length=2, max_length=20)
    count_range: int
    validation_folds_missing_class: list[int] = Field(max_length=20)
    training_folds_missing_class: list[int] = Field(max_length=20)


class Output(OutputModel):
    sample_count: int
    class_count: int
    fold_count: int
    seed: int
    small_stratum_policy: SmallPolicy
    fold_sizes: list[int] = Field(min_length=2, max_length=20)
    fold_size_range: int
    classes_smaller_than_fold_count: int
    assignments: list[Assignment] = Field(min_length=2, max_length=2_048)
    class_balance: list[ClassBalance] = Field(min_length=1, max_length=128)
    rng: Literal["python.random.Random(MT19937).shuffle"] = "python.random.Random(MT19937).shuffle"
    python_implementation: str
    python_version: str
    sample_order_policy: Literal["sorted_ids_within_sorted_labels"] = (
        "sorted_ids_within_sorted_labels"
    )


def execute(request: Input, context: OperationContext) -> Output:
    """Allocate balanced integer class quotas, then assign locally shuffled rows."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(request.samples):
        groups[sample.label].append(index)
    generator = Random(request.seed)
    sizes = [0] * request.fold_count
    assigned = [-1] * len(request.samples)
    balances: list[ClassBalance] = []
    small_count = 0
    for label, indices in sorted(groups.items()):
        indices.sort(key=lambda index: request.samples[index].sample_id)
        generator.shuffle(indices)
        tie_order = list(range(request.fold_count))
        generator.shuffle(tie_order)
        # Stable sorting preserves the shuffled order among equally loaded folds.
        extra_order = sorted(tie_order, key=sizes.__getitem__)
        base, remainder = divmod(len(indices), request.fold_count)
        quotas = [base] * request.fold_count
        for fold in extra_order[:remainder]:
            quotas[fold] += 1
        cursor = 0
        for fold, count in enumerate(quotas):
            for index in indices[cursor : cursor + count]:
                assigned[index] = fold
            cursor += count
            sizes[fold] += count
        small_count += len(indices) < request.fold_count
        balances.append(
            ClassBalance(
                label=label,
                sample_count=len(indices),
                fold_counts=quotas,
                count_range=max(quotas) - min(quotas),
                validation_folds_missing_class=[
                    fold for fold, count in enumerate(quotas) if count == 0
                ],
                training_folds_missing_class=[
                    fold for fold, count in enumerate(quotas) if count == len(indices)
                ],
            )
        )
    return Output(
        sample_count=len(request.samples),
        class_count=len(groups),
        fold_count=request.fold_count,
        seed=request.seed,
        small_stratum_policy=request.small_stratum_policy,
        fold_sizes=sizes,
        fold_size_range=max(sizes) - min(sizes),
        classes_smaller_than_fold_count=small_count,
        assignments=[
            Assignment(
                source_index=index, sample_id=sample.sample_id, validation_fold=assigned[index]
            )
            for index, sample in enumerate(request.samples)
        ],
        class_balance=balances,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
    )


OPERATION = Operation(
    id="skills.build_stratified_folds",
    kind="skill",
    description=(
        "Allocate reproducible seeded classification folds with balanced total and class counts, "
        "explicit small-stratum policy, original sample IDs and missing-class diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
