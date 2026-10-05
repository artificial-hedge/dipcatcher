"""Deterministic heuristic class-balanced folds with indivisible sample groups.

For k folds, C classes, N rows, class totals N_c and fold counts n_fc, minimize
J = (1/C)*sum_(f,c)(n_fc/N_c - 1/k)^2
    + lambda*sum_f(n_f/N - 1/k)^2.
The user supplies nonnegative lambda. Objective comparisons use exact integer
coefficients, including the binary-rational lambda, not rounded float scores.

Group IDs and classes are sorted. A local Random(seed) shuffles group ties;
groups are then stably sorted by decreasing maximum class count and size.
The first k groups seed distinct shuffled folds. Remaining groups choose the
smallest objective increment, with ties following that shuffled fold order.
Each local pass visits that same group order and applies the best strictly
improving single-group move, preserving nonempty folds. Stop after a no-change
pass or the supplied pass cap. Swaps and global optimization are not attempted.
No global optimum or general class-coverage feasibility certificate is given.

All rows in a group share a validation fold; training rows are its complement.
Outputs preserve source indices and sample IDs, and report classes occurring
in fewer than k groups as a necessary obstruction to all-fold class coverage.
Missing validation/training classes are reported even when that obstruction
does not apply. Group IDs are user declarations, not verified independence.
No temporal, availability or outcome-window isolation is enforced.

Class stratification subject to whole-group constraints is described in:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
This implementation uses the explicitly defined objective and search above.
Python RNG repeatability is scoped to its reported implementation/version:
https://docs.python.org/3/library/random.html
"""

import platform
from collections import defaultdict
from fractions import Fraction
from math import lcm
from random import Random
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class Sample(InputModel):
    sample_id: Name
    group_id: Name
    label: Name


class Input(InputModel):
    samples: list[Sample] = Field(min_length=2, max_length=2_048)
    fold_count: int = Field(default=5, strict=True, ge=2, le=20)
    seed: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)
    sample_balance_weight: float = Field(
        default=1.0, strict=True, ge=0, le=100, allow_inf_nan=False
    )
    maximum_local_passes: int = Field(default=5, strict=True, ge=0, le=20)

    @model_validator(mode="after")
    def bound_problem(self) -> Self:
        if len({sample.sample_id for sample in self.samples}) != len(self.samples):
            raise ValueError("sample_id values must be unique")
        groups = len({sample.group_id for sample in self.samples})
        if not self.fold_count <= groups <= 128:
            raise ValueError("the number of groups must be between fold_count and 128")
        if len({sample.label for sample in self.samples}) > 32:
            raise ValueError("at most 32 classes are supported")
        return self


class Assignment(OutputModel):
    source_index: int
    sample_id: str
    validation_fold: int


class GroupAssignment(OutputModel):
    group_id: str
    sample_count: int
    validation_fold: int


class ClassBalance(OutputModel):
    label: str
    sample_count: int
    groups_containing_class: int
    fewer_groups_than_folds: bool
    fold_counts: list[int] = Field(min_length=2, max_length=20)
    validation_folds_missing_class: list[int] = Field(max_length=20)
    training_folds_missing_class: list[int] = Field(max_length=20)


class Output(OutputModel):
    sample_count: int
    group_count: int
    class_count: int
    fold_count: int
    fold_sizes: list[int] = Field(min_length=2, max_length=20)
    fold_group_counts: list[int] = Field(min_length=2, max_length=20)
    sample_balance_weight: float
    initial_objective: float = Field(ge=0, allow_inf_nan=False)
    final_objective: float = Field(ge=0, allow_inf_nan=False)
    exact_objective_improved: bool
    objective_candidates_evaluated: int
    completed_local_passes: int
    accepted_local_moves: int
    termination: Literal["local_search_disabled", "no_improving_single_group_move", "pass_limit"]
    global_optimum_certified: Literal[False] = False
    general_coverage_feasibility_certified: Literal[False] = False
    assignments: list[Assignment] = Field(min_length=2, max_length=2_048)
    groups: list[GroupAssignment] = Field(min_length=2, max_length=128)
    classes: list[ClassBalance] = Field(min_length=1, max_length=32)
    seed: int
    rng: Literal["python.random.Random(MT19937).shuffle"] = "python.random.Random(MT19937).shuffle"
    python_implementation: str
    python_version: str


def execute(request: Input, context: OperationContext) -> Output:
    """Greedily allocate whole groups and run bounded exact-objective move passes."""
    size, folds = len(request.samples), request.fold_count
    labels = sorted({sample.label for sample in request.samples})
    class_index = {label: index for index, label in enumerate(labels)}
    members: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(request.samples):
        members[sample.group_id].append(index)
    group_ids = sorted(members)
    group_sizes = [len(members[group]) for group in group_ids]
    group_counts = [[0] * len(labels) for _ in group_ids]
    for group, name in enumerate(group_ids):
        for index in members[name]:
            group_counts[group][class_index[request.samples[index].label]] += 1
    totals = [sum(row[column] for row in group_counts) for column in range(len(labels))]
    balance_weight = Fraction(request.sample_balance_weight)
    class_denominators = [len(labels) * folds * folds * total * total for total in totals]
    size_denominator = balance_weight.denominator * folds * folds * size * size
    objective_denominator = lcm(size_denominator, *class_denominators)
    class_coefficients = [
        objective_denominator // denominator for denominator in class_denominators
    ]
    size_coefficient = balance_weight.numerator * (objective_denominator // size_denominator)

    def cost(counts: list[int], rows: int) -> int:
        return size_coefficient * (folds * rows - size) ** 2 + sum(
            coefficient * (folds * count - total) ** 2
            for coefficient, count, total in zip(class_coefficients, counts, totals, strict=True)
        )

    generator = Random(request.seed)
    order = list(range(len(group_ids)))
    generator.shuffle(order)
    order.sort(key=lambda group: (-max(group_counts[group]), -group_sizes[group]))
    fold_order = list(range(folds))
    generator.shuffle(fold_order)
    assignments = [-1] * len(group_ids)
    fold_counts = [[0] * len(labels) for _ in range(folds)]
    fold_sizes = [0] * folds
    fold_groups = [0] * folds
    fold_costs = [cost(row, 0) for row in fold_counts]
    evaluations = 0

    def changed_counts(fold: int, group: int, direction: int) -> list[int]:
        return [
            a + direction * b for a, b in zip(fold_counts[fold], group_counts[group], strict=True)
        ]

    for position, group in enumerate(order):
        candidates = [fold_order[position]] if position < folds else fold_order
        best_fold = candidates[0]
        best_delta: int | None = None
        for fold in candidates:
            delta = (
                cost(changed_counts(fold, group, 1), fold_sizes[fold] + group_sizes[group])
                - fold_costs[fold]
            )
            evaluations += 1
            if best_delta is None or delta < best_delta:
                best_fold, best_delta = fold, delta
        assignments[group] = best_fold
        fold_counts[best_fold] = changed_counts(best_fold, group, 1)
        fold_sizes[best_fold] += group_sizes[group]
        fold_groups[best_fold] += 1
        fold_costs[best_fold] = cost(fold_counts[best_fold], fold_sizes[best_fold])
    initial_objective = sum(fold_costs)
    moves = passes = 0
    termination: Literal[
        "local_search_disabled", "no_improving_single_group_move", "pass_limit"
    ] = "pass_limit" if request.maximum_local_passes else "local_search_disabled"
    for _ in range(request.maximum_local_passes):
        pass_moves = 0
        for group in order:
            source = assignments[group]
            if fold_groups[source] <= 1:
                continue
            reduced_counts = changed_counts(source, group, -1)
            reduced_size = fold_sizes[source] - group_sizes[group]
            reduced_cost = cost(reduced_counts, reduced_size)
            best_fold, best_delta = source, 0
            for destination in fold_order:
                if destination == source:
                    continue
                increased_cost = cost(
                    changed_counts(destination, group, 1),
                    fold_sizes[destination] + group_sizes[group],
                )
                delta = reduced_cost + increased_cost - fold_costs[source] - fold_costs[destination]
                evaluations += 1
                if delta < best_delta:
                    best_fold, best_delta = destination, delta
            if best_fold == source:
                continue
            fold_counts[source], fold_sizes[source], fold_costs[source] = (
                reduced_counts,
                reduced_size,
                reduced_cost,
            )
            fold_counts[best_fold] = changed_counts(best_fold, group, 1)
            fold_sizes[best_fold] += group_sizes[group]
            fold_costs[best_fold] = cost(fold_counts[best_fold], fold_sizes[best_fold])
            fold_groups[source] -= 1
            fold_groups[best_fold] += 1
            assignments[group] = best_fold
            pass_moves += 1
        passes += 1
        moves += pass_moves
        if not pass_moves:
            termination = "no_improving_single_group_move"
            break
    by_id = dict(zip(group_ids, assignments, strict=True))
    balances: list[ClassBalance] = []
    for column, label in enumerate(labels):
        counts = [row[column] for row in fold_counts]
        containing = sum(row[column] > 0 for row in group_counts)
        balances.append(
            ClassBalance(
                label=label,
                sample_count=totals[column],
                groups_containing_class=containing,
                fewer_groups_than_folds=containing < folds,
                fold_counts=counts,
                validation_folds_missing_class=[
                    fold for fold, count in enumerate(counts) if count == 0
                ],
                training_folds_missing_class=[
                    fold for fold, count in enumerate(counts) if count == totals[column]
                ],
            )
        )
    final_objective = sum(fold_costs)
    return Output(
        sample_count=size,
        group_count=len(group_ids),
        class_count=len(labels),
        fold_count=folds,
        fold_sizes=fold_sizes,
        fold_group_counts=fold_groups,
        sample_balance_weight=request.sample_balance_weight,
        initial_objective=float(Fraction(initial_objective, objective_denominator)),
        final_objective=float(Fraction(final_objective, objective_denominator)),
        exact_objective_improved=final_objective < initial_objective,
        objective_candidates_evaluated=evaluations,
        completed_local_passes=passes,
        accepted_local_moves=moves,
        termination=termination,
        assignments=[
            Assignment(
                source_index=index,
                sample_id=sample.sample_id,
                validation_fold=by_id[sample.group_id],
            )
            for index, sample in enumerate(request.samples)
        ],
        groups=[
            GroupAssignment(
                group_id=group, sample_count=group_sizes[index], validation_fold=assignments[index]
            )
            for index, group in enumerate(group_ids)
        ],
        classes=balances,
        seed=request.seed,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
    )


OPERATION = Operation(
    id="skills.build_group_stratified_folds",
    kind="skill",
    description=(
        "Allocate whole groups with a seeded greedy and bounded local-move balance objective, "
        "returning original sample IDs, group assignments and missing-class diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
