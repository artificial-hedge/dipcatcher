"""Finite-chain closed-class absorption through exact fundamental-matrix solves.

Rows of nonnegative transition weights are normalized exactly. Positive support
determines communicating classes; a class is closed if no edge leaves it.
Transient states precede no particular named absorbing target: every closed
class, including nontrivial recurrent classes, is treated as an entry target.
For transient block Q and one-step class probabilities R, N=(I-Q)^-1 and B=NR.
N[i,j] counts visits to transient j at times 0,...,T-1 before first closed-class
entry T. E[T]=N*1. For class c with hitting probability h, E[T*1(enter c)]=N*h;
conditional hitting time divides this quantity by h. Unconditional class hitting
time is infinite unless h=1; h=0 has no conditional time. Starting in a closed
class gives zero entry time to that class and never reaches other closed classes.

Own support closure, rational elimination and moment calculations do not fit a
model or simulate paths. Returned numbers are binary64 previews, but zero/one
support and infinite statuses use exact rationals. Boundary rounding is flagged;
unrepresentable nonzero numerical summaries fail. At most 16 states, positive
weights in [1e-6,1e6], and 16384-bit rational intermediates are supported. The
chain is time-homogeneous; transition law, step duration and PIT are assumptions.
Fundamental-matrix reference: Grinstead and Snell, Introduction to Probability,
section 11.2, https://math.dartmouth.edu/~prob/prob/prob.pdf
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e6)]
WeightRow = Annotated[list[Weight], Field(min_length=1, max_length=16)]


class Input(InputModel):
    states: list[Name] = Field(min_length=1, max_length=16)
    transition_weights: list[WeightRow] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def square_weights(self) -> Self:
        size = len(self.states)
        if len(set(self.states)) != size or len(self.transition_weights) != size:
            raise ValueError("states must be unique and transition rows must align")
        for row in self.transition_weights:
            if len(row) != size or not any(row) or any(0 < value < 1e-6 for value in row):
                raise ValueError(
                    "each aligned row needs positive mass; positive weights must be >=1e-6"
                )
        return self


class ClosedClass(OutputModel):
    class_index: int
    state_indices: list[int]
    state_names: list[str]
    singleton_absorbing_state: bool


class ClassHitting(OutputModel):
    class_index: int
    absorption_probability: float
    probability_rounded_to_boundary: bool
    unconditional_expected_steps: float | None
    unconditional_status: Literal["finite", "infinite"]
    conditional_expected_steps: float | None
    conditional_status: Literal["defined", "unreachable"]


class StateResult(OutputModel):
    state_index: int
    state_name: str
    closed_class_index: int | None
    expected_steps_to_any_closed_class: float
    expected_transient_visits: list[float]
    class_hitting: list[ClassHitting]


class Output(OutputModel):
    state_count: int
    normalized_transition_matrix: list[list[float]]
    closed_classes: list[ClosedClass]
    transient_state_indices: list[int]
    transient_state_names: list[str]
    states: list[StateResult]
    occupation_includes_initial_transient_state: Literal[True] = True
    occupation_stops_at_first_closed_class: Literal[True] = True
    closed_class_entry_is_not_single_state_absorption: Literal[True] = True
    transition_model_fitted: Literal[False] = False
    timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("Markov absorption arithmetic exceeds 16384-bit budget")
    return value


def _number(value: Fraction) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("Markov absorption summary overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError("nonzero Markov absorption summary is outside finite binary64 range")
    return result


def _inverse(matrix: list[list[Fraction]]) -> list[list[Fraction]]:
    size = len(matrix)
    augmented = [row[:] + [Fraction(i == j) for j in range(size)] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row][column]), None)
        if pivot is None:
            raise ValueError("transient fundamental-matrix system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [_bounded(value / divisor) for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier:
                augmented[row] = [
                    _bounded(value - multiplier * source)
                    for value, source in zip(augmented[row], augmented[column], strict=True)
                ]
    return [row[size:] for row in augmented]


def execute(request: Input, context: OperationContext) -> Output:
    size = len(request.states)
    transition: list[list[Fraction]] = []
    for row in request.transition_weights:
        weights = [Fraction(value) for value in row]
        total = sum(weights, Fraction())
        transition.append([_bounded(value / total) for value in weights])
    reachable = [[i == j or bool(transition[i][j]) for j in range(size)] for i in range(size)]
    for middle in range(size):
        for source in range(size):
            if reachable[source][middle]:
                for target in range(size):
                    reachable[source][target] |= reachable[middle][target]
    unseen = set(range(size))
    closed: list[list[int]] = []
    while unseen:
        source = min(unseen)
        group = [
            target
            for target in sorted(unseen)
            if reachable[source][target] and reachable[target][source]
        ]
        unseen.difference_update(group)
        members = set(group)
        if not any(transition[i][j] for i in group for j in range(size) if j not in members):
            closed.append(group)
    membership = {state: index for index, group in enumerate(closed) for state in group}
    transient = [state for state in range(size) if state not in membership]
    positions = {state: index for index, state in enumerate(transient)}
    fundamental = _inverse(
        [
            [Fraction(i == j) - transition[source][target] for j, target in enumerate(transient)]
            for i, source in enumerate(transient)
        ]
    )
    if any(value < 0 for row in fundamental for value in row):
        raise ValueError("fundamental matrix has a negative expected occupation")
    direct = [
        [sum((transition[source][target] for target in group), Fraction()) for group in closed]
        for source in transient
    ]
    absorption = [
        [
            _bounded(
                sum((fundamental[i][j] * direct[j][c] for j in range(len(transient))), Fraction())
            )
            for c in range(len(closed))
        ]
        for i in range(len(transient))
    ]
    if any(
        sum(row, Fraction()) != 1 or any(value < 0 or value > 1 for value in row)
        for row in absorption
    ):
        raise ValueError("closed-class absorption probabilities failed exact conservation")
    truncated_times = [
        [
            _bounded(
                sum(
                    (fundamental[i][j] * absorption[j][c] for j in range(len(transient))),
                    Fraction(),
                )
            )
            for c in range(len(closed))
        ]
        for i in range(len(transient))
    ]
    results: list[StateResult] = []
    for state in range(size):
        position = positions.get(state)
        occupation = (
            fundamental[position] if position is not None else [Fraction()] * len(transient)
        )
        hitting: list[ClassHitting] = []
        for class_index in range(len(closed)):
            probability = (
                absorption[position][class_index]
                if position is not None
                else Fraction(membership[state] == class_index)
            )
            conditional = (
                _bounded(truncated_times[position][class_index] / probability)
                if position is not None and probability
                else Fraction()
            )
            preview = _number(probability)
            hitting.append(
                ClassHitting(
                    class_index=class_index,
                    absorption_probability=preview,
                    probability_rounded_to_boundary=0 < probability < 1 and preview in (0, 1),
                    unconditional_expected_steps=_number(conditional) if probability == 1 else None,
                    unconditional_status="finite" if probability == 1 else "infinite",
                    conditional_expected_steps=_number(conditional) if probability else None,
                    conditional_status="defined" if probability else "unreachable",
                )
            )
        results.append(
            StateResult(
                state_index=state,
                state_name=request.states[state],
                closed_class_index=membership.get(state),
                expected_steps_to_any_closed_class=_number(sum(occupation, Fraction())),
                expected_transient_visits=[_number(value) for value in occupation],
                class_hitting=hitting,
            )
        )
    return Output(
        state_count=size,
        normalized_transition_matrix=[[_number(value) for value in row] for row in transition],
        closed_classes=[
            ClosedClass(
                class_index=index,
                state_indices=group,
                state_names=[request.states[state] for state in group],
                singleton_absorbing_state=len(group) == 1,
            )
            for index, group in enumerate(closed)
        ],
        transient_state_indices=transient,
        transient_state_names=[request.states[state] for state in transient],
        states=results,
    )


OPERATION = Operation(
    id="features.markov_absorption",
    kind="feature",
    description="Resolve finite Markov closed classes and exact absorption, transient occupation and conditional hitting-time moments.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
