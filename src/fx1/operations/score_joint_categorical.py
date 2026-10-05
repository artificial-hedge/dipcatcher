"""Score finite joint categorical distributions without assuming independence.

Axes declare outcome labels; states declare unique tuples receiving forecast
mass. Undeclared Cartesian states have exactly zero probability. Each row is
normalized by its positive total mass and marginalized separately over each
axis. Joint multicategory Brier (sum over states, no division by category count)
and logarithmic losses are proper scores. Marginal losses alone cannot identify
the joint law. See Gneiting and Raftery (2007), section 4:
https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf

The observed log joint/product-of-marginals ratio is a dependence diagnostic,
not an aggregate information estimate or forecast score. Zero joint support has
infinite log loss; a zero marginal makes the dependence ratio undefined. No
clipping, pseudocounts, fitted dependence or availability certification occurs.
Exact arithmetic uses binary-float inputs and 131072-bit cross-row accumulators.
Nonzero conversion underflow fails; logarithms are numerical approximations.
"""

from __future__ import annotations

from fractions import Fraction
from math import fsum, isfinite, log, log1p, prod
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Label = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
State = Annotated[list[Label], Field(min_length=2, max_length=6)]
Mass = Annotated[float, Field(strict=True, ge=0, le=1e100)]
MassRow = Annotated[list[Mass], Field(min_length=1, max_length=2048)]


class Axis(InputModel):
    name: Label
    categories: list[Label] = Field(min_length=2, max_length=32)


class Input(InputModel):
    axes: list[Axis] = Field(min_length=2, max_length=6)
    states: list[State] = Field(min_length=1, max_length=2048)
    outcomes: list[State] = Field(min_length=1, max_length=1000)
    masses: list[MassRow] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def aligned_states(self) -> Self:
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError("axis names must be distinct")
        categories = [set(axis.categories) for axis in self.axes]
        if any(
            len(labels) != len(axis.categories)
            for labels, axis in zip(categories, self.axes, strict=True)
        ):
            raise ValueError("categories must be distinct within each axis")
        for state in (*self.states, *self.outcomes):
            if len(state) != len(self.axes) or any(
                value not in categories[index] for index, value in enumerate(state)
            ):
                raise ValueError("each state/outcome must contain one declared category per axis")
        if len({tuple(state) for state in self.states}) != len(self.states):
            raise ValueError("supplied joint states must be unique")
        if len(self.outcomes) != len(self.masses) or any(
            len(row) != len(self.states) for row in self.masses
        ):
            raise ValueError("mass rows must align with outcomes and the shared state table")
        if len(self.states) * len(self.outcomes) * len(self.axes) > 200_000:
            raise ValueError("joint marginalization exceeds 200000 state-observation-axis cells")
        if any(not any(row) for row in self.masses):
            raise ValueError("every forecast must have positive total mass")
        return self


class RowScore(OutputModel):
    row_index: int
    joint_outcome_probability: float
    joint_brier: float
    joint_log_loss: float | None
    joint_log_loss_status: Literal["finite", "infinite_zero_support"]
    marginal_outcome_probabilities: list[float]
    marginal_brier: list[float]
    marginal_log_loss: list[float | None]
    log_joint_over_product_marginals: float | None
    dependence_ratio_status: Literal[
        "finite", "negative_infinity_zero_joint", "undefined_zero_marginal"
    ]


class AxisScore(OutputModel):
    name: str
    category_count: int
    mean_brier: float
    mean_log_loss: float | None
    infinite_log_loss_rows: int


class Output(OutputModel):
    observation_count: int
    axis_names: list[str]
    supplied_state_count: int
    cartesian_state_count: int
    implicit_zero_state_count: int
    normalized_input_mass_rows: int
    minimum_input_mass: float
    maximum_input_mass: float
    mean_joint_brier: float
    mean_joint_log_loss: float | None
    infinite_joint_log_loss_rows: int
    marginal_scores: list[AxisScore]
    rows: list[RowScore]
    brier_convention: Literal["sum_over_all_categories_without_division"] = (
        "sum_over_all_categories_without_division"
    )
    undeclared_state_policy: Literal["zero_probability"] = "zero_probability"
    maximum_fraction_bits: Literal[131072] = 131072
    independence_assumed: Literal[False] = False
    forecast_timing_validated: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 131_072:
        raise ValueError("joint scoring exceeds the 131072-bit exact arithmetic budget")
    return value


def _number(value: Fraction) -> float:
    _bounded(value)
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError("joint score exceeds finite floating-point range") from exc
    if not isfinite(converted) or (value and converted == 0):
        raise ValueError("joint score or probability is outside representable finite range")
    return converted


def _negative_log(probability: Fraction) -> float | None:
    if not probability:
        return None
    if probability == 1:
        return 0.0
    if probability > Fraction(1, 2):
        value = -log1p(_number(probability - 1))
    else:
        exponent = probability.numerator.bit_length() - probability.denominator.bit_length()
        scaled = probability / Fraction(2) ** exponent
        value = -fsum((log(float(scaled)), exponent * log(2)))
    if not isfinite(value) or value <= 0:
        raise ValueError("positive logarithmic loss is outside supported numerical range")
    return value


def execute(request: Input, context: OperationContext) -> Output:
    axis_count = len(request.axes)
    maps = [{label: index for index, label in enumerate(axis.categories)} for axis in request.axes]
    states = [[maps[axis][label] for axis, label in enumerate(state)] for state in request.states]
    positions = {tuple(state): index for index, state in enumerate(request.states)}
    rows: list[RowScore] = []
    joint_brier_sum = joint_log_sum = Fraction()
    marginal_brier_sums = [Fraction() for _ in request.axes]
    marginal_log_sums = [Fraction() for _ in request.axes]
    marginal_infinite = [0] * axis_count
    joint_infinite = 0
    input_totals: list[Fraction] = []
    for row_index, (outcome, masses) in enumerate(
        zip(request.outcomes, request.masses, strict=True)
    ):
        total = sum((Fraction(value) for value in masses), Fraction())
        input_totals.append(total)
        probabilities = [Fraction(value) / total for value in masses]
        realized_index = positions.get(tuple(outcome))
        realized = probabilities[realized_index] if realized_index is not None else Fraction()
        joint_brier = 1 - 2 * realized + sum((value * value for value in probabilities), Fraction())
        joint_loss = _negative_log(realized)
        joint_brier_sum = _bounded(joint_brier_sum + joint_brier)
        if joint_loss is None:
            joint_infinite += 1
        else:
            joint_log_sum += Fraction(joint_loss)
        marginal = [[Fraction() for _ in axis.categories] for axis in request.axes]
        for state, probability in zip(states, probabilities, strict=True):
            for axis, category in enumerate(state):
                marginal[axis][category] += probability
        outcome_masses: list[float] = []
        brier_values: list[float] = []
        log_values: list[float | None] = []
        for axis, distribution in enumerate(marginal):
            mass = distribution[maps[axis][outcome[axis]]]
            brier = 1 - 2 * mass + sum((value * value for value in distribution), Fraction())
            loss = _negative_log(mass)
            outcome_masses.append(_number(mass))
            brier_values.append(_number(brier))
            log_values.append(loss)
            marginal_brier_sums[axis] = _bounded(marginal_brier_sums[axis] + brier)
            if loss is None:
                marginal_infinite[axis] += 1
            else:
                marginal_log_sums[axis] += Fraction(loss)
        ratio: float | None = None
        ratio_status: Literal["finite", "negative_infinity_zero_joint", "undefined_zero_marginal"]
        if any(value is None for value in log_values):
            ratio_status = "undefined_zero_marginal"
        elif joint_loss is None:
            ratio_status = "negative_infinity_zero_joint"
        else:
            ratio_status = "finite"
            ratio = fsum([value for value in log_values if value is not None] + [-joint_loss])
        rows.append(
            RowScore(
                row_index=row_index,
                joint_outcome_probability=_number(realized),
                joint_brier=_number(joint_brier),
                joint_log_loss=joint_loss,
                joint_log_loss_status="finite"
                if joint_loss is not None
                else "infinite_zero_support",
                marginal_outcome_probabilities=outcome_masses,
                marginal_brier=brier_values,
                marginal_log_loss=log_values,
                log_joint_over_product_marginals=ratio,
                dependence_ratio_status=ratio_status,
            )
        )
    count = len(rows)
    cartesian = prod(len(axis.categories) for axis in request.axes)
    return Output(
        observation_count=count,
        axis_names=[axis.name for axis in request.axes],
        supplied_state_count=len(states),
        cartesian_state_count=cartesian,
        implicit_zero_state_count=cartesian - len(states),
        normalized_input_mass_rows=sum(total != 1 for total in input_totals),
        minimum_input_mass=_number(min(input_totals)),
        maximum_input_mass=_number(max(input_totals)),
        mean_joint_brier=_number(joint_brier_sum / count),
        mean_joint_log_loss=None if joint_infinite else _number(joint_log_sum / count),
        infinite_joint_log_loss_rows=joint_infinite,
        marginal_scores=[
            AxisScore(
                name=axis.name,
                category_count=len(axis.categories),
                mean_brier=_number(marginal_brier_sums[index] / count),
                mean_log_loss=None
                if marginal_infinite[index]
                else _number(marginal_log_sums[index] / count),
                infinite_log_loss_rows=marginal_infinite[index],
            )
            for index, axis in enumerate(request.axes)
        ],
        rows=rows,
    )


OPERATION = Operation(
    id="skills.score_joint_categorical",
    kind="skill",
    description="Score sparse finite joint categorical forecasts with joint and marginal Brier/log losses, exact mass aggregation, explicit zero-support outcomes and observed dependence diagnostics without an independence assumption.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
