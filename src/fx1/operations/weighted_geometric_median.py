"""A bounded modified Weiszfeld solver for the weighted Euclidean median.

Normalize supplied nonnegative weights to p_i and minimize sum_i p_i*|x_i-y|.
At an iterate y let eta be the mass exactly coincident with y, R the sum of
p_i*(x_i-y)/|x_i-y| over noncoincident points, and r=|R|. The minimum subgradient
norm is max(0,r-eta). When this is above tolerance, form the usual Weiszfeld
barycenter T of noncoincident points and update y <- (1-eta/r)*T+(eta/r)*y.
This allows departure from a data point that is not a median. Duplicate points
contribute their combined mass, and zero weights contribute nothing.

Distances use exact binary-rational differences and correctly rounded square
roots. Weighted sums and anchored updates are rational, with a 32768-bit cap;
distances, gradient components and each returned iterate round to binary64.
Diagnostics are recalculated at the returned iterate. Numerical convergence
means only that its approximate normalized subgradient residual meets the
threshold. Iteration limits, unchanged rounded iterates and objective increases
are separate unsuccessful statuses; no global-optimum or uniqueness certificate
is issued. Tiny nonzero unrepresentable outputs fail explicitly. Coordinate
units and relative scaling across dimensions are the caller's responsibility.
No timing, independence or distributional claims are made.

Modified iteration reference: Vardi and Zhang (2000):
https://www.pnas.org/doi/10.1073/pnas.97.4.1423
"""

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Point = Annotated[list[Value], Field(min_length=1, max_length=8)]
Initialization = Literal["weighted_mean", "first_positive_point"]
Status = Literal[
    "residual_tolerance", "iteration_limit", "numerical_stagnation", "objective_increase"
]
_BIT_LIMIT = 32_768


class Input(InputModel):
    points: list[Point] = Field(min_length=1, max_length=128)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=128)
    initialization: Initialization = "weighted_mean"
    residual_tolerance: float = Field(default=1e-9, strict=True, ge=1e-12, le=1e-3)
    max_iterations: int = Field(default=100, strict=True, ge=1, le=500)

    @model_validator(mode="after")
    def aligned_bounded_points(self) -> Self:
        dimension = len(self.points[0])
        if any(len(point) != dimension for point in self.points):
            raise ValueError("points must have a common dimension")
        if self.weights is not None and (
            len(self.weights) != len(self.points) or not any(self.weights)
        ):
            raise ValueError("weights must align with points and have positive total mass")
        if len(self.points) * dimension * (self.max_iterations + 2) > 200_000:
            raise ValueError("solver exceeds 200000 point-coordinate-iteration work cells")
        return self


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    positive_weight_count: int
    weight_sum: float
    normalized_weights: list[float]
    initialization: Initialization
    initial_point: list[float]
    median_estimate: list[float]
    weighted_mean_distance: float
    initial_weighted_mean_distance: float
    objective_decrease: float
    normalized_subgradient_residual: float
    noncoincident_gradient_components: list[float]
    noncoincident_gradient_norm: float
    coincident_weight_fraction: float
    coincident_source_indexes: list[int]
    accepted_iterations: int
    objective_evaluations: int
    maximum_iterations: int
    last_accepted_step_distance: float
    rejected_objective_increase: float | None
    residual_tolerance: float
    status: Status
    converged_numerically: bool
    objective_definition: Literal["sum_normalized_weight_times_euclidean_distance"] = (
        "sum_normalized_weight_times_euclidean_distance"
    )
    global_optimality_certified: Literal[False] = False
    uniqueness_certified: Literal[False] = False
    timing_verified: Literal[False] = False


@dataclass
class _State:
    distances: list[float]
    objective: Fraction
    gradient: list[float]
    gradient_norm: float
    coincident_mass: Fraction
    residual: Fraction


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _BIT_LIMIT:
        raise ValueError("geometric median rational arithmetic exceeds 32768-bit budget")
    return value


def _sum(values: Iterable[Fraction]) -> Fraction:
    result = Fraction()
    for value in values:
        result = _bounded(result + value)
    return result


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _norm(values: Iterable[Fraction]) -> float:
    squared = _sum(value * value for value in values)
    result = correctly_rounded_sqrt(squared.numerator, squared.denominator)
    if not isfinite(result) or (squared and result == 0):
        raise ValueError("nonzero Euclidean norm is outside finite binary64 range")
    return result


def _evaluate(
    points: list[list[Fraction]], probabilities: list[Fraction], y: list[float]
) -> _State:
    exact_y = [Fraction(value) for value in y]
    differences = [
        [value - center for value, center in zip(point, exact_y, strict=True)] for point in points
    ]
    distances = [_norm(delta) for delta in differences]
    objective = _sum(
        probability * Fraction(distance)
        for probability, distance in zip(probabilities, distances, strict=True)
    )
    coincident = _sum(
        probability
        for probability, distance in zip(probabilities, distances, strict=True)
        if not distance
    )
    gradient = [
        _number(
            _sum(
                -probabilities[index] * differences[index][column] / Fraction(distance)
                for index, distance in enumerate(distances)
                if distance
            ),
            "subgradient component",
        )
        for column in range(len(y))
    ]
    gradient_norm = _norm(Fraction(value) for value in gradient)
    residual = max(Fraction(), Fraction(gradient_norm) - coincident)
    return _State(distances, objective, gradient, gradient_norm, coincident, residual)


def execute(request: Input, context: OperationContext) -> Output:
    raw_weights = request.weights if request.weights is not None else [1.0] * len(request.points)
    weights = [Fraction(value) for value in raw_weights]
    total = _sum(weights)
    normalized = [weight / total for weight in weights]
    # Validate representability before commencing the iterative work.
    reported_weights = [_number(value, "normalized weight") for value in normalized]
    active = [index for index, weight in enumerate(weights) if weight]
    points = [[Fraction(value) for value in request.points[index]] for index in active]
    probabilities = [normalized[index] for index in active]
    dimension = len(points[0])
    if request.initialization == "first_positive_point":
        current = list(request.points[active[0]])
    else:
        current = [
            _number(
                _sum(
                    probability * point[column]
                    for probability, point in zip(probabilities, points, strict=True)
                ),
                "initial weighted mean",
            )
            for column in range(dimension)
        ]
    initial = list(current)
    state = _evaluate(points, probabilities, current)
    initial_objective = state.objective
    evaluations, accepted = 1, 0
    last_step = 0.0
    rejected_increase: float | None = None
    status: Status = "iteration_limit"
    for _ in range(request.max_iterations):
        if state.residual <= Fraction(request.residual_tolerance):
            status = "residual_tolerance"
            break
        minimum_distance = min(distance for distance in state.distances if distance)
        # Common distance scaling keeps inverse-distance weights bounded without
        # changing their barycenter; no small coefficient is dropped.
        ratios = [
            probability * Fraction(minimum_distance) / Fraction(distance)
            if distance
            else Fraction()
            for probability, distance in zip(probabilities, state.distances, strict=True)
        ]
        denominator = _sum(ratios)
        relaxation = 1 - state.coincident_mass / Fraction(state.gradient_norm)
        proposal = [
            _number(
                Fraction(current[column])
                + relaxation
                * _sum(
                    ratio * (point[column] - Fraction(current[column]))
                    for ratio, point in zip(ratios, points, strict=True)
                )
                / denominator,
                "updated median coordinate",
            )
            for column in range(dimension)
        ]
        if proposal == current:
            status = "numerical_stagnation"
            break
        proposed_state = _evaluate(points, probabilities, proposal)
        evaluations += 1
        if proposed_state.objective > state.objective:
            rejected_increase = _number(
                proposed_state.objective - state.objective, "rejected objective increase"
            )
            status = "objective_increase"
            break
        last_step = _norm(
            Fraction(after) - Fraction(before)
            for after, before in zip(proposal, current, strict=True)
        )
        current, state = proposal, proposed_state
        accepted += 1
    if state.residual <= Fraction(request.residual_tolerance):
        status = "residual_tolerance"
    return Output(
        observation_count=len(request.points),
        dimension_count=dimension,
        positive_weight_count=len(active),
        weight_sum=_number(total, "weight sum"),
        normalized_weights=reported_weights,
        initialization=request.initialization,
        initial_point=initial,
        median_estimate=current,
        weighted_mean_distance=_number(state.objective, "weighted mean distance"),
        initial_weighted_mean_distance=_number(initial_objective, "initial objective"),
        objective_decrease=_number(initial_objective - state.objective, "objective decrease"),
        normalized_subgradient_residual=_number(state.residual, "subgradient residual"),
        noncoincident_gradient_components=state.gradient,
        noncoincident_gradient_norm=state.gradient_norm,
        coincident_weight_fraction=_number(state.coincident_mass, "coincident weight"),
        coincident_source_indexes=[
            active[index] for index, distance in enumerate(state.distances) if not distance
        ],
        accepted_iterations=accepted,
        objective_evaluations=evaluations,
        maximum_iterations=request.max_iterations,
        last_accepted_step_distance=last_step,
        rejected_objective_increase=rejected_increase,
        residual_tolerance=request.residual_tolerance,
        status=status,
        converged_numerically=status == "residual_tolerance",
    )


OPERATION = Operation(
    id="features.weighted_geometric_median",
    kind="feature",
    description=(
        "Fit a weighted Euclidean geometric median using a bounded modified Weiszfeld "
        "iteration, including coincident points, objective diagnostics and explicit stop status."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
