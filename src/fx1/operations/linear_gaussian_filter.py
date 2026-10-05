"""Exact-rational linear Gaussian filtering with optional fixed-interval RTS smoothing.

The explicit initial prior describes x[-1]. Every step t, including zero, uses
x[t]=F[t]x[t-1]+b[t]+q[t], y[t]=H[t]x[t]+d[t]+r[t]. The state dimension is
fixed; measurement dimensions may vary by step. Null measurement coordinates
are unobserved, so the update selects their complement and the corresponding
principal submatrix of R; missing values are not zero-filled. An entirely
missing step returns its prediction unchanged. No parameter estimation occurs.

The prior and every process covariance must be exactly symmetric PSD; every
full measurement covariance must be exactly symmetric PD, even at missing
coordinates. Noise terms are assumed Gaussian, independent across time and
between process/measurement/prior. Own rational matrix products and LDL solves
compute P-=F P F'+Q, S=H P- H'+R, K=P- H' S^-1, m=m-+K innovation,
P=P--K S K'. These recursions are exact for the binary64 inputs, up to a
32768-bit intermediate budget. Means/covariances are rounded only for output.

Optional smoothing uses G[t]=P[t] F[t+1]' inverse(P-[t+1]), followed by the
standard RTS mean/covariance corrections. Each required predicted covariance
must be PD; singular smoothing systems fail, with no pseudoinverse or jitter.
Smoothing uses all supplied observations and is not an online estimate. Exact
PSD ranks and PSD checks on the returned rounded state covariance matrices are both
reported: entry rounding can destroy PSD, and no silent repair is performed.

Bounds: <=32 steps, state/measurement dimensions <=4, finite scalar magnitudes
<=1e6 and <=16384 sum-of-cubed combined dimension units. Nonzero binary64 output
underflow and overflow fail. Model, noise, ordering and PIT assumptions are not
verified. No causal availability or real-world accuracy is established.
Filtering/smoothing formulas: Särkkä, Bayesian Filtering and Smoothing (2013),
sections 4.3 and 8.2, https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e6, le=1e6)]
Values = Annotated[list[Value], Field(min_length=1, max_length=4)]
Array = Annotated[list[Values], Field(min_length=1, max_length=4)]
type Vector = list[Fraction]
type Matrix = list[list[Fraction]]


class Step(InputModel):
    transition: Array
    state_offset: Values
    process_covariance: Array
    observation_matrix: Array
    observation_offset: Values
    observation_covariance: Array
    observation: list[Value | None] = Field(min_length=1, max_length=4)


class Input(InputModel):
    initial_mean: Values
    initial_covariance: Array
    steps: list[Step] = Field(min_length=1, max_length=32)
    smooth: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def aligned_shapes(self) -> Self:
        dimension = len(self.initial_mean)

        def shape(matrix: list[list[float]], rows: int, columns: int) -> bool:
            return len(matrix) == rows and all(len(row) == columns for row in matrix)

        if not shape(self.initial_covariance, dimension, dimension):
            raise ValueError("initial covariance must align with the initial state")
        work = 0
        for step in self.steps:
            measured = len(step.observation)
            work += (dimension + measured) ** 3
            if (
                not shape(step.transition, dimension, dimension)
                or not shape(step.process_covariance, dimension, dimension)
                or len(step.state_offset) != dimension
                or not shape(step.observation_matrix, measured, dimension)
                or len(step.observation_offset) != measured
                or not shape(step.observation_covariance, measured, measured)
            ):
                raise ValueError("step arrays must match their state and measurement dimensions")
        if work > 16_384:
            raise ValueError("linear Gaussian model exceeds 16384 cubic dimension-work units")
        return self


class Estimate(OutputModel):
    mean: list[float]
    covariance: list[list[float]]
    exact_covariance_rank: int
    rounded_covariance_is_positive_semidefinite: bool


class StepResult(OutputModel):
    source_step_index: int
    status: Literal["updated", "prediction_only"]
    observed_coordinate_indices: list[int]
    predicted: Estimate
    filtered: Estimate
    smoothed: Estimate | None
    predicted_observation_mean: list[float]
    innovation: list[float]
    innovation_covariance: list[list[float]]
    rounded_innovation_covariance_is_positive_definite: bool | None
    kalman_gain: list[list[float]]
    squared_mahalanobis_innovation: float | None


class Output(OutputModel):
    state_dimension: int
    step_count: int
    observed_coordinate_count: int
    completely_missing_step_count: int
    initial_prior: Estimate
    initial_prior_state_index: Literal[-1] = -1
    steps: list[StepResult]
    smoothing_applied: bool
    smoothed_estimates_use_all_supplied_observations: bool
    covariance_repairs_applied: Literal[False] = False
    recursion_arithmetic: Literal["bounded_exact_binary_input_rationals"] = (
        "bounded_exact_binary_input_rationals"
    )
    gaussian_independent_noise_assumptions_verified: Literal[False] = False
    model_parameters_fitted: Literal[False] = False
    timing_verified: Literal[False] = False


class _CovarianceError(ValueError):
    """A symmetric covariance violates the declared positivity requirement."""


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 32_768:
        raise ValueError("linear Gaussian arithmetic exceeds 32768-bit budget")
    return value


def _number(value: Fraction) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("linear Gaussian output overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError("nonzero linear Gaussian output is outside finite binary64 range")
    return result


def _fractions(matrix: list[list[float]]) -> Matrix:
    return [[Fraction(value) for value in row] for row in matrix]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix, strict=True)]


def _dot(left: Vector, right: Vector) -> Fraction:
    total = Fraction()
    for first, second in zip(left, right, strict=True):
        total = _bounded(total + _bounded(first * second))
    return total


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    columns = _transpose(right)
    return [[_dot(row, column) for column in columns] for row in left]


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [_dot(row, vector) for row in matrix]


def _add(left: Matrix, right: Matrix, sign: int = 1) -> Matrix:
    return [
        [_bounded(a + sign * b) for a, b in zip(row, other, strict=True)]
        for row, other in zip(left, right, strict=True)
    ]


def _ldl(matrix: Matrix, positive_definite: bool, label: str) -> tuple[Matrix, Vector]:
    size = len(matrix)
    if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(i)):
        raise _CovarianceError(f"{label} must be exactly symmetric")
    lower = [[Fraction(i == j) for j in range(size)] for i in range(size)]
    diagonal: Vector = []
    for column in range(size):
        pivot = matrix[column][column]
        for k in range(column):
            pivot = _bounded(pivot - _bounded(lower[column][k] ** 2 * diagonal[k]))
        if pivot < 0 or (positive_definite and pivot == 0):
            requirement = "positive definite" if positive_definite else "positive semidefinite"
            raise _CovarianceError(f"{label} must be {requirement}")
        diagonal.append(pivot)
        for row in range(column + 1, size):
            residual = matrix[row][column]
            for k in range(column):
                residual = _bounded(
                    residual - _bounded(lower[row][k] * lower[column][k] * diagonal[k])
                )
            if pivot:
                lower[row][column] = _bounded(residual / pivot)
            elif residual:
                raise _CovarianceError(f"{label} has a nonzero off-diagonal at a zero PSD pivot")
    return lower, diagonal


def _solve_pd(matrix: Matrix, right: Matrix, label: str) -> Matrix:
    lower, diagonal = _ldl(matrix, True, label)
    size, columns = len(matrix), len(right[0])
    result = [[Fraction() for _ in range(columns)] for _ in range(size)]
    for column in range(columns):
        forward: Vector = []
        for row in range(size):
            forward.append(_bounded(right[row][column] - _dot(lower[row][:row], forward)))
        solution = [Fraction()] * size
        for row in range(size - 1, -1, -1):
            solution[row] = _bounded(
                forward[row] / diagonal[row]
                - _dot([lower[j][row] for j in range(row + 1, size)], solution[row + 1 :])
            )
        for row, value in enumerate(solution):
            result[row][column] = value
    return result


def _estimate(mean: Vector, covariance: Matrix) -> Estimate:
    _, diagonal = _ldl(covariance, False, "computed covariance")
    rounded = [[_number(value) for value in row] for row in covariance]
    rounded_psd = True
    try:
        _ldl(_fractions(rounded), False, "rounded covariance")
    except _CovarianceError:
        rounded_psd = False
    return Estimate(
        mean=[_number(value) for value in mean],
        covariance=rounded,
        exact_covariance_rank=sum(value > 0 for value in diagonal),
        rounded_covariance_is_positive_semidefinite=rounded_psd,
    )


def _rounded_innovation_pd(matrix: list[list[float]]) -> bool | None:
    if not matrix:
        return None
    try:
        _ldl(_fractions(matrix), True, "rounded innovation covariance")
    except _CovarianceError:
        return False
    return True


def execute(request: Input, context: OperationContext) -> Output:
    dimension = len(request.initial_mean)
    mean = [Fraction(value) for value in request.initial_mean]
    covariance = _fractions(request.initial_covariance)
    prior = _estimate(mean, covariance)
    predictions: list[tuple[Vector, Matrix]] = []
    filters: list[tuple[Vector, Matrix]] = []
    transitions: list[Matrix] = []
    results: list[StepResult] = []
    observed_count = missing_count = 0
    for index, step in enumerate(request.steps):
        transition = _fractions(step.transition)
        process = _fractions(step.process_covariance)
        measurement = _fractions(step.observation_matrix)
        noise = _fractions(step.observation_covariance)
        _ldl(process, False, f"process covariance at step {index}")
        _ldl(noise, True, f"measurement covariance at step {index}")
        transitions.append(transition)
        predicted_mean = [
            _bounded(value + Fraction(offset))
            for value, offset in zip(_matvec(transition, mean), step.state_offset, strict=True)
        ]
        predicted_covariance = _add(
            _multiply(_multiply(transition, covariance), _transpose(transition)), process
        )
        predictions.append((predicted_mean, predicted_covariance))
        predicted_observation = [
            _bounded(value + Fraction(offset))
            for value, offset in zip(
                _matvec(measurement, predicted_mean), step.observation_offset, strict=True
            )
        ]
        observed = [
            coordinate for coordinate, value in enumerate(step.observation) if value is not None
        ]
        observed_count += len(observed)
        innovation: Vector = []
        innovation_covariance: Matrix = []
        gain: Matrix = [[] for _ in range(dimension)]
        mahalanobis: Fraction | None = None
        if observed:
            selected_measurement = [measurement[coordinate] for coordinate in observed]
            selected_noise = [[noise[i][j] for j in observed] for i in observed]
            for coordinate in observed:
                value = step.observation[coordinate]
                if value is None:
                    raise ValueError("observed coordinate unexpectedly has no value")
                innovation.append(_bounded(Fraction(value) - predicted_observation[coordinate]))
            cross_covariance = _multiply(predicted_covariance, _transpose(selected_measurement))
            innovation_covariance = _add(
                _multiply(selected_measurement, cross_covariance), selected_noise
            )
            gain = _transpose(
                _solve_pd(
                    innovation_covariance,
                    _transpose(cross_covariance),
                    f"innovation covariance at step {index}",
                )
            )
            correction = _matvec(gain, innovation)
            mean = [
                _bounded(value + change)
                for value, change in zip(predicted_mean, correction, strict=True)
            ]
            covariance = _add(
                predicted_covariance,
                _multiply(_multiply(gain, innovation_covariance), _transpose(gain)),
                -1,
            )
            whitened = _solve_pd(
                innovation_covariance,
                [[value] for value in innovation],
                f"innovation covariance at step {index}",
            )
            mahalanobis = _dot(innovation, [row[0] for row in whitened])
            if mahalanobis < 0:
                raise ValueError("squared innovation norm became negative")
        else:
            mean, covariance = predicted_mean, predicted_covariance
            missing_count += 1
        filters.append((mean, covariance))
        rounded_innovation = [[_number(value) for value in row] for row in innovation_covariance]
        results.append(
            StepResult(
                source_step_index=index,
                status="updated" if observed else "prediction_only",
                observed_coordinate_indices=observed,
                predicted=_estimate(predicted_mean, predicted_covariance),
                filtered=_estimate(mean, covariance),
                smoothed=None,
                predicted_observation_mean=[_number(value) for value in predicted_observation],
                innovation=[_number(value) for value in innovation],
                innovation_covariance=rounded_innovation,
                rounded_innovation_covariance_is_positive_definite=_rounded_innovation_pd(
                    rounded_innovation
                ),
                kalman_gain=[[_number(value) for value in row] for row in gain],
                squared_mahalanobis_innovation=None
                if mahalanobis is None
                else _number(mahalanobis),
            )
        )
    if request.smooth:
        smoothed_mean, smoothed_covariance = filters[-1]
        results[-1].smoothed = _estimate(smoothed_mean, smoothed_covariance)
        for index in range(len(request.steps) - 2, -1, -1):
            filtered_mean, filtered_covariance = filters[index]
            next_mean, next_covariance = predictions[index + 1]
            cross = _multiply(filtered_covariance, _transpose(transitions[index + 1]))
            smoother_gain = _transpose(
                _solve_pd(
                    next_covariance,
                    _transpose(cross),
                    f"smoothing predicted covariance at step {index + 1}",
                )
            )
            correction = _matvec(
                smoother_gain,
                [_bounded(a - b) for a, b in zip(smoothed_mean, next_mean, strict=True)],
            )
            smoothed_mean = [
                _bounded(a + b) for a, b in zip(filtered_mean, correction, strict=True)
            ]
            smoothed_covariance = _add(
                filtered_covariance,
                _multiply(
                    _multiply(smoother_gain, _add(smoothed_covariance, next_covariance, -1)),
                    _transpose(smoother_gain),
                ),
            )
            results[index].smoothed = _estimate(smoothed_mean, smoothed_covariance)
    return Output(
        state_dimension=dimension,
        step_count=len(results),
        observed_coordinate_count=observed_count,
        completely_missing_step_count=missing_count,
        initial_prior=prior,
        steps=results,
        smoothing_applied=request.smooth,
        smoothed_estimates_use_all_supplied_observations=request.smooth,
    )


OPERATION = Operation(
    id="features.linear_gaussian_filter",
    kind="feature",
    description="Filter supplied time-varying linear Gaussian states with missing coordinates and optional exact-rational RTS smoothing.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
