"""Planar thin-plate reconstruction with an affine tail and supplied smoothing.

Coordinates are centered at the first fitting point and divided by one explicit
length scale. Own kernel assembly uses phi(r)=r² log(r), phi(0)=0, and a full
affine tail. The saddle system [K+lambda*I,P; P',0] is solved with own rational
elimination on rounded binary64 kernel entries and exact normalized coordinates.
This is approximate kernel arithmetic, not an exact transcendental solution.

Returned coefficients are rounded to binary64; fitting/query values and system
residuals are recomputed from those returned coefficients. Gradients use the
analytic kernel derivative with approximate logarithms and original-coordinate
units. They are not certified derivative error bounds. Noncollinear, distinct
centers are required; normalized center separation must be at least1e-12.
Extrapolation is allowed explicitly by the global radial model, with no spatial
validity or forecast claim. No automatic coordinate rescaling or smoothing fit.

Bounds:3..24 centers,256 queries, input coordinates/values magnitudes<=1e6,
length scale1e-6..1e6, smoothing0..1e6 and32768-bit rational intermediates.
All requested outputs must be finite; nonzero rounded underflow is flagged.
System and affine moment residuals expose coefficient rounding, not conditioning.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Scalar = Annotated[float, Field(strict=True, ge=-1e6, le=1e6)]
Point = tuple[Scalar, Scalar]
ExactPoint = tuple[Fraction, Fraction]


class Input(InputModel):
    points: list[Point] = Field(min_length=3, max_length=24)
    values: list[Scalar] = Field(min_length=3, max_length=24)
    queries: list[Point] = Field(default_factory=list, max_length=256)
    length_scale: float = Field(default=1.0, strict=True, ge=1e-6, le=1e6)
    smoothing: float = Field(default=0.0, strict=True, ge=0, le=1e6)

    @model_validator(mode="after")
    def aligned(self) -> Self:
        if len(self.points) != len(self.values):
            raise ValueError("values must align with fitting points")
        if len(set(self.points)) != len(self.points):
            raise ValueError("fitting centers must be distinct")
        return self


class Preview(OutputModel):
    value: float
    underflow: bool


class Evaluation(OutputModel):
    source_row: int
    value: Preview
    coordinate_gradient: tuple[Preview, Preview]


class Output(OutputModel):
    center_count: int
    coordinate_origin: tuple[float, float]
    length_scale: float
    smoothing: float
    radial_coefficients: list[Preview]
    affine_coefficients_constant_x_y: list[Preview]
    fitted_values: list[Preview]
    fit_residuals: list[Preview]
    maximum_absolute_fit_residual: Preview
    maximum_absolute_returned_system_residual: Preview
    returned_affine_moment_residuals: list[Preview]
    maximum_absolute_coefficient_rounding: Preview
    queries: list[Evaluation]
    kernel_evaluations: int
    logarithm_underflow_count: int
    kernel_underflow_count: int
    exact_rounded_kernel_system_solve_checked: Literal[True] = True
    evaluations_use_returned_coefficients: Literal[True] = True
    transcendental_kernel_exact: Literal[False] = False
    conditioning_or_error_bound_verified: Literal[False] = False
    availability_or_extrapolation_validity_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 32_768:
        raise ValueError("thin-plate arithmetic exceeds the 32768-bit rational budget")
    return value


def _preview(value: Fraction) -> Preview:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("thin-plate result exceeds finite binary64 range") from error
    if not math.isfinite(result):
        raise ValueError("thin-plate result exceeds finite binary64 range")
    return Preview(value=result, underflow=bool(value) and result == 0)


def _log(value: Fraction) -> tuple[float, bool]:
    if Fraction(1, 2) <= value <= Fraction(3, 2):
        difference = value - 1
        preview = float(difference)
        return math.log1p(preview), bool(difference) and preview == 0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    mantissa = value / (1 << exponent) if exponent >= 0 else value * (1 << -exponent)
    return math.log(float(mantissa)) + exponent * math.log(2), False


def _kernel(square: Fraction, counters: list[int]) -> tuple[Fraction, Fraction]:
    counters[0] += 1
    if not square:
        return Fraction(), Fraction()
    logarithm, underflow = _log(square)
    counters[1] += underflow
    approximate = _bounded(square * Fraction(logarithm) / 2)
    preview = _preview(approximate)
    counters[2] += preview.underflow or underflow
    return Fraction(preview.value), Fraction(logarithm)


def _dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    result = Fraction()
    for first, second in zip(left, right, strict=True):
        result = _bounded(result + _bounded(first * second))
    return result


def _solve(matrix: list[list[Fraction]], right: list[Fraction]) -> list[Fraction]:
    size = len(matrix)
    rows = [row.copy() + [value] for row, value in zip(matrix, right, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if not rows[pivot][column]:
            raise ValueError("rounded thin-plate kernel system is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [_bounded(value / divisor) for value in rows[column]]
        for row in range(size):
            if row == column or not rows[row][column]:
                continue
            multiplier = rows[row][column]
            rows[row] = [
                _bounded(value - multiplier * source)
                for value, source in zip(rows[row], rows[column], strict=True)
            ]
    solution = [row[-1] for row in rows]
    if any(_dot(row, solution) != expected for row, expected in zip(matrix, right, strict=True)):
        raise ArithmeticError("rounded-kernel system failed exact rational residual verification")
    return solution


def execute(request: Input, context: OperationContext) -> Output:
    origin = request.points[0]
    scale = Fraction(request.length_scale)

    def normalize(point: Point) -> ExactPoint:
        return (
            (Fraction(point[0]) - Fraction(origin[0])) / scale,
            (Fraction(point[1]) - Fraction(origin[1])) / scale,
        )

    centers = [normalize(point) for point in request.points]
    if not any(centers[1][0] * point[1] - centers[1][1] * point[0] for point in centers[2:]):
        raise ValueError("thin-plate affine tail requires noncollinear fitting centers")
    count = len(centers)
    system = [[Fraction() for _ in range(count + 3)] for _ in range(count + 3)]
    kernels = [[Fraction() for _ in centers] for _ in centers]
    counters = [0, 0, 0]
    for row, center in enumerate(centers):
        system[row][row] = Fraction(request.smoothing)
        for column in range(row + 1, count):
            other = centers[column]
            square = (center[0] - other[0]) ** 2 + (center[1] - other[1]) ** 2
            if square < Fraction(1, 10**24):
                raise ValueError("normalized fitting centers must be separated by at least1e-12")
            value, _ = _kernel(square, counters)
            kernels[row][column] = kernels[column][row] = value
            system[row][column] = system[column][row] = value
        for column, value in enumerate((Fraction(1), *center)):
            system[row][count + column] = system[count + column][row] = value
    data = [Fraction(value) for value in request.values]
    right = data + [Fraction()] * 3
    exact_coefficients = _solve(system, right)
    previews = [_preview(value) for value in exact_coefficients]
    coefficients = [Fraction(preview.value) for preview in previews]
    radial, affine = coefficients[:count], coefficients[count:]
    rounding = max(
        abs(value - returned)
        for value, returned in zip(exact_coefficients, coefficients, strict=True)
    )
    system_residuals = [
        _bounded(_dot(row, coefficients) - expected)
        for row, expected in zip(system, right, strict=True)
    ]
    fitted = [
        _bounded(_dot(kernels[row], radial) + _dot([Fraction(1), *center], affine))
        for row, center in enumerate(centers)
    ]
    residuals = [value - expected for value, expected in zip(fitted, data, strict=True)]
    evaluations: list[Evaluation] = []
    for row, supplied in enumerate(request.queries):
        point = normalize(supplied)
        value = _dot([Fraction(1), *point], affine)
        gradient = [affine[1], affine[2]]
        for center, coefficient in zip(centers, radial, strict=True):
            delta = [point[0] - center[0], point[1] - center[1]]
            square = delta[0] ** 2 + delta[1] ** 2
            kernel, logarithm = _kernel(square, counters)
            value = _bounded(value + coefficient * kernel)
            if square:
                for axis in range(2):
                    gradient[axis] = _bounded(
                        gradient[axis] + coefficient * delta[axis] * (logarithm + 1)
                    )
        evaluations.append(
            Evaluation(
                source_row=row,
                value=_preview(value),
                coordinate_gradient=(_preview(gradient[0] / scale), _preview(gradient[1] / scale)),
            )
        )
    return Output(
        center_count=count,
        coordinate_origin=origin,
        length_scale=request.length_scale,
        smoothing=request.smoothing,
        radial_coefficients=previews[:count],
        affine_coefficients_constant_x_y=previews[count:],
        fitted_values=[_preview(value) for value in fitted],
        fit_residuals=[_preview(value) for value in residuals],
        maximum_absolute_fit_residual=_preview(max(map(abs, residuals))),
        maximum_absolute_returned_system_residual=_preview(max(map(abs, system_residuals))),
        returned_affine_moment_residuals=[_preview(value) for value in system_residuals[count:]],
        maximum_absolute_coefficient_rounding=_preview(rounding),
        queries=evaluations,
        kernel_evaluations=counters[0],
        logarithm_underflow_count=counters[1],
        kernel_underflow_count=counters[2],
    )


OPERATION = Operation(
    id="features.thin_plate_spline",
    kind="feature",
    description="Fit a bounded planar thin-plate radial model with an affine tail and supplied smoothing, exposing returned-coefficient residuals, query gradients and approximate-kernel limits.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
