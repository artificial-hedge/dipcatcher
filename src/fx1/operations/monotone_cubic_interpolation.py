"""Shape-preserving cubic Hermite reconstruction on an explicit knot grid.

Own weighted-harmonic interior slopes and one-sided limited endpoint slopes
follow Fritsch-Butland PCHIP. Normalized interval polynomials are evaluated and
integrated with exact rational arithmetic on supplied binary floats. Outputs
are rounded previews with explicit underflow; no extrapolation is permitted.
This is whole-grid reconstruction, not a causal forecast or availability audit.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html
"""

from __future__ import annotations

import math
from bisect import bisect_right
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Scalar = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]


class Input(InputModel):
    x: list[Scalar] = Field(min_length=2, max_length=256)
    y: list[Scalar] = Field(min_length=2, max_length=256)
    queries: list[Scalar] = Field(default_factory=list, max_length=512)
    integral_bounds: list[tuple[Scalar, Scalar]] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def grid(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        if any(right <= left for left, right in zip(self.x, self.x[1:], strict=False)):
            raise ValueError("x must be strictly increasing without duplicate knots")
        if any(not self.x[0] <= query <= self.x[-1] for query in self.queries):
            raise ValueError("every query must lie inside the supplied knot grid")
        if any(
            not self.x[0] <= endpoint <= self.x[-1]
            for bounds in self.integral_bounds
            for endpoint in bounds
        ):
            raise ValueError("integral endpoints must lie inside the supplied knot grid")
        return self


class Preview(OutputModel):
    value: float
    underflow: bool


class Evaluation(OutputModel):
    query_row: int
    interval_index: int
    knot_index: int | None
    value: Preview
    first_derivative: Preview
    second_derivative: Preview


class Integral(OutputModel):
    bounds_row: int
    signed_integral: Preview
    intervals_touched: int


class Output(OutputModel):
    knot_count: int
    knot_slopes: list[Preview]
    evaluations: list[Evaluation]
    integrals: list[Integral]
    coefficient_arithmetic: Literal["exact_supplied_binary_floats"] = "exact_supplied_binary_floats"
    returned_values: Literal["rounded_binary64_with_nonzero_underflow_flags"] = (
        "rounded_binary64_with_nonzero_underflow_flags"
    )
    knot_second_derivative_side: Literal["right_except_final_knot_left"] = (
        "right_except_final_knot_left"
    )
    extrapolation: Literal["rejected"] = "rejected"
    availability_validated: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("calculation exceeds the 12000-bit rational arithmetic budget")
    return value


def _preview(value: Fraction) -> Preview:
    _bounded(value)
    try:
        preview = float(value)
    except OverflowError as error:
        raise ValueError("requested output exceeds finite binary64 range") from error
    if not math.isfinite(preview):
        raise ValueError("requested output exceeds finite binary64 range")
    return Preview(value=preview, underflow=bool(value) and preview == 0)


def _endpoint(first_h: Fraction, next_h: Fraction, first_d: Fraction, next_d: Fraction) -> Fraction:
    derivative = _bounded(
        ((2 * first_h + next_h) * first_d - first_h * next_d) / (first_h + next_h)
    )
    if derivative * first_d <= 0:
        return Fraction()
    if first_d * next_d <= 0 and abs(derivative) > 3 * abs(first_d):
        return 3 * first_d
    return derivative


def _primitive(coefficients: list[Fraction], position: Fraction) -> Fraction:
    result = Fraction()
    for degree, coefficient in enumerate(coefficients):
        result = _bounded(result + coefficient * position ** (degree + 1) / (degree + 1))
    return result


def execute(request: Input, context: OperationContext) -> Output:
    x = [Fraction(value) for value in request.x]
    y = [Fraction(value) for value in request.y]
    widths = [right - left for left, right in zip(x, x[1:], strict=False)]
    secants = [_bounded((y[index + 1] - y[index]) / width) for index, width in enumerate(widths)]
    slopes = [Fraction() for _ in x]
    if len(x) == 2:
        slopes = [secants[0], secants[0]]
    else:
        slopes[0] = _endpoint(widths[0], widths[1], secants[0], secants[1])
        slopes[-1] = _endpoint(widths[-1], widths[-2], secants[-1], secants[-2])
        for index in range(1, len(x) - 1):
            previous, following = secants[index - 1], secants[index]
            if previous * following <= 0:
                continue
            first_weight = 2 * widths[index] + widths[index - 1]
            second_weight = widths[index] + 2 * widths[index - 1]
            slopes[index] = _bounded(
                (first_weight + second_weight)
                / (first_weight / previous + second_weight / following)
            )
    polynomials: list[list[Fraction]] = []
    for index, width in enumerate(widths):
        delta = y[index + 1] - y[index]
        polynomials.append(
            [
                y[index],
                _bounded(width * slopes[index]),
                _bounded(3 * delta - width * (2 * slopes[index] + slopes[index + 1])),
                _bounded(-2 * delta + width * (slopes[index] + slopes[index + 1])),
            ]
        )
    lookup = {value: index for index, value in enumerate(x)}
    evaluations: list[Evaluation] = []
    for row, supplied in enumerate(request.queries):
        query = Fraction(supplied)
        index = min(bisect_right(x, query) - 1, len(widths) - 1)
        width = widths[index]
        position = _bounded((query - x[index]) / width)
        constant, linear, square, cube = polynomials[index]
        value = _bounded(constant + position * (linear + position * (square + position * cube)))
        derivative = _bounded((linear + position * (2 * square + 3 * position * cube)) / width)
        second = _bounded((2 * square + 6 * position * cube) / width**2)
        evaluations.append(
            Evaluation(
                query_row=row,
                interval_index=index,
                knot_index=lookup.get(query),
                value=_preview(value),
                first_derivative=_preview(derivative),
                second_derivative=_preview(second),
            )
        )
    integrals: list[Integral] = []
    for row, (first, last) in enumerate(request.integral_bounds):
        lower, upper = sorted((Fraction(first), Fraction(last)))
        integral = Fraction()
        touched = 0
        for index, width in enumerate(widths):
            left, right = max(lower, x[index]), min(upper, x[index + 1])
            if left >= right:
                continue
            start = _bounded((left - x[index]) / width)
            stop = _bounded((right - x[index]) / width)
            increment = _bounded(
                width
                * (_primitive(polynomials[index], stop) - _primitive(polynomials[index], start))
            )
            integral = _bounded(integral + increment)
            touched += 1
        integrals.append(
            Integral(
                bounds_row=row,
                signed_integral=_preview(integral if first <= last else -integral),
                intervals_touched=touched,
            )
        )
    return Output(
        knot_count=len(x),
        knot_slopes=[_preview(value) for value in slopes],
        evaluations=evaluations,
        integrals=integrals,
    )


OPERATION = Operation(
    id="features.monotone_cubic_interpolation",
    kind="feature",
    description="Reconstruct a shape-preserving cubic Hermite curve with independently computed limited slopes, exact rational query derivatives and signed interval integrals, rejecting extrapolation.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
