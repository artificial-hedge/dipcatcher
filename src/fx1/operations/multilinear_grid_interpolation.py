"""Exact tensor-hat interpolation and box integration on rectilinear grids.

Flat values use C ordering: the last axis varies fastest. Each axis is supplied
in strictly increasing order; no missing nodes or implicit spacing is allowed.
Queries use the right cell at an interior knot and the left cell at the final
knot. Optional clipping applies only to queries; derivatives of the composed
clipped function are zero along strictly clipped axes. At a boundary knot the
reported derivative is the chosen inward one-sided derivative.

Own tensor products evaluate corner weights and coordinate gradients exactly.
Box integrals separate products of integrated one-dimensional hat functions,
covering all intersected cells without numerical quadrature. Reversed endpoints
give oriented integrals; zero-width boxes have zero integral. Box endpoints must
be inside the grid regardless of query clipping policy. Final float previews
flag nonzero underflow and reject overflow. Grid fitting/PIT/units are not inferred.
Rectilinear convention: https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_regular_grid.html
"""

from __future__ import annotations

import math
from bisect import bisect_right
from fractions import Fraction
from itertools import product
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Scalar = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Point = Annotated[list[Scalar], Field(min_length=1, max_length=5)]
Bounds = Annotated[list[tuple[Scalar, Scalar]], Field(min_length=1, max_length=5)]


class Axis(InputModel):
    name: str = Field(strict=True, min_length=1, max_length=64)
    coordinates: list[Scalar] = Field(min_length=2, max_length=64)


class Input(InputModel):
    axes: list[Axis] = Field(min_length=1, max_length=5)
    values: list[Scalar] = Field(min_length=2, max_length=4096)
    queries: list[Point] = Field(default_factory=list, max_length=256)
    query_outside: Literal["reject", "clip"] = "reject"
    integral_boxes: list[Bounds] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def grid(self) -> Self:
        dimension = len(self.axes)
        if len({axis.name for axis in self.axes}) != dimension:
            raise ValueError("axis names must be unique")
        if math.prod(len(axis.coordinates) for axis in self.axes) != len(self.values):
            raise ValueError("values must contain every grid node in last-axis-fastest order")
        for axis in self.axes:
            if any(b <= a for a, b in zip(axis.coordinates, axis.coordinates[1:], strict=False)):
                raise ValueError("axis coordinates must be strictly increasing")
        for query in self.queries:
            if len(query) != dimension:
                raise ValueError("query dimension must match the axis count")
            if self.query_outside == "reject" and any(
                not axis.coordinates[0] <= value <= axis.coordinates[-1]
                for axis, value in zip(self.axes, query, strict=True)
            ):
                raise ValueError("query falls outside the supplied grid")
        for box in self.integral_boxes:
            if len(box) != dimension or any(
                not axis.coordinates[0] <= endpoint <= axis.coordinates[-1]
                for axis, bounds in zip(self.axes, box, strict=True)
                for endpoint in bounds
            ):
                raise ValueError("box endpoints must match dimensions and lie inside the grid")
        return self


class Preview(OutputModel):
    value: float
    underflow: bool


class QueryResult(OutputModel):
    query_row: int
    evaluated_coordinates: list[float]
    clipped_axes: list[int]
    lower_cell_indices: list[int]
    corner_value_rows: list[int]
    corner_weights: list[Preview]
    interpolated_value: Preview
    coordinate_gradient: list[Preview]


class IntegralResult(OutputModel):
    box_row: int
    orientation: Literal[-1, 0, 1]
    contributing_node_count: int
    integral: Preview


class Output(OutputModel):
    axis_names: list[str]
    shape: list[int]
    strides: list[int]
    node_count: int
    queries: list[QueryResult]
    integrals: list[IntegralResult]
    arithmetic: Literal["bounded_exact_supplied_binary_floats"] = (
        "bounded_exact_supplied_binary_floats"
    )
    flattening: Literal["last_axis_fastest"] = "last_axis_fastest"
    field_fitted: Literal[False] = False
    availability_or_units_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("grid arithmetic exceeds the 12000-bit rational budget")
    return value


def _preview(value: Fraction) -> Preview:
    _bounded(value)
    try:
        preview = float(value)
    except OverflowError as error:
        raise ValueError("grid result exceeds finite binary64 range") from error
    if not math.isfinite(preview):
        raise ValueError("grid result exceeds finite binary64 range")
    return Preview(value=preview, underflow=bool(value) and preview == 0)


def _product(values: list[Fraction]) -> Fraction:
    result = Fraction(1)
    for value in values:
        result = _bounded(result * value)
    return result


def _hat_integrals(axis: list[Fraction], lower: Fraction, upper: Fraction) -> list[Fraction]:
    weights = [Fraction() for _ in axis]
    for index in range(len(axis) - 1):
        left, right = max(lower, axis[index]), min(upper, axis[index + 1])
        if left >= right:
            continue
        width = axis[index + 1] - axis[index]
        start, stop = (left - axis[index]) / width, (right - axis[index]) / width
        rising = _bounded(width * (stop**2 - start**2) / 2)
        weights[index] = _bounded(weights[index] + (right - left) - rising)
        weights[index + 1] = _bounded(weights[index + 1] + rising)
    if sum(weights, Fraction()) != upper - lower or any(value < 0 for value in weights):
        raise ArithmeticError("integrated hat functions failed exact interval conservation")
    return weights


def execute(request: Input, context: OperationContext) -> Output:
    axes = [[Fraction(value) for value in axis.coordinates] for axis in request.axes]
    values = [Fraction(value) for value in request.values]
    dimension = len(axes)
    shape = [len(axis) for axis in axes]
    strides = [math.prod(shape[index + 1 :]) for index in range(dimension)]
    queries: list[QueryResult] = []
    for row, original in enumerate(request.queries):
        point = [
            min(max(Fraction(value), axis[0]), axis[-1])
            for value, axis in zip(original, axes, strict=True)
        ]
        clipped = [index for index in range(dimension) if point[index] != Fraction(original[index])]
        lower = [
            min(bisect_right(axis, value) - 1, len(axis) - 2)
            for axis, value in zip(axes, point, strict=True)
        ]
        widths = [axis[index + 1] - axis[index] for axis, index in zip(axes, lower, strict=True)]
        positions = [
            _bounded((point[index] - axes[index][lower[index]]) / widths[index])
            for index in range(dimension)
        ]
        value = Fraction()
        gradient = [Fraction() for _ in axes]
        corner_rows: list[int] = []
        corner_weights: list[Fraction] = []
        for bits in product((0, 1), repeat=dimension):
            flat = sum((lower[index] + bit) * strides[index] for index, bit in enumerate(bits))
            factors = [
                positions[index] if bit else 1 - positions[index] for index, bit in enumerate(bits)
            ]
            weight = _product(factors)
            corner_rows.append(flat)
            corner_weights.append(weight)
            value = _bounded(value + weight * values[flat])
            for index, bit in enumerate(bits):
                if index in clipped:
                    continue
                derivative = _bounded(
                    _product(factors[:index] + factors[index + 1 :])
                    * (1 if bit else -1)
                    / widths[index]
                )
                gradient[index] = _bounded(gradient[index] + derivative * values[flat])
        if sum(corner_weights, Fraction()) != 1 or any(weight < 0 for weight in corner_weights):
            raise ArithmeticError("query corner weights failed exact convex conservation")
        queries.append(
            QueryResult(
                query_row=row,
                evaluated_coordinates=[float(value) for value in point],
                clipped_axes=clipped,
                lower_cell_indices=lower,
                corner_value_rows=corner_rows,
                corner_weights=[_preview(weight) for weight in corner_weights],
                interpolated_value=_preview(value),
                coordinate_gradient=[_preview(value) for value in gradient],
            )
        )
    integrals: list[IntegralResult] = []
    for row, box in enumerate(request.integral_boxes):
        orientation: Literal[-1, 0, 1] = 1
        weights_by_axis: list[list[Fraction]] = []
        for axis, (first, last) in zip(axes, box, strict=True):
            if first == last:
                orientation = 0
            elif first > last and orientation:
                orientation = -1 if orientation == 1 else 1
            lower_bound, upper_bound = sorted((Fraction(first), Fraction(last)))
            weights_by_axis.append(_hat_integrals(axis, lower_bound, upper_bound))
        active = [
            [index for index, value in enumerate(weights) if value] for weights in weights_by_axis
        ]
        integral = Fraction()
        contributing = 0
        for coordinate in product(*active):
            flat = sum(index * stride for index, stride in zip(coordinate, strides, strict=True))
            weight = _product(
                [weights_by_axis[axis][index] for axis, index in enumerate(coordinate)]
            )
            integral = _bounded(integral + weight * values[flat])
            contributing += 1
        integrals.append(
            IntegralResult(
                box_row=row,
                orientation=orientation,
                contributing_node_count=contributing,
                integral=_preview(integral * orientation),
            )
        )
    return Output(
        axis_names=[axis.name for axis in request.axes],
        shape=shape,
        strides=strides,
        node_count=len(values),
        queries=queries,
        integrals=integrals,
    )


OPERATION = Operation(
    id="features.multilinear_grid_interpolation",
    kind="feature",
    description="Evaluate a supplied rectilinear tensor grid with exact corner weights and gradients, and integrate oriented boxes by separable hat functions with explicit clipping and knot conventions.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
