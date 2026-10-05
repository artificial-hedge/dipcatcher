"""Order-two signature of a specified piecewise-linear lead-lag lift.

For each row transition x->y the lifted path visits (x,x), (y,x), (y,y):
all lead coordinates move together, then all lag coordinates move together.
Coordinates are [lead.0,...,lead.d-1,lag.0,...,lag.d-1]. If time_values is
supplied, strictly increasing numeric time is prepended to each base point,
and is lifted alongside space; its units and scale remain caller supplied.
Without it, row order is the only time parameter and there is no clock axis.

S2[i,j] denotes the ordered integral with coordinate i first, j second.
For segment increment v, Chen's update is S2+=S1 tensor v+v tensor v/2,
then S1+=v; S0=1. This returns a signature, not a log-signature. Translating
the path has no mathematical effect. Repeated points contribute zero.

Independent binary scaling of each coordinate lifts tiny increments before
products. Compensated sums combine Chen contributions; unrepresentable
intermediate nonzero terms are rejected. Scaled coefficients and exponents
are retained when physical coefficients underflow on output. Binary64
subtractions/products still incur rounding; no exact-arithmetic claim is made.
Caller supplies ordered, point-in-time-selected observations for the whole path.

References: Ni, definition 3.1 and lemma 3.2, https://arxiv.org/abs/1509.03346;
Chevyrev and Kormilitzin, Chen's identity and linear segments,
https://arxiv.org/abs/1603.03788, section 1.3.3.
"""

from itertools import pairwise
from math import frexp, fsum, ldexp
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Point = Annotated[list[Value], Field(min_length=1, max_length=6)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=2, max_length=14)]


class Input(InputModel):
    points: list[Point] = Field(min_length=1, max_length=512)
    time_values: list[Value] | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        dimension = len(self.points[0])
        if any(len(point) != dimension for point in self.points):
            raise ValueError("every point must have the same spatial dimension")
        if self.time_values is not None:
            if len(self.time_values) != len(self.points):
                raise ValueError("time_values must align with points")
            if any(right <= left for left, right in pairwise(self.time_values)):
                raise ValueError("time_values must be strictly increasing in supplied row order")
        return self


class Output(OutputModel):
    point_count: int
    spatial_dimension: int
    base_dimension: int
    signature_dimension: int
    segment_count: int
    time_augmented: bool
    coordinate_order: list[str] = Field(min_length=2, max_length=14)
    level_zero: Literal[1] = 1
    level_one: Vector
    level_two: list[Vector] = Field(min_length=2, max_length=14)
    coordinate_binary_exponents: list[int] = Field(min_length=2, max_length=14)
    scaled_level_one: Vector
    scaled_level_two: list[Vector] = Field(min_length=2, max_length=14)
    level_one_underflow_count: int
    level_two_underflow_count: int
    path_convention: Literal["lead_block_then_lag_block"] = "lead_block_then_lag_block"


def _nonzero_product(left: float, right: float, *, half: bool = False) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    left_mantissa, left_exponent = frexp(left)
    right_mantissa, right_exponent = frexp(right)
    result = ldexp(
        left_mantissa * right_mantissa,
        left_exponent + right_exponent - int(half),
    )
    if result == 0.0:
        raise ValueError("path dynamic range underflows a nonzero normalized signature term")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Accumulate linear-segment Chen terms directly, without a signature package."""
    points = [point[:] for point in request.points]
    names = [f"value_{index}" for index in range(len(points[0]))]
    if request.time_values is not None:
        points = [[time, *point] for time, point in zip(request.time_values, points, strict=True)]
        names.insert(0, "time")
    base_dimension = len(names)
    dimension = 2 * base_dimension
    increments = [
        [
            fsum((right_value, -left_value))
            for left_value, right_value in zip(left, right, strict=True)
        ]
        for left, right in pairwise(points)
    ]
    exponents = [
        frexp(max((abs(row[column]) for row in increments), default=0.0))[1]
        for column in range(base_dimension)
    ]
    normalized: list[list[float]] = []
    for row in increments:
        scaled = [ldexp(value, -exponent) for value, exponent in zip(row, exponents, strict=True)]
        if any(value != 0.0 and result == 0.0 for value, result in zip(row, scaled, strict=True)):
            raise ValueError(
                "path dynamic range loses a nonzero coordinate increment during scaling"
            )
        normalized.append(scaled)

    # Keeping bounded term lists lets fsum compensate across the whole path,
    # including late cancellation. There are at most 1022 segments, 14 axes,
    # and 196 matrix entries, so neither work nor storage can grow unbounded.
    first_terms: list[list[float]] = [[] for _ in range(dimension)]
    second_terms: list[list[list[float]]] = [
        [[] for _ in range(dimension)] for _ in range(dimension)
    ]
    zeros = [0.0] * base_dimension
    for row in normalized:
        for increment in (row + zeros, zeros + row):
            before = [fsum(terms) for terms in first_terms]
            for column, delta in enumerate(increment):
                if delta == 0.0:
                    continue
                for index in range(dimension):
                    cross = _nonzero_product(before[index], delta)
                    segment = _nonzero_product(increment[index], delta, half=True)
                    if cross:
                        second_terms[index][column].append(cross)
                    if segment:
                        second_terms[index][column].append(segment)
            for terms, delta in zip(first_terms, increment, strict=True):
                if delta:
                    terms.append(delta)

    scale_exponents = exponents + exponents
    first = [fsum(terms) for terms in first_terms]
    second = [[fsum(terms) for terms in row] for row in second_terms]
    restored_first = [
        ldexp(value, exponent) for value, exponent in zip(first, scale_exponents, strict=True)
    ]
    restored_second = [
        [
            ldexp(value, scale_exponents[row_index] + scale_exponents[column])
            for column, value in enumerate(row)
        ]
        for row_index, row in enumerate(second)
    ]
    return Output(
        point_count=len(points),
        spatial_dimension=len(request.points[0]),
        base_dimension=base_dimension,
        signature_dimension=dimension,
        segment_count=2 * len(increments),
        time_augmented=request.time_values is not None,
        coordinate_order=[f"lead.{name}" for name in names] + [f"lag.{name}" for name in names],
        level_one=restored_first,
        level_two=restored_second,
        coordinate_binary_exponents=scale_exponents,
        scaled_level_one=first,
        scaled_level_two=second,
        level_one_underflow_count=sum(
            normalized_value != 0.0 and value == 0.0
            for normalized_value, value in zip(first, restored_first, strict=True)
        ),
        level_two_underflow_count=sum(
            normalized_value != 0.0 and value == 0.0
            for normalized_row, row in zip(second, restored_second, strict=True)
            for normalized_value, value in zip(normalized_row, row, strict=True)
        ),
    )


OPERATION = Operation(
    id="features.lead_lag_signature",
    kind="feature",
    description=(
        "Compute level-one and level-two piecewise-linear signatures by Chen updates over "
        "lead-first, lag-second block moves. Supports optional strictly increasing numeric "
        "time augmentation, explicit coordinate order and scaled underflow-preserving output."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
