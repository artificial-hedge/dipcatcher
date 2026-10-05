"""Hayashi-Yoshida overlap sum for two supplied asynchronous observed paths.

For successive observations form increments dX_i and intervals (t_(i-1),t_i].
Return sum dX_i*dY_j over pairs with max(starts)<min(ends). Intervals merely
touching at an endpoint do not overlap under this convention. Each pair has
unit weight, regardless of overlap duration. No interpolation, demeaning,
division by elapsed time or annualization is performed.

Both paths must have observed values at the same initial and final timestamps;
interior clocks can differ. This avoids silently trimming boundary increments.
price inputs produce price-difference products in the product of supplied units;
log_price inputs must already be logarithms and produce log-return products.
No logarithm is applied here. Input binary64 values are converted to exact
integer units before increments and products; only the sum is rounded. A
nonzero result that rounds to zero or infinity fails explicitly.

The caller must enforce availability and sampling assumptions. This is a
research feature, without a noise correction, standard error, or market claim.
Reference: Hayashi and Yoshida (2008), definition 1, equation (3):
https://www.ism.ac.jp/editsec/aism/pdf/060_2_0367.pdf
"""

from datetime import UTC, datetime
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Semantics = Literal["price", "log_price"]


class Observation(InputModel):
    time: AwareDatetime
    value: Value

    @field_validator("time", mode="before")
    @classmethod
    def validate_clock(cls, value: object) -> datetime:
        if isinstance(value, str) and len(value) <= 64:
            parsed = datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError(
                "time must be an aware datetime or ISO string of at most 64 characters"
            )
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("time must have an explicit timezone offset")
        try:
            return parsed.astimezone(UTC)
        except OverflowError as error:
            raise ValueError("time is outside the supported UTC range") from error


class Input(InputModel):
    x: list[Observation] = Field(min_length=2, max_length=4_096)
    y: list[Observation] = Field(min_length=2, max_length=4_096)
    value_semantics: Semantics = "log_price"
    x_unit: str = Field(default="supplied_x_unit", strict=True, min_length=1, max_length=64)
    y_unit: str = Field(default="supplied_y_unit", strict=True, min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        for path in (self.x, self.y):
            if any(right.time <= left.time for left, right in pairwise(path)):
                raise ValueError("each path must have strictly increasing observation clocks")
            if self.value_semantics == "price" and any(row.value <= 0 for row in path):
                raise ValueError("price mode requires positive observed prices")
        if self.x[0].time != self.y[0].time or self.x[-1].time != self.y[-1].time:
            raise ValueError(
                "paths require the same observed initial and final clocks; no interpolation"
            )
        return self


class Output(OutputModel):
    covariance_sum: float = Field(allow_inf_nan=False)
    value_semantics: Semantics
    x_unit: str
    y_unit: str
    output_unit_convention: Literal["product_of_supplied_value_units"] = (
        "product_of_supplied_value_units"
    )
    window_start: AwareDatetime
    window_end: AwareDatetime
    duration_microseconds: int
    x_interval_count: int
    y_interval_count: int
    overlap_pair_count: int
    zero_product_pair_count: int
    simultaneous_right_endpoints: int
    interval_convention: Literal["left_open_right_closed"] = "left_open_right_closed"
    overlap_rule: Literal["strictly_positive_duration"] = "strictly_positive_duration"


def _increments(path: list[Observation]) -> tuple[list[int], int]:
    ratios = [row.value.as_integer_ratio() for row in path]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    return [right - left for left, right in pairwise(units)], places


def execute(request: Input, context: OperationContext) -> Output:
    """Advance the earlier interval endpoint and accumulate each overlap exactly once."""
    dx, x_places = _increments(request.x)
    dy, y_places = _increments(request.y)
    left = right = pairs = zeros = simultaneous = total = 0
    while left < len(dx) and right < len(dy):
        x_start, x_end = request.x[left].time, request.x[left + 1].time
        y_start, y_end = request.y[right].time, request.y[right + 1].time
        if max(x_start, y_start) < min(x_end, y_end):
            product = dx[left] * dy[right]
            total += product
            pairs += 1
            zeros += product == 0
        simultaneous += x_end == y_end
        if x_end <= y_end:
            left += 1
        if y_end <= x_end:
            right += 1
    exact = Fraction(total, 1 << (x_places + y_places))
    try:
        covariance = float(exact)
    except OverflowError as error:
        raise ValueError("covariance sum overflows binary64") from error
    if not isfinite(covariance) or (exact != 0 and covariance == 0):
        raise ValueError("nonzero covariance sum cannot be represented as finite binary64")
    duration = request.x[-1].time - request.x[0].time
    return Output(
        covariance_sum=covariance,
        value_semantics=request.value_semantics,
        x_unit=request.x_unit,
        y_unit=request.y_unit,
        window_start=request.x[0].time,
        window_end=request.x[-1].time,
        duration_microseconds=(duration.days * 86_400 + duration.seconds) * 1_000_000
        + duration.microseconds,
        x_interval_count=len(dx),
        y_interval_count=len(dy),
        overlap_pair_count=pairs,
        zero_product_pair_count=zeros,
        simultaneous_right_endpoints=simultaneous,
    )


OPERATION = Operation(
    id="features.hayashi_yoshida_covariance",
    kind="feature",
    description=(
        "Sum exact increment products over overlapping asynchronous observation intervals, "
        "with common observed boundaries, explicit units and no interpolation or annualization."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
