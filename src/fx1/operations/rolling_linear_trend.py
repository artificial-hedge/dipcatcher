"""Causal trailing OLS fits against a centered observation index.

The fitted coordinate is x[j] = j - (window-1)/2 for j=0,...,window-1.
The intercept is the fitted level at the window midpoint, not its first row.
Residual scale is sqrt(RSS/(window-2)); R² is undefined for constant windows.
Rows must already be ordered and PIT-selected. Slopes are per observation,
not per elapsed time; no uncertainty or significance claim is attached.
"""

from math import fsum, sqrt
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Status = Literal["warmup", "constant", "finite"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    window: int = Field(default=20, strict=True, ge=3, le=1_000)


class Output(OutputModel):
    slopes_per_observation: list[Finite | None] = Field(max_length=10_000)
    centered_intercepts: list[Finite | None] = Field(max_length=10_000)
    residual_scales: list[Nonnegative | None] = Field(max_length=10_000)
    r_squared: list[Fraction | None] = Field(max_length=10_000)
    status: list[Status] = Field(max_length=10_000)
    window: int
    residual_degrees_of_freedom: int


def execute(request: Input, context: OperationContext) -> Output:
    """Fit full windows using scaled offsets to protect small and large inputs."""
    coordinate = [index - (request.window - 1) / 2.0 for index in range(request.window)]
    coordinate_square_sum = fsum(value * value for value in coordinate)
    slopes: list[float | None] = []
    intercepts: list[float | None] = []
    residual_scales: list[float | None] = []
    r_squared: list[float | None] = []
    statuses: list[Status] = []
    for index in range(len(request.values)):
        if index + 1 < request.window:
            slopes.append(None)
            intercepts.append(None)
            residual_scales.append(None)
            r_squared.append(None)
            statuses.append("warmup")
            continue
        trailing = request.values[index + 1 - request.window : index + 1]
        anchor = trailing[0]
        offsets = [value - anchor for value in trailing]
        scale = max(abs(value) for value in offsets)
        if scale == 0.0:
            slopes.append(0.0)
            intercepts.append(anchor)
            residual_scales.append(0.0)
            r_squared.append(None)
            statuses.append("constant")
            continue
        normalized = [value / scale for value in offsets]
        mean = fsum(normalized) / request.window
        centered = [value - mean for value in normalized]
        slope = (
            fsum(x * y for x, y in zip(coordinate, centered, strict=True)) / coordinate_square_sum
        )
        residual_sum = fsum((y - slope * x) ** 2 for x, y in zip(coordinate, centered, strict=True))
        total_sum = fsum(value * value for value in centered)
        slopes.append(slope * scale)
        intercepts.append(anchor + mean * scale)
        residual_scales.append(sqrt(residual_sum / (request.window - 2)) * scale)
        r_squared.append(min(1.0, max(0.0, 1.0 - residual_sum / total_sum)))
        statuses.append("finite")
    return Output(
        slopes_per_observation=slopes,
        centered_intercepts=intercepts,
        residual_scales=residual_scales,
        r_squared=r_squared,
        status=statuses,
        window=request.window,
        residual_degrees_of_freedom=request.window - 2,
    )


OPERATION = Operation(
    id="features.rolling_linear_trend",
    kind="feature",
    description=(
        "Fit trailing OLS trends against centered observation indices, reporting slopes, "
        "midpoint intercepts, sqrt(RSS/(window-2)) residual scales and R-squared. Warmup and "
        "constant windows are explicit; caller supplies ordered PIT-selected values."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
