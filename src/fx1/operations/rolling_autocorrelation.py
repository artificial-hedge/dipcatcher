"""Pearson correlation of lagged pairs inside each complete trailing window.

For a window x[0:w], correlate x[0:w-lag] with x[lag:w]. Each pair vector
has its own mean and norm; this is not the alternative ACF estimator using a
single full-window mean and variance denominator. Rows must be ordered and
PIT-selected by the caller. No inferential significance is claimed.
"""

from math import fsum, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Correlation = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
Status = Literal["warmup", "zero_variance", "finite"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    window: int = Field(default=20, strict=True, ge=3, le=1_000)
    lag: int = Field(default=1, strict=True, ge=1, le=999)
    minimum_pairs: int = Field(default=2, strict=True, ge=2, le=999)

    @model_validator(mode="after")
    def validate_pair_count(self) -> Self:
        if self.window - self.lag < self.minimum_pairs:
            raise ValueError("window minus lag must provide at least minimum_pairs observations")
        return self


class Output(OutputModel):
    correlations: list[Correlation | None] = Field(max_length=10_000)
    status: list[Status] = Field(max_length=10_000)
    window: int
    lag: int
    pairs_per_full_window: int


def execute(request: Input, context: OperationContext) -> Output:
    """Center relative to an anchor and scale before calculating each norm."""
    correlations: list[float | None] = []
    statuses: list[Status] = []
    pairs = request.window - request.lag
    for index in range(len(request.values)):
        if index + 1 < request.window:
            correlations.append(None)
            statuses.append("warmup")
            continue
        trailing = request.values[index + 1 - request.window : index + 1]
        left_offsets = [value - trailing[0] for value in trailing[:pairs]]
        right_offsets = [value - trailing[request.lag] for value in trailing[request.lag :]]
        left_scale = max(abs(value) for value in left_offsets)
        right_scale = max(abs(value) for value in right_offsets)
        if left_scale == 0.0 or right_scale == 0.0:
            correlations.append(None)
            statuses.append("zero_variance")
            continue
        left_scaled = [value / left_scale for value in left_offsets]
        right_scaled = [value / right_scale for value in right_offsets]
        left_mean = fsum(left_scaled) / pairs
        right_mean = fsum(right_scaled) / pairs
        left_centered = [value - left_mean for value in left_scaled]
        right_centered = [value - right_mean for value in right_scaled]
        covariance_sum = fsum(
            left * right for left, right in zip(left_centered, right_centered, strict=True)
        )
        left_norm = sqrt(fsum(value * value for value in left_centered))
        right_norm = sqrt(fsum(value * value for value in right_centered))
        correlation = covariance_sum / (left_norm * right_norm)
        # In exact arithmetic Cauchy-Schwarz bounds this result; remove only
        # floating-point overshoot rather than exposing a correlation above 1.
        correlations.append(min(1.0, max(-1.0, correlation)))
        statuses.append("finite")
    return Output(
        correlations=correlations,
        status=statuses,
        window=request.window,
        lag=request.lag,
        pairs_per_full_window=pairs,
    )


OPERATION = Operation(
    id="features.rolling_autocorrelation",
    kind="feature",
    description=(
        "Compute Pearson correlation of lagged pairs within complete trailing windows, "
        "centering the two pair vectors separately. Enforces minimum pairs; warmup and "
        "zero variance return null. Caller supplies ordered, PIT-selected values."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
