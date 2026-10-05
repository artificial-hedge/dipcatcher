"""Normalized QLIKE loss for strictly positive variance targets and forecasts.

L(y,h)=y/h-log(y/h)-1. Inputs are variances, not standard deviations or log
variances. Strictly positive targets are required by this normalization; zeros
are rejected instead of being silently floored. Patton (2011), equation 24,
doi:10.1016/j.jeconom.2010.03.034. No inference about proxy validity is made.
"""

from math import fsum, log
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Variance = Annotated[float, Field(strict=True, ge=1e-150, le=1e150)]


class Input(InputModel):
    realized_variances: list[Variance] = Field(min_length=1, max_length=10_000)
    forecast_variances: list[Variance] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def equal_lengths(self) -> Self:
        if len(self.realized_variances) != len(self.forecast_variances):
            raise ValueError("realized and forecast variances must have equal lengths")
        return self


class Output(OutputModel):
    observation_count: int
    mean_qlike: float
    minimum_qlike: float
    maximum_qlike: float
    qlike_by_observation: list[float]
    normalization: str = "realized_over_forecast_minus_log_ratio_minus_one"
    clipping_applied: bool = False


def execute(request: Input, context: OperationContext) -> Output:
    losses: list[float] = []
    for realized, forecast in zip(
        request.realized_variances, request.forecast_variances, strict=True
    ):
        delta = (realized - forecast) / forecast
        if abs(delta) <= 1e-3:
            # delta-log(1+delta) cancels near equality. This alternating series
            # starts at delta^2/2; its omitted terms are below floating precision
            # relative to that leading term throughout this branch.
            loss = fsum(
                ((-1.0) ** exponent) * delta**exponent / exponent for exponent in range(2, 13)
            )
        else:
            ratio = realized / forecast
            loss = ratio - log(ratio) - 1.0
        losses.append(max(0.0, loss))
    return Output(
        observation_count=len(losses),
        mean_qlike=fsum(losses) / len(losses),
        minimum_qlike=min(losses),
        maximum_qlike=max(losses),
        qlike_by_observation=losses,
    )


OPERATION = Operation(
    id="skills.score_variance_qlike",
    kind="skill",
    description=(
        "Compute normalized lower-is-better QLIKE for positive realized/forecast variances "
        "with a cancellation-resistant near-equality formula; no zero flooring or unit conversion."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
