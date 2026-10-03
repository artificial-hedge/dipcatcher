"""Causal exponentially weighted second-moment forecasts under a zero mean.

The variance before observing return t is v[t]. Once return t is available,
v[t+1] = decay * v[t] + (1 - decay) * return[t]**2. The supplied initial
variance must be known before the first observation. No future rows initialize
the recursion. Source availability and ordering remain the caller's duty.
"""

from typing import Annotated

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Return = Annotated[float, Field(strict=True, ge=-1e150, le=1e150, allow_inf_nan=False)]
Variance = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class Input(InputModel):
    returns: list[Return] = Field(min_length=1, max_length=10_000)
    decay: float = Field(default=0.94, strict=True, ge=0.0, lt=1.0, allow_inf_nan=False)
    initial_variance: float = Field(default=0.0, strict=True, ge=0.0, le=1e300, allow_inf_nan=False)


class Output(OutputModel):
    forecast_variances: list[Variance]
    next_variance: Variance
    decay: float
    zero_mean_assumption: bool = True


def execute(request: Input, context: OperationContext) -> Output:
    """Return pre-observation forecasts and the next forecast after the last row."""
    variance = request.initial_variance
    forecasts: list[float] = []
    innovation_weight = 1.0 - request.decay
    for observation in request.returns:
        forecasts.append(variance)
        variance = request.decay * variance + innovation_weight * (observation * observation)
    return Output(forecast_variances=forecasts, next_variance=variance, decay=request.decay)


OPERATION = Operation(
    id="features.ewma_variance",
    kind="feature",
    description=(
        "Compute causal zero-mean EWMA variance forecasts from ordered returns and a prior "
        "variance; each aligned forecast excludes its same-row observation."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
