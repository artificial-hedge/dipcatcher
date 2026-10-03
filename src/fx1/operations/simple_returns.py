"""Causal lagged fractional changes for a caller-supplied positive price path.

Inputs must already be ordered and selected using the caller's point-in-time
availability rules. This transform neither adjusts corporate actions nor checks
source availability. Outputs are descriptive features, not research evidence.
"""

from typing import Annotated

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Price = Annotated[float, Field(strict=True, ge=1e-150, le=1e150, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class Input(InputModel):
    prices: list[Price] = Field(min_length=1, max_length=10_000)
    lag: int = Field(default=1, strict=True, ge=1, le=10_000)


class Output(OutputModel):
    returns: list[FiniteFloat | None]
    lag: int


def execute(request: Input, context: OperationContext) -> Output:
    """Return p[t] / p[t-lag] - 1; the first ``lag`` values are unknown."""
    result: list[float | None] = [None] * min(request.lag, len(request.prices))
    for index in range(request.lag, len(request.prices)):
        result.append(request.prices[index] / request.prices[index - request.lag] - 1.0)
    return Output(returns=result, lag=request.lag)


OPERATION = Operation(
    id="features.simple_returns",
    kind="feature",
    description=(
        "Compute lagged fractional changes of an ordered positive price path; "
        "warmup rows are null. Caller must enforce source availability and price basis."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
