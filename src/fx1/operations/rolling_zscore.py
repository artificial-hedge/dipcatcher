"""Current-inclusive trailing population z-scores with explicit warmup.

Rows must be ordered and available at the intended decision time before this
operation is invoked. Full windows containing a constant value produce null,
since dividing by a zero standard deviation has no defined z-score.
"""

from math import fsum, sqrt
from typing import Annotated

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e150, le=1e150, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    window: int = Field(default=20, strict=True, ge=2, le=1_000)


class Output(OutputModel):
    zscores: list[FiniteFloat | None]
    window: int
    population_variance: bool = True


def execute(request: Input, context: OperationContext) -> Output:
    """Standardize each observation against its own complete trailing window.

    Centering relative to the first window value avoids rounding a small mean
    offset into a large level. Scaling centered deviations before squaring also preserves
    subnormal-magnitude differences that otherwise underflow to zero variance.
    """
    result: list[float | None] = []
    for index, value in enumerate(request.values):
        if index + 1 < request.window:
            result.append(None)
            continue
        trailing = request.values[index + 1 - request.window : index + 1]
        anchor = trailing[0]
        offsets = [item - anchor for item in trailing]
        mean_offset = fsum(offsets) / request.window
        centered = [item - mean_offset for item in offsets]
        scale = max(abs(item) for item in centered)
        if scale == 0.0:
            result.append(None)
            continue
        normalized_variance = fsum((item / scale) ** 2 for item in centered) / request.window
        result.append(((value - anchor - mean_offset) / scale) / sqrt(normalized_variance))
    return Output(zscores=result, window=request.window)


OPERATION = Operation(
    id="features.rolling_zscore",
    kind="feature",
    description=(
        "Compute full-window trailing population z-scores including the current observation; "
        "warmup and constant windows are null. Caller supplies PIT-filtered ordered values."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
