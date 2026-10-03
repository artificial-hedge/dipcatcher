"""Current-inclusive trailing percentile ranks with average ranks for ties.

The percentile is (number smaller + (number tied + 1)/2) / window.
Ranks are one-based: a unique maximum is 1 and a unique minimum is 1/window.
The caller must order and select all rows using source availability; this
array transform cannot establish point-in-time provenance itself.
"""

from typing import Annotated

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Percentile = Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    window: int = Field(default=20, strict=True, ge=1, le=1_000)


class Output(OutputModel):
    percentiles: list[Percentile | None] = Field(max_length=10_000)
    window: int
    tie_convention: str = "average_one_based_rank_divided_by_window"


def execute(request: Input, context: OperationContext) -> Output:
    """Return null until the complete window exists, then rank the current row."""
    percentiles: list[float | None] = []
    for index, current in enumerate(request.values):
        if index + 1 < request.window:
            percentiles.append(None)
            continue
        trailing = request.values[index + 1 - request.window : index + 1]
        smaller = sum(value < current for value in trailing)
        ties = sum(value == current for value in trailing)
        percentiles.append((smaller + (ties + 1) / 2.0) / request.window)
    return Output(percentiles=percentiles, window=request.window)


OPERATION = Operation(
    id="features.rolling_rank",
    kind="feature",
    description=(
        "Rank each observation in its complete current-inclusive trailing window using "
        "average one-based tie rank divided by window length. Warmup rows are null; "
        "the caller supplies ordered, PIT-selected values."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
