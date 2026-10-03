"""Describe an observed positive path relative to its running high-water mark.

This is an ordered-array transformation, not an investment performance claim or
a research headline metric. The caller selects the price basis and point-in-time
rows. Reattaining an earlier maximum resets duration to zero.
"""

from typing import Annotated

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Price = Annotated[float, Field(strict=True, ge=1e-150, le=1e150, allow_inf_nan=False)]
Fraction = Annotated[float, Field(ge=-1.0, le=0.0, allow_inf_nan=False)]


class Input(InputModel):
    prices: list[Price] = Field(min_length=1, max_length=10_000)


class Output(OutputModel):
    high_water_marks: list[Price]
    drawdowns: list[Fraction]
    durations: list[Annotated[int, Field(ge=0)]]


def execute(request: Input, context: OperationContext) -> Output:
    """Return running peak, p[t]/peak[t]-1, and rows since the latest peak."""
    high_water_mark = request.prices[0]
    peak_index = 0
    peaks: list[float] = []
    drawdowns: list[float] = []
    durations: list[int] = []
    for index, price in enumerate(request.prices):
        if price >= high_water_mark:
            high_water_mark = price
            peak_index = index
        peaks.append(high_water_mark)
        drawdowns.append(price / high_water_mark - 1.0)
        durations.append(index - peak_index)
    return Output(high_water_marks=peaks, drawdowns=drawdowns, durations=durations)


OPERATION = Operation(
    id="features.drawdown_path",
    kind="feature",
    description=(
        "Describe an ordered positive path using running peaks, nonpositive fractional "
        "drawdowns, and rows since the latest attained peak; this is a descriptive feature."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
