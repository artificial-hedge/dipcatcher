"""Trailing median, raw median absolute deviation, and unscaled robust scores.

The robust score is (current - median) / median_absolute_deviation. No normal
distribution consistency factor is applied and no normality claim is implied.
Warmup and zero-MAD rows have null scores. If a nonzero, extremely small MAD
makes the score unrepresentable, its status explicitly reports numeric overflow.
The caller supplies rows ordered and selected under PIT availability rules.
"""

from math import isfinite
from statistics import median
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Deviation = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
Status = Literal["warmup", "zero_mad", "numeric_overflow", "finite"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    window: int = Field(default=20, strict=True, ge=1, le=1_000)


class Output(OutputModel):
    medians: list[Finite | None] = Field(max_length=10_000)
    median_absolute_deviations: list[Deviation | None] = Field(max_length=10_000)
    robust_zscores: list[Finite | None] = Field(max_length=10_000)
    zscore_status: list[Status] = Field(max_length=10_000)
    window: int
    normal_consistency_factor_applied: bool = False


def execute(request: Input, context: OperationContext) -> Output:
    """Compute each full trailing window directly, including tied observations."""
    medians: list[float | None] = []
    deviations: list[float | None] = []
    scores: list[float | None] = []
    statuses: list[Status] = []
    for index, current in enumerate(request.values):
        if index + 1 < request.window:
            medians.append(None)
            deviations.append(None)
            scores.append(None)
            statuses.append("warmup")
            continue
        trailing = request.values[index + 1 - request.window : index + 1]
        # An even-window median can lie between adjacent representable floats.
        # Keep its small offset separate from the large level for the deviation
        # and score calculations; only the reported absolute median is rounded.
        ordered = sorted(trailing)
        anchor = ordered[(request.window - 1) // 2]
        center_offset = (
            (ordered[request.window // 2] - anchor) / 2.0 if request.window % 2 == 0 else 0.0
        )
        center = anchor + center_offset
        deviation = float(median(abs(value - anchor - center_offset) for value in trailing))
        medians.append(center)
        deviations.append(deviation)
        if deviation == 0.0:
            scores.append(None)
            statuses.append("zero_mad")
            continue
        score = (current - anchor - center_offset) / deviation
        if isfinite(score):
            scores.append(score)
            statuses.append("finite")
        else:
            scores.append(None)
            statuses.append("numeric_overflow")
    return Output(
        medians=medians,
        median_absolute_deviations=deviations,
        robust_zscores=scores,
        zscore_status=statuses,
        window=request.window,
    )


OPERATION = Operation(
    id="features.rolling_mad",
    kind="feature",
    description=(
        "Compute trailing medians, raw median absolute deviations, and unscaled robust "
        "scores. Warmup, zero MAD, and unrepresentable scores are explicitly distinguished; "
        "no normality assumption or consistency factor is applied."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
