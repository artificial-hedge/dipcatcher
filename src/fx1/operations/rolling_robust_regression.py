"""Trailing Theil-Sen slopes with the joint median-intercept convention.

Each full, current-inclusive window contributes every unordered pair with
distinct x values, including multiplicity from repeated observations. The slope
is the median pair slope; the intercept is median(y-slope*x). An even median is
the arithmetic mean of its central two values. Equal-x pairs are omitted; an
all-equal-x window has no estimate. Partial windows return warmup status.

All pair differences, ratios, median comparisons and intercepts use exact
binary-rational input values. Only the two final estimates are rounded to
binary64. A nonzero estimate that rounds to zero, or overflows, fails the call.
The intercept uses the exact median slope before either value is rounded, so
recomputing it from the returned rounded slope can give a different result.
No confidence interval or robustness guarantee for arbitrary data is supplied.
The caller must order rows and select samples available at its decision time.

Definition and joint intercept convention: SciPy theilslopes documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Status = Literal["warmup", "all_x_equal", "estimated", "estimated_with_repeated_x"]


class Input(InputModel):
    x: list[Value] = Field(min_length=1, max_length=2_048)
    y: list[Value] = Field(min_length=1, max_length=2_048)
    window: int = Field(default=20, strict=True, ge=2, le=256)

    @model_validator(mode="after")
    def validate_work(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have the same length")
        windows = max(0, len(self.x) - self.window + 1)
        if windows * self.window * (self.window - 1) // 2 > 250_000:
            raise ValueError("at most 250000 candidate pairs across full windows are supported")
        return self


class Estimate(OutputModel):
    source_index: int
    window_start_index: int | None
    status: Status
    distinct_x_count: int
    usable_pair_count: int
    skipped_equal_x_pair_count: int
    slope: Finite | None
    intercept: Finite | None


class Output(OutputModel):
    window: int
    observation_count: int
    full_window_count: int
    evaluated_candidate_pairs: int
    intercept_convention: Literal["median_y_minus_exact_slope_x"] = "median_y_minus_exact_slope_x"
    estimates: list[Estimate] = Field(min_length=1, max_length=2_048)


def _median(values: list[Fraction]) -> Fraction:
    values.sort()
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _representable(value: Fraction, source_index: int, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} at source index {source_index} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"nonzero {name} at source index {source_index} is not representable")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Compute independent exact pair-slope medians for bounded trailing windows."""
    xs = [Fraction(value) for value in request.x]
    ys = [Fraction(value) for value in request.y]
    estimates: list[Estimate] = []
    candidate_count = request.window * (request.window - 1) // 2
    for end in range(len(xs)):
        if end + 1 < request.window:
            estimates.append(
                Estimate(
                    source_index=end,
                    window_start_index=None,
                    status="warmup",
                    distinct_x_count=0,
                    usable_pair_count=0,
                    skipped_equal_x_pair_count=0,
                    slope=None,
                    intercept=None,
                )
            )
            continue
        start = end + 1 - request.window
        slopes = [
            (ys[right] - ys[left]) / (xs[right] - xs[left])
            for left in range(start, end)
            for right in range(left + 1, end + 1)
            if xs[right] != xs[left]
        ]
        pair_count = len(slopes)
        distinct = len(set(xs[start : end + 1]))
        slope = _median(slopes) if slopes else None
        intercept = (
            _median([ys[index] - slope * xs[index] for index in range(start, end + 1)])
            if slope is not None
            else None
        )
        status: Status = "all_x_equal"
        if slope is not None:
            status = "estimated" if distinct == request.window else "estimated_with_repeated_x"
        estimates.append(
            Estimate(
                source_index=end,
                window_start_index=start,
                status=status,
                distinct_x_count=distinct,
                usable_pair_count=pair_count,
                skipped_equal_x_pair_count=candidate_count - pair_count,
                slope=_representable(slope, end, "slope") if slope is not None else None,
                intercept=(
                    _representable(intercept, end, "intercept") if intercept is not None else None
                ),
            )
        )
    full_windows = max(0, len(xs) - request.window + 1)
    return Output(
        window=request.window,
        observation_count=len(xs),
        full_window_count=full_windows,
        evaluated_candidate_pairs=full_windows * candidate_count,
        estimates=estimates,
    )


OPERATION = Operation(
    id="features.rolling_robust_regression",
    kind="feature",
    description=(
        "Compute bounded trailing Theil-Sen pair-slope medians and joint median intercepts "
        "with exact rational arithmetic, explicit warmup and equal-x statuses."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
