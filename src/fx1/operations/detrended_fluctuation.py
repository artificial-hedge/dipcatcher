"""First-order detrended fluctuation analysis with explicit scale selection.

Form Y_k=sum_(i<=k)(x_i-mean(x)) over the entire equally spaced input. For each
supplied scale s, split from the beginning into floor(N/s) nonoverlapping boxes;
drop the final N mod s profile points, without a reverse pass or overlap. Fit
an intercept and linear trend in each box by least squares. F(s) is the square
root of the total squared residual divided by the number of retained points.
At least two boxes are required per scale, and s must be at least four.

The profile and local residual sums are exact binary-rational calculations.
Positive F(s) values are rounded with an exact-ratio square root; underflow is
an explicit error. The exponent is unweighted OLS of log F(s) on log s over
positive eligible scales. Logs are floating approximations, formed as ratios
to the first valid scale to preserve small relative differences. Fit arithmetic
after those logs is exact rational arithmetic before final float conversion.
R-squared is undefined for constant log fluctuations. Constant input and fewer
than two usable scales return explicit statuses and no exponent.

The caller must provide equally spaced, ordered, availability-selected samples.
This descriptive fit does not establish long memory, a Hurst interpretation,
stationarity, a scaling range or a statistical confidence interval.
Definition: C-K Peng's PhysioNet DFA description (linear local detrending):
https://physionet.org/content/dfa/1.0.0/
"""

from fractions import Fraction
from itertools import pairwise
from math import isfinite, log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Scale = Annotated[int, Field(strict=True, ge=4, le=4_096)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class Input(InputModel):
    values: list[Value] = Field(min_length=4, max_length=4_096)
    scales: list[Scale] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_scales(self) -> Self:
        if any(right <= left for left, right in pairwise(self.scales)):
            raise ValueError("scales must be strictly increasing and unique")
        if len(self.values) * len(self.scales) > 131_072:
            raise ValueError("at most 131072 input-scale cells are supported")
        return self


class ScaleResult(OutputModel):
    scale: int
    segment_count: int
    retained_point_count: int
    discarded_tail_count: int
    fluctuation: Finite | None
    status: Literal["valid", "zero_fluctuation", "insufficient_segments"]


class Output(OutputModel):
    observation_count: int
    constant_input: bool
    valid_scale_count: int
    scaling_exponent: Finite | None
    log_fit_intercept: Finite | None
    log_fit_r_squared: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None
    fit_status: Literal[
        "fitted", "constant_log_fluctuation", "constant_input", "insufficient_valid_scales"
    ]
    scales: list[ScaleResult] = Field(min_length=1, max_length=32)
    remainder_policy: Literal["discard_tail_no_reverse_pass"] = "discard_tail_no_reverse_pass"
    trend_order: Literal[1] = 1
    log_base: Literal["natural"] = "natural"


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _log_ratio(value: Fraction) -> float:
    difference = value - 1
    result = (
        log1p(_number(difference, "log-ratio deviation"))
        if abs(difference) <= Fraction(1, 2)
        else log(value.numerator) - log(value.denominator)
    )
    if not isfinite(result) or (value != 1 and result == 0):
        raise ValueError("nonzero logarithmic ratio is outside supported precision")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Accumulate integer profile and exact local OLS residual sums for each scale."""
    count = len(request.values)
    ratios = [value.as_integer_ratio() for value in request.values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    total = sum(units)
    accumulated = 0
    profile: list[int] = []
    for value in units:
        accumulated += count * value - total
        profile.append(accumulated)
    profile_denominator_squared = (count << places) ** 2
    results: list[ScaleResult] = []
    valid: list[tuple[int, Fraction]] = []
    for scale in request.scales:
        segments, tail = divmod(count, scale)
        used = segments * scale
        if segments < 2:
            results.append(
                ScaleResult(
                    scale=scale,
                    segment_count=segments,
                    retained_point_count=used,
                    discarded_tail_count=tail,
                    fluctuation=None,
                    status="insufficient_segments",
                )
            )
            continue
        centered_index = [2 * index - (scale - 1) for index in range(scale)]
        index_squares = sum(index * index for index in centered_index)
        residual_numerator = 0
        for start in range(0, used, scale):
            box = profile[start : start + scale]
            box_sum = sum(box)
            syy = sum(value * value for value in box)
            sxy = sum(index * value for index, value in zip(centered_index, box, strict=True))
            residual_numerator += (
                scale * index_squares * syy - index_squares * box_sum * box_sum - scale * sxy * sxy
            )
        squared = Fraction(
            residual_numerator, scale * index_squares * used * profile_denominator_squared
        )
        fluctuation = correctly_rounded_sqrt(squared.numerator, squared.denominator)
        if not isfinite(fluctuation) or (squared != 0 and fluctuation == 0):
            raise ValueError(
                f"positive fluctuation at scale {scale} is outside finite binary64 range"
            )
        results.append(
            ScaleResult(
                scale=scale,
                segment_count=segments,
                retained_point_count=used,
                discarded_tail_count=tail,
                fluctuation=fluctuation,
                status="valid" if squared else "zero_fluctuation",
            )
        )
        if squared:
            valid.append((scale, squared))
    constant = len(set(units)) == 1
    exponent = intercept = r_squared = None
    status: Literal[
        "fitted", "constant_log_fluctuation", "constant_input", "insufficient_valid_scales"
    ] = "constant_input" if constant else "insufficient_valid_scales"
    if len(valid) >= 2:
        base_scale, base_squared = valid[0]
        xs = [Fraction(_log_ratio(Fraction(scale, base_scale))) for scale, _ in valid]
        ys = [Fraction(_log_ratio(squared / base_squared)) / 2 for _, squared in valid]
        size = len(valid)
        sx, sy = sum(xs, Fraction()), sum(ys, Fraction())
        xx = size * sum((x * x for x in xs), Fraction()) - sx * sx
        yy = size * sum((y * y for y in ys), Fraction()) - sy * sy
        xy = size * sum((x * y for x, y in zip(xs, ys, strict=True)), Fraction()) - sx * sy
        slope = xy / xx
        fitted_intercept = (
            Fraction(_log_ratio(base_squared)) / 2
            + (sy - slope * sx) / size
            - slope * Fraction(log(base_scale))
        )
        exponent = _number(slope, "scaling exponent")
        intercept = _number(fitted_intercept, "log-fit intercept")
        r_squared = _number(xy * xy / (xx * yy), "log-fit R-squared") if yy else None
        status = "fitted" if yy else "constant_log_fluctuation"
    return Output(
        observation_count=count,
        constant_input=constant,
        valid_scale_count=len(valid),
        scaling_exponent=exponent,
        log_fit_intercept=intercept,
        log_fit_r_squared=r_squared,
        fit_status=status,
        scales=results,
    )


OPERATION = Operation(
    id="features.detrended_fluctuation",
    kind="feature",
    description=(
        "Compute bounded DFA1 fluctuations from exact demeaned profiles and local linear residuals, "
        "with explicit tail handling, usable-scale diagnostics and descriptive log-log fit."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
