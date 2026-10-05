"""Exact optimal partitioning for penalized, constant-mean segments.

Minimize sum of within-segment squared errors plus penalty times the number
of changes, covering every supplied observation with contiguous segments of
at least minimum_segment_length. Segment cost is Q-S*S/length, evaluated
from integer prefix sums in common binary units. Rational dynamic programming
compares the exact values represented by the supplied binary64 inputs.
At every prefix, equal objectives prefer fewer segments, then the earlier
start of the final segment; previously selected prefix ties remain fixed.

Means and cost summaries are rounded only for output. A separate residual
cost uses those returned means. Nonzero values outside finite binary64 range
fail instead of becoming zero. Up to 256 observations and 32896 candidate
segments are supported; exact fractions have a 16384-bit budget. This fits
the entire supplied block and is not an online detector or a significance
test. Caller controls source ordering and point-in-time availability.

Optimal-partitioning recurrence, with a penalty per change:
https://www.lancaster.ac.uk/~romano/teaching/2425MATH337/3_multiple_changes.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=256)
    penalty_per_change: float = Field(strict=True, ge=0, le=1e200)
    minimum_segment_length: int = Field(default=1, strict=True, ge=1, le=256)

    @model_validator(mode="after")
    def feasible_length(self) -> Self:
        if self.minimum_segment_length > len(self.values):
            raise ValueError("minimum segment length exceeds the observation count")
        return self


class Segment(OutputModel):
    start_index: int
    end_index_exclusive: int
    observation_count: int
    mean: float
    minimum_squared_error: float
    squared_error_from_returned_mean: float


class Output(OutputModel):
    observation_count: int
    minimum_segment_length: int
    penalty_per_change: float
    segment_count: int
    change_indices: list[int]
    segments: list[Segment]
    fitted_values: list[float]
    minimum_squared_error: float
    penalty_cost: float
    objective: float
    objective_exact_numerator: str
    objective_exact_denominator: str
    squared_error_from_returned_means: float
    candidate_segments_evaluated: int
    tie_rule: Literal["fewer_segments_then_earlier_final_start_at_each_prefix"] = (
        "fewer_segments_then_earlier_final_start_at_each_prefix"
    )
    objective_units: Literal["sum_squared_input_units_plus_supplied_penalty"] = (
        "sum_squared_input_units_plus_supplied_penalty"
    )
    timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("segmentation arithmetic exceeds the 16384-bit budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite binary64 range") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.values)
    ratios = [value.as_integer_ratio() for value in request.values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    sums, squares = [0], [0]
    for value in units:
        sums.append(sums[-1] + value)
        squares.append(squares[-1] + value * value)

    def cost(start: int, end: int) -> Fraction:
        length = end - start
        total = sums[end] - sums[start]
        return Fraction(
            length * (squares[end] - squares[start]) - total * total, length << (2 * places)
        )

    penalty = Fraction(request.penalty_per_change)
    best: list[Fraction | None] = [Fraction()] + [None] * count
    lengths = [0] * (count + 1)
    previous = [-1] * (count + 1)
    evaluated = 0
    for end in range(request.minimum_segment_length, count + 1):
        chosen: tuple[Fraction, int, int] | None = None
        for start in range(end - request.minimum_segment_length + 1):
            prefix = best[start]
            if prefix is None:
                continue
            evaluated += 1
            candidate = _bounded(prefix + cost(start, end) + (penalty if start else 0))
            key = (candidate, lengths[start] + 1, start)
            if chosen is None or key < chosen:
                chosen = key
        if chosen is not None:
            best[end], lengths[end], previous[end] = chosen
    objective = best[count]
    if objective is None:
        raise ValueError("no partition satisfies the minimum segment length")
    intervals: list[tuple[int, int]] = []
    end = count
    while end:
        start = previous[end]
        if start < 0 or start >= end:
            raise ValueError("invalid segmentation backpointer")
        intervals.append((start, end))
        end = start
    intervals.reverse()
    segments: list[Segment] = []
    fitted: list[float] = []
    minimum_error, returned_error = Fraction(), Fraction()
    for start, end in intervals:
        length = end - start
        mean = Fraction(sums[end] - sums[start], length << places)
        returned_mean = _number(mean, "segment mean")
        segment_error = cost(start, end)
        emitted_error = _bounded(
            sum(
                (
                    (Fraction(request.values[index]) - Fraction(returned_mean)) ** 2
                    for index in range(start, end)
                ),
                Fraction(),
            )
        )
        minimum_error = _bounded(minimum_error + segment_error)
        returned_error = _bounded(returned_error + emitted_error)
        fitted.extend([returned_mean] * length)
        segments.append(
            Segment(
                start_index=start,
                end_index_exclusive=end,
                observation_count=length,
                mean=returned_mean,
                minimum_squared_error=_number(segment_error, "segment minimum squared error"),
                squared_error_from_returned_mean=_number(
                    emitted_error, "returned-mean squared error"
                ),
            )
        )
    penalty_cost = penalty * (len(segments) - 1)
    if minimum_error + penalty_cost != objective:
        raise ValueError("reconstructed partition does not match its objective")
    return Output(
        observation_count=count,
        minimum_segment_length=request.minimum_segment_length,
        penalty_per_change=request.penalty_per_change,
        segment_count=len(segments),
        change_indices=[start for start, _ in intervals[1:]],
        segments=segments,
        fitted_values=fitted,
        minimum_squared_error=_number(minimum_error, "minimum squared error"),
        penalty_cost=_number(penalty_cost, "penalty cost"),
        objective=_number(objective, "objective"),
        objective_exact_numerator=str(objective.numerator),
        objective_exact_denominator=str(objective.denominator),
        squared_error_from_returned_means=_number(returned_error, "returned-means squared error"),
        candidate_segments_evaluated=evaluated,
    )


OPERATION = Operation(
    id="features.segment_mean_changes",
    kind="feature",
    description="Exactly minimize penalized constant-mean segment SSE with bounded optimal partitioning.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
