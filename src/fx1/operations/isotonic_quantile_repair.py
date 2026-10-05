"""Project supplied quantile vectors onto the nondecreasing cone with own PAVA.

For every prediction row, minimize sum_j w_j*(z_j-q_j)^2 subject to
z_j<=z_(j+1). Supplied quantile levels must already be strictly increasing;
their spacing does not implicitly become a weight. All projection weights
are positive and shared across rows. Equal neighboring pool means need not
be merged. Every emitted row is the rounded weighted least-squares projection.

Binary-rational integer pooling makes comparisons and block means exact for
the submitted binary64 values and weights. Output conversion rejects a nonzero
quantity that would underflow or overflow. Adjustment diagnostics use emitted
values; an additional count records changes in the exact projection before
rounding. The operation does not assess calibration, forecast skill, proper
scores, or improvement in any score. Caller enforces point-in-time availability.

Reference for weighted isotonic objective and PAVA:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Level = Annotated[float, Field(strict=True, gt=0, lt=1, allow_inf_nan=False)]
Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Weight = Annotated[float, Field(strict=True, gt=0, le=1e100, allow_inf_nan=False)]
Prediction = Annotated[list[Value], Field(min_length=1, max_length=256)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
ResultRow = Annotated[list[Finite], Field(min_length=1, max_length=256)]


class Input(InputModel):
    quantile_levels: list[Level] = Field(min_length=1, max_length=256)
    predictions: list[Prediction] = Field(min_length=1, max_length=512)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        size = len(self.quantile_levels)
        if any(right <= left for left, right in pairwise(self.quantile_levels)):
            raise ValueError("quantile_levels must be strictly increasing")
        if any(len(row) != size for row in self.predictions):
            raise ValueError("every prediction row must align with quantile_levels")
        if self.weights is not None and len(self.weights) != size:
            raise ValueError("weights must align with quantile_levels")
        if len(self.predictions) * size > 16_384:
            raise ValueError("at most 16384 prediction cells are supported")
        return self


class RowSummary(OutputModel):
    row_index: int
    adjacent_violations_before: int
    pooled_block_count: int
    exact_projection_changed_count: int
    returned_changed_count: int
    maximum_absolute_adjustment: float = Field(ge=0, allow_inf_nan=False)
    returned_weighted_mean_squared_adjustment: float = Field(ge=0, allow_inf_nan=False)


class Output(OutputModel):
    quantile_levels: list[Level] = Field(max_length=256)
    normalized_weights: list[Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]] = Field(
        max_length=256
    )
    repaired_predictions: list[ResultRow] = Field(max_length=512)
    absolute_adjustments: list[ResultRow] = Field(max_length=512)
    rows: list[RowSummary] = Field(max_length=512)
    rows_with_crossings: int
    rows_changed_after_rounding: int
    total_changed_after_rounding: int


@dataclass
class _Pool:
    start: int
    stop: int
    weight: int
    weighted_value: int


def _units(values: list[float]) -> tuple[list[int], int]:
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ], places


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0.0):
        raise ValueError(f"{label} is not representable as a finite nonzero binary64 number")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Pool adjacent violating means, retaining exact integer sufficient statistics."""
    supplied_weights = (
        request.weights if request.weights is not None else [1.0] * len(request.quantile_levels)
    )
    weights, _ = _units(supplied_weights)
    weight_sum = sum(weights)
    probabilities = [
        _number(Fraction(weight, weight_sum), "normalized weight") for weight in weights
    ]
    repaired: list[list[float]] = []
    adjustments: list[list[float]] = []
    summaries: list[RowSummary] = []
    for row_index, row in enumerate(request.predictions):
        units, places = _units(row)
        pools: list[_Pool] = []
        for index, (value, weight) in enumerate(zip(units, weights, strict=True)):
            pools.append(_Pool(index, index + 1, weight, weight * value))
            while len(pools) >= 2:
                left, right = pools[-2], pools[-1]
                if left.weighted_value * right.weight <= right.weighted_value * left.weight:
                    break
                pools.pop()
                pools[-1] = _Pool(
                    left.start,
                    right.stop,
                    left.weight + right.weight,
                    left.weighted_value + right.weighted_value,
                )
        fitted: list[float] = []
        exact_changed = 0
        for pool in pools:
            exact_value = Fraction(pool.weighted_value, pool.weight << places)
            rounded_value = _number(exact_value, f"projected quantile in row {row_index}")
            fitted.extend([rounded_value] * (pool.stop - pool.start))
            exact_changed += sum(
                pool.weighted_value != units[index] * pool.weight
                for index in range(pool.start, pool.stop)
            )
        exact_differences = [
            Fraction(value) - Fraction(original)
            for value, original in zip(fitted, row, strict=True)
        ]
        absolute_changes = [
            _number(abs(value), "absolute adjustment") for value in exact_differences
        ]
        objective = (
            sum(
                (
                    weight * difference * difference
                    for weight, difference in zip(weights, exact_differences, strict=True)
                ),
                Fraction(),
            )
            / weight_sum
        )
        changed = sum(value != original for value, original in zip(fitted, row, strict=True))
        summaries.append(
            RowSummary(
                row_index=row_index,
                adjacent_violations_before=sum(left > right for left, right in pairwise(row)),
                pooled_block_count=len(pools),
                exact_projection_changed_count=exact_changed,
                returned_changed_count=changed,
                maximum_absolute_adjustment=max(absolute_changes),
                returned_weighted_mean_squared_adjustment=_number(
                    objective, "weighted adjustment objective"
                ),
            )
        )
        repaired.append(fitted)
        adjustments.append(absolute_changes)
    return Output(
        quantile_levels=request.quantile_levels,
        normalized_weights=probabilities,
        repaired_predictions=repaired,
        absolute_adjustments=adjustments,
        rows=summaries,
        rows_with_crossings=sum(row.adjacent_violations_before > 0 for row in summaries),
        rows_changed_after_rounding=sum(row.returned_changed_count > 0 for row in summaries),
        total_changed_after_rounding=sum(row.returned_changed_count for row in summaries),
    )


OPERATION = Operation(
    id="features.isotonic_quantile_repair",
    kind="feature",
    description=(
        "Repair quantile crossings with independently implemented weighted PAVA over "
        "supplied increasing levels, exact pooling arithmetic and explicit returned-value "
        "adjustments. Makes no calibration or forecast-score improvement claim."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
