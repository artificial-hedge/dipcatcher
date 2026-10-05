"""Weighted covariance using exact binary-rational centered moment sums.

Rows are observations and columns are variables. Nonnegative weights are
normalized as p_i=w_i/sum(w). Zero-weight rows do not affect moments.
Population covariance is sum(p_i*(x_i-mu)*(x_i-mu)^T). The reliability_unbiased
mode divides this by 1-sum(p_i^2); it requires at least two positive weights.
These are reliability/analytic weights, not repeated-observation frequencies.
The correction is unbiased under independent observations with common mean
and covariance and fixed weights; no claim covers arbitrary dependent data.

Binary64 inputs are converted to exact integer units. Anchored moment sums,
centering and the reliability denominator are computed with integers before
one final float conversion. An unrepresentable nonzero returned quantity
raises rather than silently becoming zero or infinity. Returned probabilities
and moments are rounded binary64 approximations to those exact calculations.
The caller must select observations available at the intended decision time.

Reference: NumPy cov notes, with fweights=1 and aweights=w:
https://numpy.org/doc/stable/reference/generated/numpy.cov.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
Row = Annotated[list[Value], Field(min_length=1, max_length=8)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Vector = Annotated[list[Finite], Field(min_length=1, max_length=8)]
Mode = Literal["population", "reliability_unbiased"]


class Input(InputModel):
    observations: list[Row] = Field(min_length=1, max_length=2_048)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=2_048)
    normalization: Mode = "population"

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        width = len(self.observations[0])
        if any(len(row) != width for row in self.observations):
            raise ValueError("observations must have a common column count")
        if self.weights is not None:
            if len(self.weights) != len(self.observations):
                raise ValueError("weights must align with observation rows")
            if not any(self.weights):
                raise ValueError("at least one weight must be positive")
        return self


class Output(OutputModel):
    observation_count: int
    variable_count: int
    positive_weight_count: int
    normalization: Mode
    weight_sum: float = Field(gt=0, allow_inf_nan=False)
    normalized_weights: list[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]] = Field(
        max_length=2_048
    )
    effective_sample_size: float = Field(ge=1, le=2_048, allow_inf_nan=False)
    reliability_correction_denominator: float = Field(ge=0, le=1, allow_inf_nan=False)
    means: Vector
    covariance: list[Vector] = Field(min_length=1, max_length=8)
    constant_columns: list[int] = Field(max_length=8)
    weight_interpretation: Literal["reliability_not_frequency"] = "reliability_not_frequency"


def _integer_units(values: list[float]) -> tuple[list[int], int]:
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return [
        numerator << (places - (denominator.bit_length() - 1)) for numerator, denominator in ratios
    ], places


def _finite_number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0.0):
        raise ValueError(f"{label} cannot be represented as a finite nonzero binary64 number")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Accumulate weighted centered products exactly within bounded integer sizes."""
    weights = request.weights if request.weights is not None else [1.0] * len(request.observations)
    integer_weights, weight_places = _integer_units(weights)
    total = sum(integer_weights)
    squares = sum(weight * weight for weight in integer_weights)
    total_squared = total * total
    reliability_denominator = total_squared - squares
    if request.normalization == "reliability_unbiased" and reliability_denominator == 0:
        raise ValueError("reliability_unbiased covariance requires at least two positive weights")
    probabilities = [
        _finite_number(Fraction(weight, total), "normalized weight") for weight in integer_weights
    ]
    active = [index for index, weight in enumerate(integer_weights) if weight]
    anchor_index = max(active, key=integer_weights.__getitem__)
    dimension = len(request.observations[0])
    centered_columns: list[list[int]] = []
    weighted_sums: list[int] = []
    column_places: list[int] = []
    means: list[float] = []
    constant: list[int] = []
    for column in range(dimension):
        units, places = _integer_units([row[column] for row in request.observations])
        anchor = units[anchor_index]
        centered = [value - anchor for value in units]
        weighted_sum = sum(integer_weights[index] * centered[index] for index in active)
        means.append(
            _finite_number(
                Fraction(anchor * total + weighted_sum, total << places),
                f"mean for column {column}",
            )
        )
        if all(centered[index] == 0 for index in active):
            constant.append(column)
        centered_columns.append(centered)
        column_places.append(places)
        weighted_sums.append(weighted_sum)
    denominator = (
        total_squared if request.normalization == "population" else reliability_denominator
    )
    covariance = [[0.0] * dimension for _ in range(dimension)]
    for left in range(dimension):
        for right in range(left, dimension):
            product_sum = sum(
                integer_weights[index]
                * centered_columns[left][index]
                * centered_columns[right][index]
                for index in active
            )
            numerator = product_sum * total - weighted_sums[left] * weighted_sums[right]
            value = _finite_number(
                Fraction(numerator, denominator << (column_places[left] + column_places[right])),
                f"covariance[{left},{right}]",
            )
            covariance[left][right] = covariance[right][left] = value
    return Output(
        observation_count=len(request.observations),
        variable_count=dimension,
        positive_weight_count=len(active),
        normalization=request.normalization,
        weight_sum=_finite_number(Fraction(total, 1 << weight_places), "weight sum"),
        normalized_weights=probabilities,
        effective_sample_size=_finite_number(
            Fraction(total_squared, squares), "effective sample size"
        ),
        reliability_correction_denominator=_finite_number(
            Fraction(reliability_denominator, total_squared), "reliability correction denominator"
        ),
        means=means,
        covariance=covariance,
        constant_columns=constant,
    )


OPERATION = Operation(
    id="features.weighted_covariance",
    kind="feature",
    description=(
        "Compute weighted means and population or reliability-corrected covariance using "
        "exact integer moment accumulation, normalized weights and effective sample size. "
        "Bounds rows/columns and rejects unrepresentable nonzero outputs."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
