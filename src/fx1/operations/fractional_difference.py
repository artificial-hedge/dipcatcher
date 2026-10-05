"""A finite binomial lag polynomial, with fixed width and full-window warmup.

For order d in [0,2], w[0]=1 and w[k]=w[k-1]*(k-1-d)/k. The result at
row t is sum(w[k]*x[t-k], k=0..width-1). All earlier rows are null, even
when d is an integer and some lag weights vanish. Width is the number of
included lags, including lag zero; omitted weights are set to zero. There
is no threshold-based width selection, renormalization, or expanding window.
The finite polynomial's DC gain is reported, not assumed to be zero.

Products are aligned by their binary exponents before compensated summation.
Inputs whose weight recurrence or product alignment loses a nonzero term
are rejected. Final results may round to zero at binary64's subnormal limit
and are flagged. This filter does not establish stationarity or persistence.
Caller supplies ordered observations available at the intended decision time.

Reference for the binomial lag expansion (equations 2-3):
https://pmc.ncbi.nlm.nih.gov/articles/PMC9554581/
The recurrence follows by dividing consecutive binomial coefficients.
"""

from math import frexp, fsum, ldexp
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Order = Annotated[float, Field(strict=True, ge=0, le=2, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Status = Literal["warmup", "finite", "rounded_to_zero"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    order: Order
    width: int = Field(default=64, strict=True, ge=1, le=1_024)

    @model_validator(mode="after")
    def bound_work(self) -> Self:
        output_count = max(0, len(self.values) - self.width + 1)
        if output_count * self.width > 2_000_000:
            raise ValueError("at most 2000000 weighted lag terms are supported")
        return self


class Output(OutputModel):
    values: list[Finite | None] = Field(max_length=10_000)
    status: list[Status] = Field(max_length=10_000)
    order: Order
    width: int
    weights_newest_first: list[Finite] = Field(max_length=1_024)
    coefficient_sum_dc_gain: Finite
    coefficient_l1_norm: float = Field(ge=1, allow_inf_nan=False)
    nonzero_weight_count: int
    warmup_count: int
    computed_count: int
    result_underflow_count: int
    truncation: Literal["lags_at_or_above_width_are_zero"] = "lags_at_or_above_width_are_zero"


def _lag_sum(weights: list[float], values: list[float], end: int) -> tuple[float, bool]:
    # Multiplying mantissas first avoids an underflowing w*x intermediate.
    # Aligning products rather than the raw observations also handles a tiny
    # nonzero weight applied to a large observation when other terms are zero.
    products: list[tuple[float, int]] = []
    for lag, weight in enumerate(weights):
        value = values[end - lag]
        if weight == 0.0 or value == 0.0:
            continue
        weight_mantissa, weight_exponent = frexp(weight)
        value_mantissa, value_exponent = frexp(value)
        products.append((weight_mantissa * value_mantissa, weight_exponent + value_exponent))
    if not products:
        return 0.0, False
    common_exponent = max(exponent for _, exponent in products)
    aligned = [ldexp(mantissa, exponent - common_exponent) for mantissa, exponent in products]
    if any(value == 0.0 for value in aligned):
        raise ValueError(f"weighted terms at row {end} exceed supported binary64 dynamic range")
    normalized_sum = fsum(aligned)
    result = ldexp(normalized_sum, common_exponent)
    return result, normalized_sum != 0.0 and result == 0.0


def execute(request: Input, context: OperationContext) -> Output:
    """Build one fixed lag polynomial and apply it without future observations."""
    weights = [1.0]
    for lag in range(1, request.width):
        numerator = (lag - 1) - request.order
        # Division can underflow even though numerator*previous does not;
        # normalization keeps the recurrence's single final rounding explicit.
        previous_mantissa, previous_exponent = frexp(weights[-1])
        numerator_mantissa, numerator_exponent = frexp(numerator)
        next_weight = ldexp(
            previous_mantissa * numerator_mantissa / lag,
            previous_exponent + numerator_exponent,
        )
        if weights[-1] != 0.0 and numerator != 0.0 and next_weight == 0.0:
            raise ValueError(f"binomial coefficient at lag {lag} underflows binary64")
        weights.append(next_weight)
    warmup = min(len(request.values), request.width - 1)
    values: list[float | None] = [None] * warmup
    statuses: list[Status] = ["warmup"] * warmup
    underflows = 0
    for index in range(request.width - 1, len(request.values)):
        result, rounded_to_zero = _lag_sum(weights, request.values, index)
        values.append(result)
        statuses.append("rounded_to_zero" if rounded_to_zero else "finite")
        underflows += rounded_to_zero
    return Output(
        values=values,
        status=statuses,
        order=request.order,
        width=request.width,
        weights_newest_first=weights,
        coefficient_sum_dc_gain=fsum(weights),
        coefficient_l1_norm=fsum(map(abs, weights)),
        nonzero_weight_count=sum(weight != 0.0 for weight in weights),
        warmup_count=warmup,
        computed_count=len(request.values) - warmup,
        result_underflow_count=underflows,
    )


OPERATION = Operation(
    id="features.fractional_difference",
    kind="feature",
    description=(
        "Apply a fixed-width binomial fractional-difference polynomial with order in [0,2], "
        "explicit full-window warmup, newest-first weights and finite-width DC gain. "
        "No stationarity claim; caller controls observation order and point-in-time selection."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
