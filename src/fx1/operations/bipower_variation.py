"""Realized squared variation and adjacent absolute-product bipower variation.

RV = sum(r[i]**2). BV = (pi/2)*factor*sum(abs(r[i-1])*abs(r[i])).
The named factor is 1 by default or n/(n-1) when explicitly selected. Values
are not annualized or demeaned. They are descriptive realized features, not
a jump-significance test or a claim of integrated-variance consistency for
arbitrary data. Caller controls ordered, equally spaced return construction
and point-in-time source availability.

Base normalization: Barndorff-Nielsen and Shephard (2004),
https://public.econ.duke.edu/~get/browse/courses/883/Spr16/COURSE-MATERIALS/Z_Papers/BNSJFEC2004.pdf
The optional n/(n-1) policy explicitly compensates for the n-1 adjacent pairs.
"""

from collections.abc import Iterable
from itertools import pairwise
from math import frexp, fsum, ldexp, pi
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Return = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Correction = Literal["none", "n_over_n_minus_1"]


class Input(InputModel):
    returns: list[Return] = Field(min_length=1, max_length=10_000)
    finite_sample_policy: Correction = "none"


class Output(OutputModel):
    observation_count: int
    adjacent_pair_count: int
    realized_variation: Nonnegative
    adjacent_absolute_product_sum: Nonnegative | None
    bipower_variation: Nonnegative | None
    finite_sample_policy: Correction
    applied_finite_sample_factor: Nonnegative | None
    bipower_status: Literal["finite", "insufficient_pairs"]


def _product_sum(pairs: Iterable[tuple[float, float]], multiplier: float = 1.0) -> float:
    """Sum nonnegative products without prematurely underflowing each term."""
    products: list[tuple[float, int]] = []
    for left, right in pairs:
        if left == 0.0 or right == 0.0:
            continue
        left_mantissa, left_exponent = frexp(abs(left))
        right_mantissa, right_exponent = frexp(abs(right))
        products.append((left_mantissa * right_mantissa, left_exponent + right_exponent))
    if not products:
        return 0.0
    largest_exponent = max(exponent for _, exponent in products)
    scaled = fsum(ldexp(mantissa, exponent - largest_exponent) for mantissa, exponent in products)
    return ldexp(scaled * multiplier, largest_exponent)


def execute(request: Input, context: OperationContext) -> Output:
    """Preserve the requested normalization and expose insufficient pair history."""
    count = len(request.returns)
    realized = _product_sum((value, value) for value in request.returns)
    product_sum: float | None = None
    bipower: float | None = None
    factor: float | None = None
    if count >= 2:
        factor = count / (count - 1) if request.finite_sample_policy == "n_over_n_minus_1" else 1.0
        product_sum = _product_sum(pairwise(request.returns))
        bipower = _product_sum(pairwise(request.returns), multiplier=(pi / 2.0) * factor)
    return Output(
        observation_count=count,
        adjacent_pair_count=count - 1,
        realized_variation=realized,
        adjacent_absolute_product_sum=product_sum,
        bipower_variation=bipower,
        finite_sample_policy=request.finite_sample_policy,
        applied_finite_sample_factor=factor,
        bipower_status="finite" if count >= 2 else "insufficient_pairs",
    )


OPERATION = Operation(
    id="features.bipower_variation",
    kind="feature",
    description=(
        "Compute realized squared variation and pi/2-normalized adjacent absolute-product "
        "bipower variation. The optional n/(n-1) correction is named explicitly; one return "
        "has insufficient bipower history. No annualization or jump significance is implied."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
