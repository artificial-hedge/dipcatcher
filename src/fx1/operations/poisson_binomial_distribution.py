"""Exact integer convolution for a sum of independent Bernoulli variables.

Multiply the polynomials (1-p_i)+p_i*z with a common power-of-two denominator
from the supplied binary64 probabilities. No odds division, Fourier inversion,
normal approximation or equal-probability assumption is used. Zero/one trials
remain exact. The full PMF is retained as integer numerators over one shared
denominator, with numerical PMF/CDF/strict-survival and logarithm previews.

Quantiles select the first supported count whose exact CDF reaches the supplied
level; level zero explicitly selects the minimum positive-mass support point.
Numerical previews can underflow or reach a boundary and are flagged. Exact
coefficients remain authoritative. Independence and forecast timing are caller
assumptions; this feature does not establish calibration or score forecasts.

Bounds: 256 trials, 256 quantiles and an 8192-bit common denominator. The sum
of input denominator exponents is checked before convolution. Output has at
most 257 masses and one denominator string. Definition and convolution context:
https://doi.org/10.1016/j.csda.2018.01.007
"""

from __future__ import annotations

from fractions import Fraction
from math import log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Probability = Annotated[float, Field(strict=True, ge=0, le=1)]


class Input(InputModel):
    probabilities: list[Probability] = Field(min_length=1, max_length=256)
    quantile_levels: list[Probability] = Field(default_factory=list, max_length=256)

    @model_validator(mode="after")
    def denominator_budget(self) -> Self:
        exponent = sum(value.as_integer_ratio()[1].bit_length() - 1 for value in self.probabilities)
        if exponent >= 8192:
            raise ValueError("exact convolution denominator would exceed 8192 bits")
        return self


class Mass(OutputModel):
    count: int
    numerator: str
    probability: float
    cumulative_probability: float
    strict_survival_probability: float
    log_probability: float | None
    status: Literal["positive", "zero_support"]
    probability_underflow: bool
    cdf_rounded_to_boundary: bool
    survival_rounded_to_boundary: bool
    log_probability_rounded_to_zero: bool


class Quantile(OutputModel):
    query_index: int
    level: float
    count: int
    cumulative_numerator: str


class Output(OutputModel):
    trial_count: int
    certain_successes: int
    certain_failures: int
    uncertain_trials: int
    minimum_support: int
    maximum_support: int
    common_denominator: str
    denominator_bits: int
    mean: float
    variance: float
    standard_deviation: float
    variance_positive_underflow: bool
    masses: list[Mass]
    quantiles: list[Quantile]
    exact_mass_sum_is_one: Literal[True] = True
    zero_quantile_convention: Literal["minimum_positive_mass_support"] = (
        "minimum_positive_mass_support"
    )
    independent_bernoulli_trials_assumed: Literal[True] = True
    independence_verified: Literal[False] = False
    forecast_timing_verified: Literal[False] = False


def _log_probability(numerator: int, denominator: int) -> float:
    if 2 * numerator > denominator:
        return log1p(float(Fraction(numerator - denominator, denominator)))
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        scaled = Fraction(numerator, denominator << exponent)
    else:
        scaled = Fraction(numerator << -exponent, denominator)
    return log(float(scaled)) + exponent * log(2)


def execute(request: Input, context: OperationContext) -> Output:
    coefficients = [1]
    denominator = 1
    mean = variance = Fraction()
    successes = failures = 0
    for probability in request.probabilities:
        numerator, divisor = probability.as_integer_ratio()
        failure = divisor - numerator
        updated = [0] * (len(coefficients) + 1)
        for count, coefficient in enumerate(coefficients):
            updated[count] += coefficient * failure
            updated[count + 1] += coefficient * numerator
        coefficients = updated
        denominator *= divisor
        exact = Fraction(numerator, divisor)
        mean += exact
        variance += exact * (1 - exact)
        successes += numerator == divisor
        failures += numerator == 0
    if sum(coefficients) != denominator:
        raise ValueError("internal exact convolution failed mass conservation")
    maximum = len(request.probabilities) - failures
    masses: list[Mass] = []
    cumulative_numerators: list[int] = []
    cumulative = 0
    for count, numerator in enumerate(coefficients):
        cumulative += numerator
        cumulative_numerators.append(cumulative)
        probability = float(Fraction(numerator, denominator))
        cdf = float(Fraction(cumulative, denominator))
        survival = float(Fraction(denominator - cumulative, denominator))
        logarithm = _log_probability(numerator, denominator) if numerator else None
        masses.append(
            Mass(
                count=count,
                numerator=str(numerator),
                probability=probability,
                cumulative_probability=cdf,
                strict_survival_probability=survival,
                log_probability=logarithm,
                status="positive" if numerator else "zero_support",
                probability_underflow=bool(numerator and probability == 0),
                cdf_rounded_to_boundary=0 < cumulative < denominator and cdf in (0, 1),
                survival_rounded_to_boundary=0 < cumulative < denominator and survival in (0, 1),
                log_probability_rounded_to_zero=0 < numerator < denominator and logarithm == 0,
            )
        )
    quantiles: list[Quantile] = []
    for query_index, level in enumerate(request.quantile_levels):
        threshold = Fraction(level) * denominator
        selected = next(
            count
            for count in range(successes, maximum + 1)
            if cumulative_numerators[count] >= threshold
        )
        quantiles.append(
            Quantile(
                query_index=query_index,
                level=level,
                count=selected,
                cumulative_numerator=str(cumulative_numerators[selected]),
            )
        )
    return Output(
        trial_count=len(request.probabilities),
        certain_successes=successes,
        certain_failures=failures,
        uncertain_trials=len(request.probabilities) - successes - failures,
        minimum_support=successes,
        maximum_support=maximum,
        common_denominator=str(denominator),
        denominator_bits=denominator.bit_length(),
        mean=float(mean),
        variance=float(variance),
        standard_deviation=correctly_rounded_sqrt(variance.numerator, variance.denominator),
        variance_positive_underflow=bool(variance and float(variance) == 0),
        masses=masses,
        quantiles=quantiles,
    )


OPERATION = Operation(
    id="features.poisson_binomial_distribution",
    kind="feature",
    description="Convolve heterogeneous independent Bernoulli trials exactly into a finite count law, retaining integer masses, tail previews and exact-CDF quantile selection.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
