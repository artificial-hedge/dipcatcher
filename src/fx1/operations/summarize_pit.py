"""Uniform-reference and serial diagnostics for supplied probability integral transforms.

The empirical-CDF discrepancy follows the one-sample KS statistic definition:
https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
No p-values or exchangeability claims are inferred from a sequence of PIT values.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from math import fsum, isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Lag = Annotated[int, Field(strict=True, ge=1, le=1000)]


class Input(InputModel):
    values: list[Probability] = Field(min_length=1, max_length=50_000)
    bin_edges: list[Probability] = Field(min_length=2, max_length=101)
    lags: list[Lag] = Field(default_factory=lambda: [1], max_length=20)

    @model_validator(mode="after")
    def validate_edges(self) -> Self:
        if self.bin_edges[0] != 0 or self.bin_edges[-1] != 1:
            raise ValueError("bin_edges must start at zero and end at one")
        if any(
            left >= right
            for left, right in zip(self.bin_edges[:-1], self.bin_edges[1:], strict=True)
        ):
            raise ValueError("bin_edges must be strictly increasing")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique")
        return self


class Bin(OutputModel):
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    observed_fraction: float
    uniform_reference_fraction: float


class SerialDiagnostic(OutputModel):
    lag: int
    pair_count: int
    correlation: float | None
    status: Literal["finite", "too_few_pairs", "constant_margin"]


class Output(OutputModel):
    observation_count: int
    mean: float
    standard_deviation: float
    zero_count: int
    one_count: int
    distinct_value_count: int
    bins: list[Bin]
    cdf_positive_discrepancy: float
    cdf_negative_discrepancy: float
    ks_statistic: float
    cramer_von_mises_statistic: float
    serial_diagnostics: list[SerialDiagnostic]
    input_order: Literal["preserved_for_serial_diagnostics"] = "preserved_for_serial_diagnostics"
    p_values_computed: Literal[False] = False
    calibration_certified: Literal[False] = False


def _centered(values: list[float]) -> tuple[list[int], int, Fraction]:
    """Represent centered values exactly as integer/(n*2**places)."""
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    total = sum(units)
    size = len(values)
    return [size * value - total for value in units], places, Fraction(total, size << places)


def _sqrt_ratio(numerator: int, denominator: int) -> float:
    """Round the exact rational root once and reject nonzero-result underflow."""
    result = correctly_rounded_sqrt(numerator, denominator)
    if not isfinite(result) or (numerator != 0 and result == 0):
        raise ValueError("PIT dispersion/correlation is outside the supported floating-point range")
    return result


def _serial(values: list[float], lag: int) -> SerialDiagnostic:
    size = max(0, len(values) - lag)
    if size < 2:
        return SerialDiagnostic(lag=lag, pair_count=size, correlation=None, status="too_few_pairs")
    dx, _, _ = _centered(values[:size])
    dy, _, _ = _centered(values[lag:])
    xx, yy = sum(value * value for value in dx), sum(value * value for value in dy)
    if xx == 0 or yy == 0:
        return SerialDiagnostic(
            lag=lag, pair_count=size, correlation=None, status="constant_margin"
        )
    xy = sum(a * b for a, b in zip(dx, dy, strict=True))
    coefficient = _sqrt_ratio(xy * xy, xx * yy) * (-1 if xy < 0 else 1)
    if not isfinite(coefficient) or abs(coefficient) > 1 + 1e-12:
        raise ValueError("PIT serial correlation exceeds numerical range")
    return SerialDiagnostic(
        lag=lag,
        pair_count=size,
        correlation=max(-1.0, min(1.0, coefficient)),
        status="finite",
    )


def execute(request: Input, context: OperationContext) -> Output:
    """Summarize marginal shape and supplied-order lags without fitting a reference."""
    values = request.values
    size = len(values)
    ordered = sorted(values)
    counts = [0] * (len(request.bin_edges) - 1)
    for value in values:
        index = min(bisect_right(request.bin_edges, value) - 1, len(counts) - 1)
        counts[index] += 1
    deviations, places, exact_mean = _centered(values)
    mean = float(exact_mean)
    if exact_mean and mean == 0:
        raise ValueError("PIT mean is below the supported floating-point range")
    standard_deviation = _sqrt_ratio(
        sum(value * value for value in deviations), size**3 << (2 * places)
    )
    d_plus = max((index + 1) / size - value for index, value in enumerate(ordered))
    d_minus = max(value - index / size for index, value in enumerate(ordered))
    cvm = 1 / (12 * size) + fsum(
        (value - (2 * index + 1) / (2 * size)) ** 2 for index, value in enumerate(ordered)
    )
    return Output(
        observation_count=size,
        mean=mean,
        standard_deviation=standard_deviation,
        zero_count=values.count(0),
        one_count=values.count(1),
        distinct_value_count=len(set(values)),
        bins=[
            Bin(
                lower=request.bin_edges[index],
                upper=request.bin_edges[index + 1],
                upper_inclusive=index == len(counts) - 1,
                count=count,
                observed_fraction=count / size,
                uniform_reference_fraction=request.bin_edges[index + 1] - request.bin_edges[index],
            )
            for index, count in enumerate(counts)
        ],
        cdf_positive_discrepancy=d_plus,
        cdf_negative_discrepancy=d_minus,
        ks_statistic=max(d_plus, d_minus),
        cramer_von_mises_statistic=cvm,
        serial_diagnostics=[_serial(values, lag) for lag in request.lags],
    )


OPERATION = Operation(
    id="skills.summarize_pit",
    kind="skill",
    description=(
        "Summarize supplied PIT values with explicit bins, endpoint counts, KS/Cramer-von-Mises "
        "discrepancies and ordered lag correlations; no p-values or calibration certification."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
