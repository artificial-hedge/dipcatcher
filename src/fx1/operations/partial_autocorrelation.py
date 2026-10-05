"""Biased-autocovariance Yule-Walker PACF through exact Levinson-Durbin stages.

For a supplied evenly spaced block, center by its exact sample mean or zero,
then gamma_k=sum(t=k..N-1,z_t*z_(t-k))/N. Every lag uses N, with no missing
values, taper, lag-specific degrees-of-freedom adjustment or FFT. Integer
centered products and rational recursion avoid cancellation from large levels.
Let rho_k=gamma_k/gamma_0 and E_0=1. Stage k sets
phi_kk=(rho_k-sum(j=1..k-1,phi_(k-1,j)*rho_(k-j)))/E_(k-1),
phi_kj=phi_(k-1,j)-phi_kk*phi_(k-1,k-j), E_k=E_(k-1)*(1-phi_kk^2).

PACF at zero is one only for nonzero centered energy. A zero-energy block has
undefined PACF (null entries) and no fitted lags. Exact singularity and a
nonunit reflection rounding to a unit boundary stop explicitly. Fractions
are limited to 65536 numerator/denominator bits; nonzero output underflow
fails. All completed stages use exact, not returned rounded, coefficients.
Innovation energies are the Toeplitz prediction recursion, not fitted residual
sample variances. Whole-block fitting does not establish stationarity or PIT.
Conventions and recurrence reference:
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.levinson_durbin.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.yule_walker.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Centering = Literal["demean", "none"]
Status = Literal["fitted", "zero_energy", "singular_prediction", "numerical_boundary"]
_BIT_LIMIT = 65_536


class Input(InputModel):
    values: list[Value] = Field(min_length=2, max_length=4_096)
    order: int = Field(default=1, strict=True, ge=0, le=64)
    centering: Centering = "demean"

    @model_validator(mode="after")
    def bounded_order(self) -> Self:
        if self.order >= len(self.values):
            raise ValueError("order must be smaller than the observation count")
        if len(self.values) * (self.order + 1) > 131_072:
            raise ValueError("autocovariance work exceeds 131072 observation-lag cells")
        return self


class Stage(OutputModel):
    lag: int
    partial_autocorrelation: float
    autoregressive_coefficients: list[float]
    innovation_variance: float
    innovation_to_initial_variance_ratio: float


class Output(OutputModel):
    observation_count: int
    requested_order: int
    fitted_order: int
    centering: Centering
    centering_offset: float
    constant_input: bool
    status: Status
    stopping_lag: int | None
    stopping_rounded_reflection: float | None
    autocovariances: list[float]
    autocorrelations: list[float | None]
    partial_autocorrelations: list[float | None]
    stages: list[Stage]
    autoregressive_coefficients: list[float]
    intercept_from_returned_coefficients: float
    initial_variance: float
    innovation_variance: float
    autocovariance_divisor: Literal["observation_count_at_every_lag"] = (
        "observation_count_at_every_lag"
    )
    evenly_spaced_input_verified: Literal[False] = False
    stationarity_verified: Literal[False] = False
    timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _BIT_LIMIT:
        raise ValueError("exact Levinson-Durbin arithmetic exceeds 65536-bit budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
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
    center = Fraction(sum(units), count << places) if request.centering == "demean" else Fraction()
    if request.centering == "demean":
        total = sum(units)
        centered = [count * value - total for value in units]
        scale = count << places
    else:
        centered, scale = units, 1 << places
    products = [
        sum(centered[index] * centered[index - lag] for index in range(lag, count))
        for lag in range(request.order + 1)
    ]
    covariance = [Fraction(product, count * scale * scale) for product in products]
    variance = covariance[0]
    autocorrelations: list[Fraction] = (
        [Fraction(product, products[0]) for product in products] if products[0] else []
    )
    coefficients: list[Fraction] = []
    energy_ratio = Fraction(1)
    stages: list[Stage] = []
    pacf: list[float | None] = [1.0] if variance else [None]
    status: Status = "fitted" if variance else "zero_energy"
    stopping_lag: int | None = None
    stopping_reflection: float | None = None
    if variance:
        for lag in range(1, request.order + 1):
            if not energy_ratio:
                status, stopping_lag = "singular_prediction", lag
                break
            numerator = autocorrelations[lag]
            for index, coefficient in enumerate(coefficients, start=1):
                numerator = _bounded(numerator - coefficient * autocorrelations[lag - index])
            reflection = _bounded(numerator / energy_ratio)
            if abs(reflection) > 1:
                raise ValueError("biased autocovariance recursion lost positive semidefiniteness")
            rounded = _number(reflection, "partial autocorrelation")
            if abs(reflection) < 1 and abs(rounded) == 1:
                status, stopping_lag, stopping_reflection = "numerical_boundary", lag, rounded
                break
            updated = [
                _bounded(value - reflection * coefficients[-index - 1])
                for index, value in enumerate(coefficients)
            ] + [reflection]
            energy_ratio = _bounded(energy_ratio * (1 - reflection * reflection))
            coefficients = updated
            pacf.append(rounded)
            stages.append(
                Stage(
                    lag=lag,
                    partial_autocorrelation=rounded,
                    autoregressive_coefficients=[
                        _number(value, "stage AR coefficient") for value in coefficients
                    ],
                    innovation_variance=_number(
                        variance * energy_ratio, "stage innovation variance"
                    ),
                    innovation_to_initial_variance_ratio=_number(energy_ratio, "innovation ratio"),
                )
            )
            if not energy_ratio:
                status, stopping_lag, stopping_reflection = "singular_prediction", lag, rounded
                break
    reported = [_number(value, "AR coefficient") for value in coefficients]
    pacf.extend([None] * (request.order + 1 - len(pacf)))
    return Output(
        observation_count=count,
        requested_order=request.order,
        fitted_order=len(stages),
        centering=request.centering,
        centering_offset=_number(center, "centering offset"),
        constant_input=all(value == request.values[0] for value in request.values),
        status=status,
        stopping_lag=stopping_lag,
        stopping_rounded_reflection=stopping_reflection,
        autocovariances=[_number(value, "autocovariance") for value in covariance],
        autocorrelations=(
            [_number(value, "autocorrelation") for value in autocorrelations]
            if variance
            else [None] * (request.order + 1)
        ),
        partial_autocorrelations=pacf,
        stages=stages,
        autoregressive_coefficients=reported,
        intercept_from_returned_coefficients=_number(
            center * (1 - sum((Fraction(value) for value in reported), Fraction())), "intercept"
        ),
        initial_variance=_number(variance, "initial variance"),
        innovation_variance=_number(variance * energy_ratio, "innovation variance"),
    )


OPERATION = Operation(
    id="features.partial_autocorrelation",
    kind="feature",
    description=(
        "Estimate PACF and AR prediction coefficients from biased autocovariances using exact "
        "bounded Levinson-Durbin recursion, with centering, stage energies and degeneracy status."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
