"""Log score and closed-form CRPS for supplied heteroscedastic normal mixtures.

For normalized component masses w_i, CRPS is sum_i w_i*A(y-mu_i,sigma_i)
minus half sum_ij w_i*w_j*A(mu_i-mu_j,sqrt(sigma_i^2+sigma_j^2)), where
A(d,s)=E|N(d,s^2)|. Exact location differences and rational weighted arithmetic
retain the atomic absolute-distance contribution; only the positive Gaussian
excess above |d| is approximated. A scaled log excess avoids discarding a
representable physical correction when the unit-normal density underflows.

Mixture log density uses log-sum-exp. Normal excess uses erfc below |z|=12 and
a bounded alternating Mills expansion above it. Logs, exponentials and pairwise
combined standard deviations remain approximate. A negative or zero computed
CRPS fails instead of being clipped. Underflowing excess terms and component
responsibilities are counted, and PIT rounding to zero/one is reported.

No mixture fitting or outcome-based component selection occurs. Density losses
depend on the required measurement unit and can be negative. Proper scoring
requires forecasts and observation weights selected before outcomes; those
assumptions are not verified. Limits: 256 forecasts, 32 components each, 32768
component-pair work cells, positive scales/weights in [1e-100,1e100], and at most
one million standard deviations in every scored observation/component or
component-pair comparison. Rational accumulators and converted fractions are
capped at 131072 bits, including aggregation across different mixture weights.
Formula reference:
https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_mixnorm.html
"""

from __future__ import annotations

from fractions import Fraction
from math import erfc, exp, fsum, isfinite, log, log1p, pi, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Positive = Annotated[float, Field(strict=True, ge=1e-100, le=1e100)]
_LOG_NORMALIZER = 0.5 * log(2 * pi)
_SQRT_TWO = sqrt(2)


class Component(InputModel):
    mean: Value
    standard_deviation: Positive
    mass: float = Field(default=1, strict=True, ge=0, le=1e100)


class Forecast(InputModel):
    outcome: Value
    components: list[Component] = Field(min_length=1, max_length=32)
    observation_weight: Positive = 1.0

    @model_validator(mode="after")
    def positive_total_mass(self) -> Self:
        if not any(component.mass for component in self.components):
            raise ValueError("mixture requires positive component mass")
        if any(0 < component.mass < 1e-100 for component in self.components):
            raise ValueError("positive mixture masses must be at least 1e-100")
        return self


class Input(InputModel):
    forecasts: list[Forecast] = Field(min_length=1, max_length=256)
    measurement_unit: str = Field(strict=True, min_length=1, max_length=64)

    @model_validator(mode="after")
    def pair_work_budget(self) -> Self:
        if sum(len(row.components) ** 2 for row in self.forecasts) > 32_768:
            raise ValueError("mixture scoring exceeds 32768 component-pair work cells")
        return self


class RowScore(OutputModel):
    row_index: int
    component_count: int
    positive_component_count: int
    observation_weight: float
    normalized_component_masses: list[float]
    posterior_component_responsibilities: list[float]
    negative_log_density: float
    crps: float
    pit: float
    pit_rounded_to_boundary: bool
    underflowed_gaussian_excess_terms: int
    underflowed_component_responsibilities: int
    underflowed_standardized_quadratics: int


class Output(OutputModel):
    observation_count: int
    measurement_unit: str
    weighted_mean_negative_log_density: float
    weighted_mean_crps: float
    rows: list[RowScore]
    mixture_parameters_fitted: Literal[False] = False
    component_selection_verified: Literal[False] = False
    outcome_independent_weights_verified: Literal[False] = False
    forecast_timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 131_072:
        raise ValueError("Gaussian mixture arithmetic exceeds 131072-bit rational budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    result = float(value)
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _standardized(delta: Fraction, scale: Fraction) -> tuple[float, float, bool]:
    standardized = delta / scale
    if abs(standardized) > 1_000_000:
        raise ValueError("mixture separation exceeds one million standard deviations")
    quadratic = standardized * standardized / 2
    return (
        _number(standardized, "standardized displacement"),
        float(quadratic),
        bool(quadratic and float(quadratic) == 0),
    )


def _absolute_moment(delta: Fraction, scale: Fraction) -> tuple[Fraction, bool]:
    z, quadratic, _tiny = _standardized(abs(delta), scale)
    if z < 12:
        density = exp(-quadratic - _LOG_NORMALIZER)
        factor = 1 - z * (0.5 * erfc(z / _SQRT_TWO)) / density
    else:
        inverse_square = 1 / (z * z)
        term = inverse_square
        terms = [term]
        for index in range(1, 65):
            following = -term * (2 * index + 1) * inverse_square
            if abs(following) >= abs(term):
                break
            terms.append(following)
            term = following
            if abs(term) <= 1e-18 * inverse_square:
                break
        factor = fsum(terms)
    if factor <= 0:
        raise ValueError("Gaussian absolute-moment excess lost positivity")
    log_excess = fsum(
        (
            log(_number(scale, "standard deviation")),
            log(2),
            -quadratic,
            -_LOG_NORMALIZER,
            log(factor),
        )
    )
    excess = exp(log_excess)
    return abs(delta) + Fraction(excess), excess == 0


def _log_weight(value: Fraction) -> float:
    return (
        log1p(_number(value - 1, "mass deviation"))
        if value > Fraction(1, 2)
        else log(_number(value, "normalized component mass"))
    )


def execute(request: Input, context: OperationContext) -> Output:
    rows: list[RowScore] = []
    total_weight = weighted_log = weighted_crps = Fraction()
    for row_index, forecast in enumerate(request.forecasts):
        masses = [Fraction(component.mass) for component in forecast.components]
        total_mass = sum(masses, Fraction())
        weights = [mass / total_mass for mass in masses]
        active = [index for index, mass in enumerate(masses) if mass]
        means = [Fraction(component.mean) for component in forecast.components]
        scales = [Fraction(component.standard_deviation) for component in forecast.components]
        outcome = Fraction(forecast.outcome)
        first_moment = pit = Fraction()
        log_terms: list[float] = []
        excess_underflows = tiny_quadratics = 0
        for index in active:
            delta = outcome - means[index]
            z, quadratic, tiny = _standardized(delta, scales[index])
            tiny_quadratics += tiny
            moment, underflow = _absolute_moment(delta, scales[index])
            excess_underflows += underflow
            first_moment = _bounded(first_moment + weights[index] * moment)
            pit = _bounded(pit + weights[index] * Fraction(0.5 * erfc(-z / _SQRT_TWO)))
            log_terms.append(
                fsum(
                    (
                        _log_weight(weights[index]),
                        -log(float(scales[index])),
                        -quadratic,
                        -_LOG_NORMALIZER,
                    )
                )
            )
        second_moment = Fraction()
        for position, first in enumerate(active):
            for second in active[: position + 1]:
                variance = scales[first] ** 2 + scales[second] ** 2
                combined = correctly_rounded_sqrt(variance.numerator, variance.denominator)
                moment, underflow = _absolute_moment(
                    means[first] - means[second], Fraction(combined)
                )
                excess_underflows += underflow
                factor = 1 if first == second else 2
                second_moment = _bounded(
                    second_moment + factor * weights[first] * weights[second] * moment
                )
        crps = _bounded(first_moment - second_moment / 2)
        if crps <= 0:
            raise ValueError("positive Gaussian-mixture CRPS was lost to numerical approximation")
        largest = max(log_terms)
        relative = [exp(value - largest) for value in log_terms]
        log_density = fsum((largest, log(fsum(relative))))
        relative_sum = sum((Fraction(value) for value in relative), Fraction())
        responsibilities = [0.0] * len(weights)
        for index, value in zip(active, relative, strict=True):
            responsibilities[index] = float(Fraction(value) / relative_sum)
        observation_weight = Fraction(forecast.observation_weight)
        weighted_log = _bounded(weighted_log + observation_weight * Fraction(-log_density))
        weighted_crps = _bounded(weighted_crps + observation_weight * crps)
        total_weight = _bounded(total_weight + observation_weight)
        rows.append(
            RowScore(
                row_index=row_index,
                component_count=len(weights),
                positive_component_count=len(active),
                observation_weight=forecast.observation_weight,
                normalized_component_masses=[float(weight) for weight in weights],
                posterior_component_responsibilities=responsibilities,
                negative_log_density=-log_density,
                crps=_number(crps, "CRPS"),
                pit=float(pit),
                pit_rounded_to_boundary=float(pit) in (0, 1),
                underflowed_gaussian_excess_terms=excess_underflows,
                underflowed_component_responsibilities=sum(
                    responsibilities[index] == 0 for index in active
                ),
                underflowed_standardized_quadratics=tiny_quadratics,
            )
        )
    return Output(
        observation_count=len(rows),
        measurement_unit=request.measurement_unit,
        weighted_mean_negative_log_density=_number(weighted_log / total_weight, "mean log score"),
        weighted_mean_crps=_number(weighted_crps / total_weight, "mean CRPS"),
        rows=rows,
    )


OPERATION = Operation(
    id="skills.score_gaussian_mixtures",
    kind="skill",
    description="Score supplied heteroscedastic Gaussian mixtures by log density and closed-form CRPS, retaining component responsibilities, PIT values and numerical underflow diagnostics.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
