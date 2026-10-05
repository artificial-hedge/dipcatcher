"""Logarithmic scoring of explicit Poisson, binomial and negative-binomial laws.

This one operation owns all three count-family branches rather than counting
parameterizations as separate capabilities. Outcomes are nonnegative integers.
Poisson uses its mean; binomial uses trial count and success probability;
negative binomial uses mean mu and positive shape r, with variance mu+mu^2/r.
Parameters are supplied predictions, never fitted from the scored outcomes.

Log losses are evaluated directly without forming tiny probability masses.
Binomial coefficients use bounded sums of log ratios; negative-binomial rising
factorials use log1p ratios, avoiding subtraction of large log-gamma values.
Poisson uses lgamma(k+1). Elementary functions remain binary64 approximations.
Impossible outcomes retain infinite-loss status, with null aggregate loss if
any positive-weight row is impossible; there is no clipping or finite-only mean.

Count/trial bounds are one million, positive means/probabilities at least 1e-100,
means at most one million, NB shape 1e-6 through 1e6, and 200000 total coefficient
terms. Weights are positive, bounded and fixed by the caller; weighted scores are
proper only when weighting is independent of realized outcomes. No timing or
distributional adequacy is certified. Reference probability mass functions:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html
"""

from __future__ import annotations

from fractions import Fraction
from math import fsum, isfinite, lgamma, log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Count = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
Weight = Annotated[float, Field(strict=True, ge=1e-100, le=1e100)]


class Poisson(InputModel):
    family: Literal["poisson"]
    mean: float = Field(strict=True, ge=0, le=1_000_000)

    @field_validator("mean")
    @classmethod
    def supported_mean(cls, value: float) -> float:
        if 0 < value < 1e-100:
            raise ValueError("positive count means must be at least 1e-100")
        return value


class Binomial(InputModel):
    family: Literal["binomial"]
    trials: Count
    probability: float = Field(strict=True, ge=0, le=1)

    @field_validator("probability")
    @classmethod
    def supported_probability(cls, value: float) -> float:
        if 0 < value < 1e-100:
            raise ValueError("positive binomial probability must be at least 1e-100")
        return value


class NegativeBinomial(InputModel):
    family: Literal["negative_binomial"]
    mean: float = Field(strict=True, ge=0, le=1_000_000)
    shape: float = Field(strict=True, ge=1e-6, le=1e6)

    @field_validator("mean")
    @classmethod
    def supported_mean(cls, value: float) -> float:
        if 0 < value < 1e-100:
            raise ValueError("positive count means must be at least 1e-100")
        return value


Forecast = Annotated[Poisson | Binomial | NegativeBinomial, Field(discriminator="family")]


class Input(InputModel):
    outcomes: list[Count] = Field(min_length=1, max_length=1000)
    forecasts: list[Forecast] = Field(min_length=1, max_length=1000)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def aligned_bounded_work(self) -> Self:
        if len(self.outcomes) != len(self.forecasts):
            raise ValueError("one count forecast is required per outcome")
        if self.weights is not None and len(self.weights) != len(self.outcomes):
            raise ValueError("weights must align with outcomes")
        work = sum(
            _terms(outcome, forecast)
            for outcome, forecast in zip(self.outcomes, self.forecasts, strict=True)
        )
        if work > 200_000:
            raise ValueError("count scoring exceeds 200000 coefficient terms")
        return self


class RowScore(OutputModel):
    row_index: int
    family: Literal["poisson", "binomial", "negative_binomial"]
    outcome: int
    forecast_mean: float
    forecast_variance: float
    observation_weight: float
    log_loss: float | None
    status: Literal["finite", "infinite_zero_probability"]
    coefficient_terms: int


class Output(OutputModel):
    observation_count: int
    family_counts: dict[str, int]
    total_observation_weight: float
    weighted_mean_log_loss: float | None
    aggregate_status: Literal["finite", "infinite_zero_probability"]
    infinite_loss_rows: int
    coefficient_terms: int
    rows: list[RowScore]
    score_units: Literal["natural_log_probability_mass"] = "natural_log_probability_mass"
    probability_clipping_applied: Literal[False] = False
    outcome_independent_weights_validated: Literal[False] = False
    forecast_timing_validated: Literal[False] = False


def _terms(outcome: int, forecast: Poisson | Binomial | NegativeBinomial) -> int:
    if isinstance(forecast, Binomial):
        if outcome > forecast.trials or forecast.probability in (0, 1):
            return 0
        return min(outcome, forecast.trials - outcome)
    if isinstance(forecast, NegativeBinomial) and forecast.mean:
        return outcome
    return 0


def _loss(outcome: int, forecast: Poisson | Binomial | NegativeBinomial) -> float | None:
    if isinstance(forecast, Binomial):
        trials, probability = forecast.trials, forecast.probability
        if outcome > trials:
            return None
        if probability == 0:
            return 0.0 if outcome == 0 else None
        if probability == 1:
            return 0.0 if outcome == trials else None
        smaller = min(outcome, trials - outcome)
        log_coefficient = fsum(log1p((trials - smaller) / index) for index in range(1, smaller + 1))
        result = fsum(
            (
                -log_coefficient,
                -outcome * log(probability),
                -(trials - outcome) * log1p(-probability),
            )
        )
    elif forecast.mean == 0:
        return 0.0 if outcome == 0 else None
    elif isinstance(forecast, Poisson):
        result = fsum((forecast.mean, -outcome * log(forecast.mean), lgamma(outcome + 1)))
    else:
        shape, mean = forecast.shape, forecast.mean
        log_coefficient = (
            fsum([log(shape)] + [log1p((shape - 1) / (index + 1)) for index in range(1, outcome)])
            if outcome
            else 0.0
        )
        result = fsum(
            (shape * log1p(mean / shape), outcome * log1p(shape / mean), -log_coefficient)
        )
    if not isfinite(result) or result < 0:
        raise ValueError("count log loss is nonfinite or negative after numerical evaluation")
    # Only a zero-trial binomial can be exactly deterministic in the interior branch.
    if result == 0 and not (isinstance(forecast, Binomial) and forecast.trials == 0):
        raise ValueError("positive count log loss rounded to zero")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    weights = (
        [Fraction(weight) for weight in request.weights]
        if request.weights is not None
        else [Fraction(1)] * len(request.outcomes)
    )
    weighted_loss = Fraction()
    infinite = work = 0
    family_counts: dict[str, int] = {}
    rows: list[RowScore] = []
    for index, (outcome, forecast, weight) in enumerate(
        zip(request.outcomes, request.forecasts, weights, strict=True)
    ):
        family_counts[forecast.family] = family_counts.get(forecast.family, 0) + 1
        if isinstance(forecast, Binomial):
            probability = Fraction(forecast.probability)
            mean = forecast.trials * probability
            variance = mean * (1 - probability)
        else:
            mean = Fraction(forecast.mean)
            variance = (
                mean
                if isinstance(forecast, Poisson)
                else mean + mean * mean / Fraction(forecast.shape)
            )
        loss = _loss(outcome, forecast)
        if loss is None:
            infinite += 1
        else:
            weighted_loss += weight * Fraction(loss)
        terms = _terms(outcome, forecast)
        work += terms
        rows.append(
            RowScore(
                row_index=index,
                family=forecast.family,
                outcome=outcome,
                forecast_mean=float(mean),
                forecast_variance=float(variance),
                observation_weight=float(weight),
                log_loss=loss,
                status="finite" if loss is not None else "infinite_zero_probability",
                coefficient_terms=terms,
            )
        )
    total_weight = sum(weights, Fraction())
    average = weighted_loss / total_weight
    converted = float(average)
    if not isfinite(converted) or (average and converted == 0):
        raise ValueError("weighted count log loss is outside representable finite range")
    return Output(
        observation_count=len(rows),
        family_counts=dict(sorted(family_counts.items())),
        total_observation_weight=float(total_weight),
        weighted_mean_log_loss=None if infinite else converted,
        aggregate_status="infinite_zero_probability" if infinite else "finite",
        infinite_loss_rows=infinite,
        coefficient_terms=work,
        rows=rows,
    )


OPERATION = Operation(
    id="skills.score_count_forecasts",
    kind="skill",
    description="Evaluate proper logarithmic losses for supplied Poisson, binomial and mean/shape negative-binomial count forecasts, with bounded coefficient sums, exact weight aggregation and explicit infinite losses for impossible outcomes.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
