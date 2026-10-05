"""IPCW Brier scores for right-censored outcomes and a supplied censoring curve.

At horizon t, observed events with T<=t contribute S(t)^2/G(T-), subjects
with T>t contribute (1-S(t))^2/G(t), and subjects censored by t contribute
zero. The divisor remains the full subject count, not the informative count
or the weight sum. Event observation uses the convention event_time<=censor_time;
the event weight therefore uses the left limit of G(u)=P(C>u). Ties and
right-continuous curve evaluation are explicit rather than inferred.

The caller supplies G and its valid-through horizon. This operation neither
fits nor validates the censoring model. Interpretation requires noninformative
censoring for this marginal curve, positive relevant survival, held-out
forecasts and appropriate sampling. No left truncation or competing risks.
Reference: Graf et al. (1999), as documented in scikit-survival's Brier score:
https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Time = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
SurvivalRow = Annotated[list[Probability], Field(min_length=1, max_length=200)]


class Outcome(InputModel):
    duration: Time
    event_observed: bool = Field(strict=True)


class CensoringKnot(InputModel):
    time: Time
    survival: Probability


class Input(InputModel):
    outcomes: list[Outcome] = Field(min_length=1, max_length=2000)
    horizons: list[Time] = Field(min_length=1, max_length=200)
    survival_probabilities: list[SurvivalRow] = Field(min_length=1, max_length=2000)
    censoring_curve: list[CensoringKnot] = Field(max_length=2000)
    censoring_valid_through: Time

    @model_validator(mode="after")
    def coherent_survival(self) -> Self:
        if any(right <= left for left, right in pairwise(self.horizons)):
            raise ValueError("horizons must be strictly increasing")
        if len(self.outcomes) != len(self.survival_probabilities):
            raise ValueError("survival forecast rows must align with outcomes")
        if len(self.outcomes) * len(self.horizons) > 100_000:
            raise ValueError("survival scoring exceeds 100000 subject-horizon cells")
        for row in self.survival_probabilities:
            if len(row) != len(self.horizons) or any(b > a for a, b in pairwise(row)):
                raise ValueError("survival rows must align with horizons and be nonincreasing")
        if any(
            right.time <= left.time or right.survival > left.survival
            for left, right in pairwise(self.censoring_curve)
        ):
            raise ValueError("censoring knots need increasing times and nonincreasing survival")
        if self.horizons[-1] > self.censoring_valid_through or (
            self.censoring_curve and self.censoring_curve[-1].time > self.censoring_valid_through
        ):
            raise ValueError("horizons and censoring knots must lie within declared curve support")
        return self


class HorizonScore(OutputModel):
    horizon: float
    ipcw_brier_score: float
    event_contribution: float
    survivor_contribution: float
    observed_event_count: int
    known_survivor_count: int
    censored_by_horizon_count: int
    informative_count: int
    censoring_survival_at_horizon: float
    maximum_applied_weight: float | None
    applied_weight_sum: float


class Output(OutputModel):
    subject_count: int
    horizon_count: int
    scores: list[HorizonScore]
    integrated_brier_score: float | None
    integration_status: Literal["trapezoid_over_supplied_horizons", "single_horizon"]
    event_weight_evaluation: Literal["G_at_event_left_limit"] = "G_at_event_left_limit"
    survivor_weight_evaluation: Literal["G_at_horizon_right_continuous"] = (
        "G_at_horizon_right_continuous"
    )
    score_denominator: Literal["all_subjects"] = "all_subjects"
    censoring_valid_through: float
    censoring_curve_fitted: Literal[False] = False
    assumptions_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside representable finite range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Evaluate exact weighted squared errors, then emit finite rounded diagnostics."""
    knot_times = [knot.time for knot in request.censoring_curve]
    knot_values = [Fraction(knot.survival) for knot in request.censoring_curve]

    def curve(time: float, *, left: bool) -> Fraction:
        index = (bisect_left if left else bisect_right)(knot_times, time) - 1
        return knot_values[index] if index >= 0 else Fraction(1)

    event_weights: list[Fraction | None] = []
    for outcome in request.outcomes:
        weight = None
        if outcome.event_observed and outcome.duration <= request.horizons[-1]:
            probability = curve(outcome.duration, left=True)
            if probability == 0:
                raise ValueError("an observed event has zero censoring survival at its left limit")
            weight = 1 / probability
        event_weights.append(weight)
    exact_scores: list[Fraction] = []
    scores: list[HorizonScore] = []
    size = len(request.outcomes)
    for column, horizon in enumerate(request.horizons):
        g_horizon = curve(horizon, left=False)
        events = survivors = censored = 0
        event_loss = survivor_loss = weight_sum = Fraction()
        maximum_weight: Fraction | None = None
        for index, (outcome, prediction) in enumerate(
            zip(request.outcomes, request.survival_probabilities, strict=True)
        ):
            probability = Fraction(prediction[column])
            if outcome.duration <= horizon:
                if not outcome.event_observed:
                    censored += 1
                    continue
                weight = event_weights[index]
                if weight is None:
                    raise ValueError("missing event censoring weight")
                events += 1
                event_loss += weight * probability * probability
            else:
                if g_horizon == 0:
                    raise ValueError("a known survivor has zero censoring survival at the horizon")
                weight = 1 / g_horizon
                survivors += 1
                survivor_loss += weight * (1 - probability) ** 2
            weight_sum += weight
            maximum_weight = weight if maximum_weight is None else max(weight, maximum_weight)
        score = (event_loss + survivor_loss) / size
        exact_scores.append(score)
        scores.append(
            HorizonScore(
                horizon=horizon,
                ipcw_brier_score=_number(score, "IPCW Brier score"),
                event_contribution=_number(event_loss / size, "event contribution"),
                survivor_contribution=_number(survivor_loss / size, "survivor contribution"),
                observed_event_count=events,
                known_survivor_count=survivors,
                censored_by_horizon_count=censored,
                informative_count=events + survivors,
                censoring_survival_at_horizon=float(g_horizon),
                maximum_applied_weight=None
                if maximum_weight is None
                else _number(maximum_weight, "maximum applied weight"),
                applied_weight_sum=_number(weight_sum, "weight sum"),
            )
        )
    integrated = None
    if len(request.horizons) > 1:
        area = sum(
            (
                (Fraction(right) - Fraction(left)) * (a + b) / 2
                for (left, right), (a, b) in zip(
                    pairwise(request.horizons), pairwise(exact_scores), strict=True
                )
            ),
            Fraction(),
        )
        span = Fraction(request.horizons[-1]) - Fraction(request.horizons[0])
        integrated = _number(area / span, "integrated Brier score")
    return Output(
        subject_count=size,
        horizon_count=len(request.horizons),
        scores=scores,
        integrated_brier_score=integrated,
        integration_status="single_horizon"
        if integrated is None
        else "trapezoid_over_supplied_horizons",
        censoring_valid_through=request.censoring_valid_through,
    )


OPERATION = Operation(
    id="skills.score_censored_brier",
    kind="skill",
    description=(
        "Compute right-censored IPCW Brier curves and trapezoidal integration using a supplied "
        "censoring-survival curve, explicit event-left/survivor-right weights and full-sample "
        "denominators. Reports informative counts without validating censoring assumptions."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
