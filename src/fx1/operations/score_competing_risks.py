"""IPCW multicategory Brier curves for mutually exclusive terminal event causes.

Predictions are cumulative cause probabilities F_k(t). The event-free state
has probability 1-sum_k(F_k(t)). At each horizon, sum squared errors over all
causes AND the event-free state. An observed cause by t is a one-hot event;
a subject followed beyond t is one-hot event-free. A subject censored by t
contributes zero. Divide by the full cohort size, not the informative size.
With one cause, this convention is twice the usual survival Brier score.

Use 1/G(T-) for observed terminal events and 1/G(t) for survivors, under the
explicit convention T<=C means an observed event. G is supplied, right-continuous
and marginal; it is not fitted here. Independent/noninformative censoring,
appropriate held-out forecasts, positive relevant G and sampling assumptions
are needed for the usual proper-score interpretation and are not established.
No left truncation, recurrent events or arbitrary multi-state transitions.
Cause-specific IPCW reference: Cho et al.,
https://arxiv.org/abs/2106.12948
This implementation sums the cause-specific losses plus the event-free loss.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Time = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1)]
Cause = Annotated[str, Field(min_length=1, max_length=64)]
CauseVector = Annotated[list[Probability], Field(min_length=1, max_length=16)]
ForecastCurve = Annotated[list[CauseVector], Field(min_length=1, max_length=100)]


class Outcome(InputModel):
    duration: Time
    cause: Cause | None


class CensoringKnot(InputModel):
    time: Time
    survival: Probability


class Input(InputModel):
    causes: list[Cause] = Field(min_length=1, max_length=16)
    outcomes: list[Outcome] = Field(min_length=1, max_length=1000)
    horizons: list[Time] = Field(min_length=1, max_length=100)
    cumulative_incidence: list[ForecastCurve] = Field(min_length=1, max_length=1000)
    censoring_curve: list[CensoringKnot] = Field(max_length=2000)
    censoring_valid_through: Time

    @model_validator(mode="after")
    def coherent_incidence(self) -> Self:
        if len(set(self.causes)) != len(self.causes):
            raise ValueError("cause labels must be unique")
        if any(
            outcome.cause is not None and outcome.cause not in self.causes
            for outcome in self.outcomes
        ):
            raise ValueError("every observed cause must have a declared forecast coordinate")
        if any(right <= left for left, right in pairwise(self.horizons)):
            raise ValueError("horizons must be strictly increasing")
        if len(self.outcomes) != len(self.cumulative_incidence):
            raise ValueError("cumulative incidence curves must align with outcomes")
        if len(self.outcomes) * len(self.horizons) * (len(self.causes) + 1) > 100_000:
            raise ValueError("competing-risk scoring exceeds 100000 state cells")
        for curve in self.cumulative_incidence:
            if len(curve) != len(self.horizons) or any(
                len(row) != len(self.causes) for row in curve
            ):
                raise ValueError("forecast axes must be subject, horizon, then declared cause")
            if any(sum((Fraction(value) for value in row), Fraction()) > 1 for row in curve):
                raise ValueError(
                    "cause probabilities must sum to at most one without renormalization"
                )
            if any(
                any(b < a for a, b in zip(left, right, strict=True))
                for left, right in pairwise(curve)
            ):
                raise ValueError("each cause's cumulative incidence must be nondecreasing")
        if any(
            right.time <= left.time or right.survival > left.survival
            for left, right in pairwise(self.censoring_curve)
        ):
            raise ValueError("censoring knots require increasing times and nonincreasing survival")
        if self.horizons[-1] > self.censoring_valid_through or (
            self.censoring_curve and self.censoring_curve[-1].time > self.censoring_valid_through
        ):
            raise ValueError("horizons and curve knots must be within declared censoring support")
        return self


class HorizonScore(OutputModel):
    horizon: float
    multicategory_brier_score: float
    event_free_component: float
    cause_components: list[float]
    observed_cause_counts: list[int]
    known_event_free_count: int
    censored_by_horizon_count: int
    informative_count: int
    applied_weight_sum: float
    maximum_applied_weight: float | None
    censoring_survival: float


class Output(OutputModel):
    causes: list[str]
    subject_count: int
    scores: list[HorizonScore]
    integrated_multicategory_brier_score: float | None
    integration_status: Literal["single_horizon", "trapezoid_over_supplied_horizons"]
    score_convention: Literal["sum_cause_and_event_free_squared_errors"] = (
        "sum_cause_and_event_free_squared_errors"
    )
    score_denominator: Literal["all_subjects"] = "all_subjects"
    event_weights: Literal["inverse_censoring_survival_left_limit"] = (
        "inverse_censoring_survival_left_limit"
    )
    survivor_weights: Literal["inverse_censoring_survival_at_horizon"] = (
        "inverse_censoring_survival_at_horizon"
    )
    censoring_valid_through: float
    censoring_model_fitted: Literal[False] = False
    assumptions_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside the representable finite range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    knot_times = [knot.time for knot in request.censoring_curve]
    knot_values = [Fraction(knot.survival) for knot in request.censoring_curve]
    cause_indices = {cause: index for index, cause in enumerate(request.causes)}
    count = len(request.outcomes)

    def censoring(time: float, *, left: bool) -> Fraction:
        index = (bisect_left if left else bisect_right)(knot_times, time) - 1
        return knot_values[index] if index >= 0 else Fraction(1)

    event_weights: list[Fraction | None] = []
    for outcome in request.outcomes:
        weight = None
        if outcome.cause is not None and outcome.duration <= request.horizons[-1]:
            probability = censoring(outcome.duration, left=True)
            if probability == 0:
                raise ValueError("an observed event has zero left-limit censoring survival")
            weight = 1 / probability
        event_weights.append(weight)
    scores: list[HorizonScore] = []
    exact_scores: list[Fraction] = []
    for horizon_index, horizon in enumerate(request.horizons):
        g_horizon = censoring(horizon, left=False)
        components = [Fraction() for _ in range(len(request.causes) + 1)]
        cause_counts = [0] * len(request.causes)
        known_free = censored = 0
        weight_sum = Fraction()
        maximum_weight: Fraction | None = None
        for subject_index, (outcome, curve) in enumerate(
            zip(request.outcomes, request.cumulative_incidence, strict=True)
        ):
            if outcome.duration <= horizon:
                if outcome.cause is None:
                    censored += 1
                    continue
                target = cause_indices[outcome.cause] + 1
                cause_counts[target - 1] += 1
                weight = event_weights[subject_index]
                if weight is None:
                    raise ValueError("missing observed-event censoring weight")
            else:
                if g_horizon == 0:
                    raise ValueError("a known event-free subject has zero censoring survival")
                weight = 1 / g_horizon
                target = 0
                known_free += 1
            causes = [Fraction(value) for value in curve[horizon_index]]
            probabilities = [1 - sum(causes, Fraction()), *causes]
            for state, probability in enumerate(probabilities):
                components[state] += weight * (probability - int(state == target)) ** 2
            weight_sum += weight
            maximum_weight = weight if maximum_weight is None else max(maximum_weight, weight)
        score = sum(components, Fraction()) / count
        exact_scores.append(score)
        scores.append(
            HorizonScore(
                horizon=horizon,
                multicategory_brier_score=_number(score, "multicategory Brier score"),
                event_free_component=_number(components[0] / count, "event-free component"),
                cause_components=[
                    _number(value / count, "cause component") for value in components[1:]
                ],
                observed_cause_counts=cause_counts,
                known_event_free_count=known_free,
                censored_by_horizon_count=censored,
                informative_count=known_free + sum(cause_counts),
                applied_weight_sum=_number(weight_sum, "applied weight sum"),
                maximum_applied_weight=None
                if maximum_weight is None
                else _number(maximum_weight, "maximum weight"),
                censoring_survival=float(g_horizon),
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
        integrated = _number(
            area / (Fraction(request.horizons[-1]) - Fraction(request.horizons[0])),
            "integrated score",
        )
    return Output(
        causes=request.causes,
        subject_count=count,
        scores=scores,
        integrated_multicategory_brier_score=integrated,
        integration_status="single_horizon"
        if integrated is None
        else "trapezoid_over_supplied_horizons",
        censoring_valid_through=request.censoring_valid_through,
    )


OPERATION = Operation(
    id="skills.score_competing_risks",
    kind="skill",
    description=(
        "Compute IPCW Brier curves for mutually exclusive terminal causes and the event-free "
        "state, retaining cause contributions, informative counts, exact CIF mass checks and "
        "optional horizon integration. Uses supplied censoring curves and explicit event ties."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
