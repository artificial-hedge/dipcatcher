"""Product-limit event or censoring survival for an unweighted right-censored cohort.

All subjects enter at time zero. Observed events win event/censoring ties.
Event survival updates by (n-d)/n before removing both events and censors.
Reverse censoring survival updates by (n-d-c)/(n-d), after first removing
observed events. A zero denominator with zero censors makes no update.
Products and restricted-mean integration are exact rational calculations on
the supplied binary-float durations, rounded only for returned values.

No left truncation, interval censoring, competing causes, conditional model,
confidence intervals or censoring-independence assessment is implemented.
Evaluation beyond the largest observed duration is explicitly unavailable.
Reference: Kaplan and Meier (1958); NIST product-limit description:
https://www.itl.nist.gov/div898/handbook/apr/section2/apr215.htm
Reverse tie convention:
https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.nonparametric.kaplan_meier_estimator.html
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Time = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Estimand = Literal["event_survival", "censoring_survival"]


class Subject(InputModel):
    duration: Time
    event_observed: bool = Field(strict=True)


class Input(InputModel):
    subjects: list[Subject] = Field(min_length=1, max_length=5000)
    estimand: Estimand = "event_survival"
    evaluation_times: list[Time] = Field(default_factory=list, max_length=1000)
    restricted_mean_through: Time | None = None
    offset: int = Field(default=0, strict=True, ge=0, le=5000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)

    @model_validator(mode="after")
    def observed_integration_range(self) -> Self:
        if self.restricted_mean_through is not None and self.restricted_mean_through > max(
            subject.duration for subject in self.subjects
        ):
            raise ValueError("restricted mean horizon cannot exceed the last observed duration")
        return self


class CurveRow(OutputModel):
    time: float
    at_risk_before: int
    observed_events: int
    observed_censors: int
    update_denominator: int
    update_failures: int
    survival_left: float
    survival: float
    removed_count: int
    at_risk_after: int


class Evaluation(OutputModel):
    time: float
    survival: float | None
    status: Literal["within_observed_followup", "beyond_observed_followup"]


class Output(OutputModel):
    estimand: Estimand
    subject_count: int
    observed_event_count: int
    observed_censor_count: int
    unique_time_count: int
    maximum_observed_duration: float
    final_survival: float
    median_time: float | None
    median_status: Literal["reached", "not_reached"]
    curve: list[CurveRow]
    offset: int
    has_more: bool
    evaluations: list[Evaluation]
    restricted_mean_through: float | None
    restricted_mean: float | None
    tie_convention: Literal["observed_events_before_censoring"] = "observed_events_before_censoring"
    curve_continuity: Literal["right_continuous"] = "right_continuous"
    all_subjects_enter_at_zero: Literal[True] = True
    extrapolation_performed: Literal[False] = False
    assumptions_verified: Literal[False] = False


def _number(value: Fraction) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("product-limit result exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError("nonzero product-limit result is not representable as a finite float")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    counts: dict[float, list[int]] = {}
    for subject in request.subjects:
        group = counts.setdefault(subject.duration, [0, 0])
        group[0 if subject.event_observed else 1] += 1
    times = sorted(counts)
    remaining = len(request.subjects)
    survival = Fraction(1)
    median: float | None = None
    values: list[Fraction] = []
    rows: list[CurveRow] = []
    for index, time in enumerate(times):
        events, censors = counts[time]
        before = survival
        denominator = remaining if request.estimand == "event_survival" else remaining - events
        failures = events if request.estimand == "event_survival" else censors
        if denominator:
            survival *= Fraction(denominator - failures, denominator)
        elif failures:
            raise ValueError("a product-limit update has failures outside its risk set")
        values.append(survival)
        if median is None and survival <= Fraction(1, 2):
            median = time
        # Validate every estimate even if its row is outside the requested page.
        left_value, right_value = _number(before), _number(survival)
        if request.offset <= index < request.offset + request.limit:
            rows.append(
                CurveRow(
                    time=time,
                    at_risk_before=remaining,
                    observed_events=events,
                    observed_censors=censors,
                    update_denominator=denominator,
                    update_failures=failures,
                    survival_left=left_value,
                    survival=right_value,
                    removed_count=events + censors,
                    at_risk_after=remaining - events - censors,
                )
            )
        remaining -= events + censors
    evaluations: list[Evaluation] = []
    for time in request.evaluation_times:
        if time > times[-1]:
            evaluations.append(
                Evaluation(time=time, survival=None, status="beyond_observed_followup")
            )
            continue
        index = bisect_right(times, time) - 1
        evaluations.append(
            Evaluation(
                time=time,
                survival=_number(values[index] if index >= 0 else Fraction(1)),
                status="within_observed_followup",
            )
        )
    restricted_mean = None
    if request.restricted_mean_through is not None:
        horizon = Fraction(request.restricted_mean_through)
        previous_time = Fraction()
        previous_survival = Fraction(1)
        area = Fraction()
        for time, value in zip(times, values, strict=True):
            stop = min(Fraction(time), horizon)
            area += (stop - previous_time) * previous_survival
            if stop == horizon:
                break
            previous_time, previous_survival = stop, value
        restricted_mean = _number(area)
    events_total = sum(subject.event_observed for subject in request.subjects)
    return Output(
        estimand=request.estimand,
        subject_count=len(request.subjects),
        observed_event_count=events_total,
        observed_censor_count=len(request.subjects) - events_total,
        unique_time_count=len(times),
        maximum_observed_duration=times[-1],
        final_survival=_number(survival),
        median_time=median,
        median_status="reached" if median is not None else "not_reached",
        curve=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < len(times),
        evaluations=evaluations,
        restricted_mean_through=request.restricted_mean_through,
        restricted_mean=restricted_mean,
    )


OPERATION = Operation(
    id="features.estimate_product_limit",
    kind="feature",
    description=(
        "Estimate event or reverse-censoring product-limit survival from a right-censored "
        "cohort, with exact tie-aware risk-set updates, bounded curve pages, supported-time "
        "evaluation and optional restricted-mean integration. Assumptions remain unverified."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
