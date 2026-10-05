"""Conditional survival likelihood for supplied piecewise-constant hazards.

Hazard h_j applies on [start_j,start_(j+1)), with start_0=0 and an infinite last
interval. The last hazard must be positive, making survival vanish at infinity.
H(t)=integral_0^t h(s)ds is accumulated exactly from binary inputs. At a knot,
exact-event densities use the hazard to its right. Entry e conditions on T>=e.
Losses are H(t)-H(e)-log h(t) for exact events, H(t)-H(e) for right censoring,
and H(l)-H(e)-log(1-exp(-(H(u)-H(l)))) for interval observations l<T<=u.

Zero hazard at an exact event or zero interval hazard yields infinite loss.
The observation-process likelihood is excluded: interpreting these as proper
predictive likelihood scores requires noninformative censoring/truncation and
outcome-independent weighting. Exact-event densities depend on the declared
time unit and may have negative losses. No fit, timing or assumption checks are
performed. See the censored-likelihood identity in SciPy's fit documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html

Positive times/hazards lie in [1e-100,1e100], with zero also allowed except for
the last hazard. Log/exp are approximate; expm1/log1p stabilize interval terms.
Underflow of a negligible interval tail term is disclosed; if it would make
the entire positive loss vanish, execution fails instead of reporting zero.
There are at most 1000 rows, 256 intervals and 100000 row-interval cells.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from math import exp, expm1, isfinite, log, log1p
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


def _positive_floor(value: float) -> float:
    if 0 < value < 1e-100:
        raise ValueError("positive times and hazard rates must be at least 1e-100")
    return value


Scalar = Annotated[float, Field(strict=True, ge=0, le=1e100), AfterValidator(_positive_floor)]
RateRow = Annotated[list[Scalar], Field(min_length=1, max_length=256)]
Weight = Annotated[float, Field(strict=True, ge=1e-100, le=1e100)]


class ExactEvent(InputModel):
    kind: Literal["event"]
    time: Scalar
    entry_time: Scalar = 0.0


class RightCensored(InputModel):
    kind: Literal["right_censored"]
    time: Scalar
    entry_time: Scalar = 0.0


class IntervalCensored(InputModel):
    kind: Literal["interval_censored"]
    lower: Scalar
    upper: Scalar
    entry_time: Scalar = 0.0


Observation = Annotated[ExactEvent | RightCensored | IntervalCensored, Field(discriminator="kind")]


class Input(InputModel):
    interval_starts: list[Scalar] = Field(min_length=1, max_length=256)
    observations: list[Observation] = Field(min_length=1, max_length=1000)
    hazards: list[RateRow] = Field(min_length=1, max_length=1000)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=1000)
    time_unit: str = Field(strict=True, min_length=1, max_length=64)

    @model_validator(mode="after")
    def supported_forecast_grid(self) -> Self:
        if self.interval_starts[0] != 0 or any(
            right <= left
            for left, right in zip(self.interval_starts, self.interval_starts[1:], strict=False)
        ):
            raise ValueError("hazard intervals must start at zero and increase strictly")
        if len(self.hazards) != len(self.observations) or any(
            len(row) != len(self.interval_starts) or row[-1] <= 0 for row in self.hazards
        ):
            raise ValueError(
                "each observation needs aligned hazard rates with a positive final tail"
            )
        if len(self.observations) * len(self.interval_starts) > 100_000:
            raise ValueError("hazard scoring exceeds 100000 row-interval cells")
        if self.weights is not None and len(self.weights) != len(self.observations):
            raise ValueError("weights must align with observations")
        for observation in self.observations:
            lower = (
                observation.lower if isinstance(observation, IntervalCensored) else observation.time
            )
            if observation.entry_time > lower:
                raise ValueError(
                    "entry time must not follow the observed event/censoring lower time"
                )
            if isinstance(observation, IntervalCensored) and observation.upper <= observation.lower:
                raise ValueError("interval-censored observations require lower < upper")
        return self


class RowScore(OutputModel):
    row_index: int
    kind: Literal["event", "right_censored", "interval_censored"]
    entry_time: float
    lower_time: float
    upper_time: float | None
    observation_weight: float
    hazard_from_entry_to_lower: float
    interval_hazard_increment: float | None
    exact_event_hazard: float | None
    log_loss: float | None
    status: Literal["finite", "infinite_zero_likelihood"]
    interval_tail_term_underflowed: bool


class Output(OutputModel):
    observation_count: int
    observation_kind_counts: dict[str, int]
    interval_count: int
    time_unit: str
    weighted_mean_log_loss: float | None
    aggregate_status: Literal["finite", "infinite_zero_likelihood"]
    infinite_loss_rows: int
    interval_tail_underflow_rows: int
    total_observation_weight: float
    rows: list[RowScore]
    exact_knot_density_hazard: Literal["right_hand_interval"] = "right_hand_interval"
    infinite_tail_has_positive_hazard: Literal[True] = True
    observation_process_likelihood_included: Literal[False] = False
    noninformative_censoring_and_entry_verified: Literal[False] = False
    outcome_independent_weights_verified: Literal[False] = False
    forecast_timing_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite numerical range") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite numerical range")
    return result


def _interval_term(increment: Fraction) -> tuple[float, bool]:
    value = _number(increment, "interval cumulative hazard")
    if value <= log(2):
        return -log(-expm1(-value)), False
    tail = exp(-value)
    return -log1p(-tail), tail == 0


def _cumulative(
    time: float, starts: list[Fraction], prefix: list[Fraction], rates: list[Fraction]
) -> Fraction:
    exact_time = Fraction(time)
    index = bisect_right(starts, exact_time) - 1
    return prefix[index] + rates[index] * (exact_time - starts[index])


def execute(request: Input, context: OperationContext) -> Output:
    starts = [Fraction(value) for value in request.interval_starts]
    weights = (
        [Fraction(value) for value in request.weights]
        if request.weights is not None
        else [Fraction(1)] * len(request.observations)
    )
    rows: list[RowScore] = []
    kinds: dict[str, int] = {}
    infinite = underflowed = 0
    weighted_loss = Fraction()
    for row_index, (observation, supplied_rates, weight) in enumerate(
        zip(request.observations, request.hazards, weights, strict=True)
    ):
        kinds[observation.kind] = kinds.get(observation.kind, 0) + 1
        rates = [Fraction(value) for value in supplied_rates]
        prefix = [Fraction()]
        for index in range(1, len(starts)):
            prefix.append(prefix[-1] + rates[index - 1] * (starts[index] - starts[index - 1]))

        lower = observation.lower if isinstance(observation, IntervalCensored) else observation.time
        lower_hazard = _cumulative(lower, starts, prefix, rates)
        conditional_hazard = lower_hazard - _cumulative(
            observation.entry_time, starts, prefix, rates
        )
        increment: Fraction | None = None
        event_hazard: Fraction | None = None
        tail_underflow = False
        loss: Fraction | None
        if isinstance(observation, ExactEvent):
            event_hazard = rates[bisect_right(request.interval_starts, observation.time) - 1]
            loss = (
                conditional_hazard - Fraction(log(_number(event_hazard, "event hazard")))
                if event_hazard
                else None
            )
            upper: float | None = observation.time
        elif isinstance(observation, RightCensored):
            loss, upper = conditional_hazard, None
        else:
            upper = observation.upper
            increment = _cumulative(upper, starts, prefix, rates) - lower_hazard
            if not increment:
                loss = None
            else:
                term, tail_underflow = _interval_term(increment)
                loss = conditional_hazard + Fraction(term)
                if not loss:
                    raise ValueError("positive interval likelihood loss rounded to zero")
        if loss is None:
            infinite += 1
        else:
            weighted_loss += weight * loss
        underflowed += tail_underflow
        rows.append(
            RowScore(
                row_index=row_index,
                kind=observation.kind,
                entry_time=observation.entry_time,
                lower_time=lower,
                upper_time=upper,
                observation_weight=_number(weight, "weight"),
                hazard_from_entry_to_lower=_number(conditional_hazard, "conditional hazard"),
                interval_hazard_increment=None
                if increment is None
                else _number(increment, "interval hazard"),
                exact_event_hazard=None
                if event_hazard is None
                else _number(event_hazard, "event hazard"),
                log_loss=None if loss is None else _number(loss, "row log loss"),
                status="infinite_zero_likelihood" if loss is None else "finite",
                interval_tail_term_underflowed=tail_underflow,
            )
        )
    total_weight = sum(weights, Fraction())
    return Output(
        observation_count=len(rows),
        observation_kind_counts=dict(sorted(kinds.items())),
        interval_count=len(starts),
        time_unit=request.time_unit,
        weighted_mean_log_loss=None
        if infinite
        else _number(weighted_loss / total_weight, "mean log loss"),
        aggregate_status="infinite_zero_likelihood" if infinite else "finite",
        infinite_loss_rows=infinite,
        interval_tail_underflow_rows=underflowed,
        total_observation_weight=_number(total_weight, "total weight"),
        rows=rows,
    )


OPERATION = Operation(
    id="skills.score_piecewise_hazards",
    kind="skill",
    description="Score supplied piecewise hazard forecasts through conditional exact-event, right-censored and interval-censored likelihoods, with entry conditioning, explicit time units and zero-support outcomes.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
