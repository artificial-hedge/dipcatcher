"""Duration-weighted inverse label concurrency, independently per security.

For label i with interval [s_i,e_i), define c_g(t) as the number of supplied
intervals active for its security. Its weight is
u_i = integral_[s_i,e_i) 1/c_g(t) dt / (e_i-s_i).
This is a duration-weighted definition, not a discrete bar average. Endpoint
touching intervals do not overlap. Duplicate intervals count as separate labels.
Weights remain raw u_i in (0,1]; they are not normalized across labels or groups.
Neither these weights nor their sum establish sample independence.

Aware clocks are normalized to UTC; durations are exact integer microseconds.
An endpoint sweep accumulates an exact rational integral, then endpoint prefix
differences recover each label's contribution. Outputs preserve original source
indices and expose exact numerator/denominator weights alongside rounded floats.
Each security's total allocated inverse-concurrency duration equals its covered
union duration. This is descriptive overlap accounting, not a causal feature:
the caller must select intervals whose metadata is available for its purpose.
"""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Security = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


def _clock(value: object) -> datetime:
    if isinstance(value, str):
        if len(value) > 64:
            raise ValueError("interval clock strings are bounded to 64 characters")
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("interval clocks must be explicit timezone-aware ISO datetimes")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("interval clocks require a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("interval clock is outside the supported UTC range") from error


def _microseconds(delta: timedelta) -> int:
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class Label(InputModel):
    security_id: Security
    start: AwareDatetime
    end: AwareDatetime

    @field_validator("start", "end", mode="before")
    @classmethod
    def validate_clock(cls, value: object) -> datetime:
        return _clock(value)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if not self.start < self.end:
            raise ValueError("each half-open interval must have positive duration")
        if self.end - self.start > timedelta(days=366):
            raise ValueError("each label duration is bounded to 366 days")
        return self


class Input(InputModel):
    labels: list[Label] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def bound_groups(self) -> Self:
        if len({label.security_id for label in self.labels}) > 128:
            raise ValueError("at most 128 distinct securities are supported")
        return self


class LabelWeight(OutputModel):
    source_index: int
    security_id: str
    duration_microseconds: int = Field(gt=0)
    inverse_concurrency_integral_microseconds_numerator: int = Field(gt=0)
    inverse_concurrency_integral_microseconds_denominator: int = Field(gt=0)
    uniqueness_numerator: int = Field(gt=0)
    uniqueness_denominator: int = Field(gt=0)
    uniqueness_weight: float = Field(gt=0, le=1, allow_inf_nan=False)


class SecuritySummary(OutputModel):
    security_id: str
    label_count: int
    unique_endpoint_count: int
    covered_duration_microseconds: int
    summed_label_duration_microseconds: int
    peak_concurrency: int


class Output(OutputModel):
    label_count: int
    security_count: int
    interval_convention: Literal["start_inclusive_end_exclusive"] = "start_inclusive_end_exclusive"
    weight_normalization: Literal["raw_duration_averaged_inverse_concurrency"] = (
        "raw_duration_averaged_inverse_concurrency"
    )
    weights: list[LabelWeight] = Field(min_length=1, max_length=512)
    securities: list[SecuritySummary] = Field(min_length=1, max_length=128)


def execute(request: Input, context: OperationContext) -> Output:
    """Sweep simultaneous endpoints once and query exact integral prefixes per label."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(request.labels):
        grouped[label.security_id].append(index)
    weights: list[LabelWeight] = []
    summaries: list[SecuritySummary] = []
    for security, indices in sorted(grouped.items()):
        changes: dict[datetime, int] = defaultdict(int)
        for index in indices:
            label = request.labels[index]
            changes[label.start] += 1
            changes[label.end] -= 1
        clocks = sorted(changes)
        prefixes: dict[datetime, Fraction] = {}
        integral = Fraction(0)
        concurrent = peak = covered = 0
        previous = clocks[0]
        for clock in clocks:
            duration = _microseconds(clock - previous)
            if concurrent:
                integral += Fraction(duration, concurrent)
                covered += duration
            prefixes[clock] = integral
            concurrent += changes[clock]
            peak = max(peak, concurrent)
            previous = clock
        total_duration = 0
        for index in indices:
            label = request.labels[index]
            duration = _microseconds(label.end - label.start)
            exposure = prefixes[label.end] - prefixes[label.start]
            weight = exposure / duration
            total_duration += duration
            weights.append(
                LabelWeight(
                    source_index=index,
                    security_id=security,
                    duration_microseconds=duration,
                    inverse_concurrency_integral_microseconds_numerator=exposure.numerator,
                    inverse_concurrency_integral_microseconds_denominator=exposure.denominator,
                    uniqueness_numerator=weight.numerator,
                    uniqueness_denominator=weight.denominator,
                    uniqueness_weight=float(weight),
                )
            )
        summaries.append(
            SecuritySummary(
                security_id=security,
                label_count=len(indices),
                unique_endpoint_count=len(clocks),
                covered_duration_microseconds=covered,
                summed_label_duration_microseconds=total_duration,
                peak_concurrency=peak,
            )
        )
    weights.sort(key=lambda row: row.source_index)
    return Output(
        label_count=len(request.labels),
        security_count=len(grouped),
        weights=weights,
        securities=summaries,
    )


OPERATION = Operation(
    id="features.label_uniqueness",
    kind="feature",
    description=(
        "Sweep half-open label intervals per security and return exact duration-averaged "
        "inverse-concurrency weights, source indices and coverage diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
