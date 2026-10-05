"""Two-sided tabular CUSUM detection with caller-supplied scale and thresholds.

Starting at zero, update U=max(0,U+x-target-drift) and
L=max(0,L+target-x-drift). A direction emits an event when its updated
accumulator is >= threshold. Both directions are evaluated before reset;
the event records their pre-reset values. Reset either both accumulators on
any event, or only each direction that triggered. There is no head start,
implicit standardization, threshold fitting, or false-alarm probability.

Rows retain their supplied order. The caller enforces source availability
and chooses a target, drift and threshold in the same units as observations.
Reference: NIST Dataplot, positive/negative tabular CUSUM recurrences:
https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/cusum.htm
"""

from math import fsum
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Drift = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
Threshold = Annotated[float, Field(strict=True, gt=0, le=1e100, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
ResetPolicy = Literal["both", "triggered"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    target: Value
    drift: Drift
    threshold: Threshold
    reset_policy: ResetPolicy = "both"
    max_events: int = Field(default=100, strict=True, ge=0, le=500)


class Event(OutputModel):
    row_index: int = Field(ge=0)
    direction: Literal["upward", "downward"]
    observed_value: Value
    upward_before_reset: Nonnegative
    downward_before_reset: Nonnegative


class Output(OutputModel):
    observation_count: int
    target: Value
    drift: Drift
    threshold: Threshold
    reset_policy: ResetPolicy
    event_count: int
    upward_event_count: int
    downward_event_count: int
    events: list[Event] = Field(max_length=500)
    omitted_event_count: int
    final_upward_accumulator: Nonnegative
    final_downward_accumulator: Nonnegative
    maximum_upward_before_reset: Nonnegative
    maximum_downward_before_reset: Nonnegative


def execute(request: Input, context: OperationContext) -> Output:
    """Process every row; output truncation never changes state or event counts."""
    upward = downward = 0.0
    maximum_upward = maximum_downward = 0.0
    upward_count = downward_count = 0
    events: list[Event] = []
    for index, value in enumerate(request.values):
        # Separate terms preserve small drift and residuals near a large target;
        # forming target+drift first would discard sub-ULP drift information.
        upward = max(0.0, fsum((upward, value, -request.target, -request.drift)))
        downward = max(0.0, fsum((downward, request.target, -value, -request.drift)))
        maximum_upward = max(maximum_upward, upward)
        maximum_downward = max(maximum_downward, downward)
        upward_hit = upward >= request.threshold
        downward_hit = downward >= request.threshold
        upward_count += upward_hit
        downward_count += downward_hit
        if upward_hit and len(events) < request.max_events:
            events.append(
                Event(
                    row_index=index,
                    direction="upward",
                    observed_value=value,
                    upward_before_reset=upward,
                    downward_before_reset=downward,
                )
            )
        if downward_hit and len(events) < request.max_events:
            events.append(
                Event(
                    row_index=index,
                    direction="downward",
                    observed_value=value,
                    upward_before_reset=upward,
                    downward_before_reset=downward,
                )
            )
        if upward_hit or downward_hit:
            if request.reset_policy == "both":
                upward = downward = 0.0
            else:
                if upward_hit:
                    upward = 0.0
                if downward_hit:
                    downward = 0.0
    count = upward_count + downward_count
    return Output(
        observation_count=len(request.values),
        target=request.target,
        drift=request.drift,
        threshold=request.threshold,
        reset_policy=request.reset_policy,
        event_count=count,
        upward_event_count=upward_count,
        downward_event_count=downward_count,
        events=events,
        omitted_event_count=count - len(events),
        final_upward_accumulator=upward,
        final_downward_accumulator=downward,
        maximum_upward_before_reset=maximum_upward,
        maximum_downward_before_reset=maximum_downward,
    )


OPERATION = Operation(
    id="features.cusum_events",
    kind="feature",
    description=(
        "Extract two-sided tabular CUSUM threshold events using supplied target, nonnegative "
        "drift and positive threshold, with explicit both/triggered reset and inclusive "
        "crossing semantics. Reports detections and state, not false-alarm probabilities."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
