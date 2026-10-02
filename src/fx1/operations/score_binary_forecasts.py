"""Brier and logarithmic loss with explicit endpoint and clipping semantics.

The Brier loss here is (p-y)^2 (the one-event convention). Both scores use
lower-is-better orientation. Natural logarithms produce log loss in nats.
"""

from math import fsum, log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Probability = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=1)]
Binary = Annotated[int, Field(strict=True, ge=0, le=1)]
Clip = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=1e-15, lt=0.5)]


class Input(InputModel):
    outcomes: list[Binary] = Field(min_length=1, max_length=100_000)
    probabilities: list[Probability] = Field(min_length=1, max_length=100_000)
    probability_clip: Clip | None = Field(
        default=None,
        description=(
            "Optional epsilon clips log-loss probabilities into [epsilon, 1-epsilon]. "
            "Brier always uses the original probabilities. Omit for exact log loss."
        ),
    )

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if len(self.outcomes) != len(self.probabilities):
            raise ValueError("outcomes and probabilities must have equal lengths")
        return self


class Output(OutputModel):
    observation_count: int
    brier_score: float
    mean_log_loss: float | None
    log_loss_status: Literal["finite", "positive_infinity"]
    impossible_event_count: int
    clipped_probability_count: int
    probability_clip: float | None


def execute(request: Input, context: OperationContext) -> Output:
    """Represent infinite exact log loss as null plus an explicit status."""
    brier: list[float] = []
    logarithmic: list[float] = []
    impossible = 0
    clipped = 0
    for outcome, probability in zip(request.outcomes, request.probabilities, strict=True):
        brier.append((probability - outcome) ** 2)
        if (outcome == 1 and probability == 0) or (outcome == 0 and probability == 1):
            impossible += 1
        scored_probability = probability
        if request.probability_clip is not None:
            epsilon = request.probability_clip
            scored_probability = min(1.0 - epsilon, max(epsilon, probability))
            clipped += int(scored_probability != probability)
        if (outcome == 1 and scored_probability == 0) or (outcome == 0 and scored_probability == 1):
            continue
        logarithmic.append(-log(scored_probability) if outcome else -log1p(-scored_probability))
    infinite = impossible > 0 and request.probability_clip is None
    return Output(
        observation_count=len(request.outcomes),
        brier_score=fsum(brier) / len(brier),
        mean_log_loss=None if infinite else fsum(logarithmic) / len(request.outcomes),
        log_loss_status="positive_infinity" if infinite else "finite",
        impossible_event_count=impossible,
        clipped_probability_count=clipped,
        probability_clip=request.probability_clip,
    )


OPERATION = Operation(
    id="skills.score_binary_forecasts",
    kind="skill",
    description=(
        "Compute binary Brier score and natural-log loss, preserving impossible endpoint "
        "forecasts as explicit infinite loss unless optional clipping is requested."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
