"""Exact CRPS of equally weighted empirical forecast distributions.

Integrates squared CDF error across sorted sample gaps in O(n log n) time
and O(n) memory. No quadratic pairwise-distance matrix is constructed.
Gneiting and Raftery (2007), eqs. 20-21, doi:10.1198/016214506000001437.
"""

from itertools import pairwise
from math import fsum
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100)]
Sample = Annotated[list[Value], Field(min_length=1, max_length=4096)]


class Input(InputModel):
    outcomes: list[Value] = Field(min_length=1, max_length=4096)
    samples: list[Sample] = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if len(self.outcomes) != len(self.samples):
            raise ValueError("samples must have one nonempty row per outcome")
        if sum(map(len, self.samples)) > 250_000:
            raise ValueError("at most 250000 sample values are supported")
        return self


class Output(OutputModel):
    observation_count: int
    mean_crps: float
    crps_by_observation: list[float]
    minimum_sample_count: int
    maximum_sample_count: int


def execute(request: Input, context: OperationContext) -> Output:
    """Score the empirical distribution itself, without finite-ensemble correction."""
    scores: list[float] = []
    for outcome, sample in zip(request.outcomes, request.samples, strict=True):
        ordered = sorted(sample)
        count = len(ordered)
        areas = [max(ordered[0] - outcome, 0.0), max(outcome - ordered[-1], 0.0)]
        for left_count, (lower, upper) in enumerate(pairwise(ordered), start=1):
            cdf = left_count / count
            # Each portion lies entirely below or above the observation. Summing
            # nonnegative areas avoids cancellation in E|X-y| - E|X-X'|/2.
            below_length = max(min(upper, outcome) - lower, 0.0)
            above_length = max(upper - max(lower, outcome), 0.0)
            areas.append(below_length * cdf**2 + above_length * (1.0 - cdf) ** 2)
        scores.append(fsum(areas))
    sample_counts = list(map(len, request.samples))
    return Output(
        observation_count=len(scores),
        mean_crps=fsum(scores) / len(scores),
        crps_by_observation=scores,
        minimum_sample_count=min(sample_counts),
        maximum_sample_count=max(sample_counts),
    )


OPERATION = Operation(
    id="skills.score_empirical_crps",
    kind="skill",
    description=(
        "Compute exact lower-is-better CRPS for equally weighted empirical forecast samples "
        "using sorted CDF integration. Supports unequal ensemble sizes and tied samples."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
