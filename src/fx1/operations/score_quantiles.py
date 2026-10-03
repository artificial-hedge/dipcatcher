"""Pinball losses for ordered quantile forecasts, with crossing rejected.

Uses the negatively oriented quantile score (lower is better); see Gneiting
and Raftery (2007), section 6.1, doi:10.1198/016214506000001437.
"""

from itertools import pairwise
from math import fsum
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100)]
Level = Annotated[float, Field(strict=True, allow_inf_nan=False, gt=0, lt=1)]
QuantileRow = Annotated[list[Value], Field(min_length=1, max_length=128)]


class Input(InputModel):
    outcomes: list[Value] = Field(min_length=1, max_length=10_000)
    levels: list[Level] = Field(min_length=1, max_length=128)
    quantiles: list[QuantileRow] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if len(self.quantiles) != len(self.outcomes):
            raise ValueError("quantiles must have one row per outcome")
        if len(self.levels) * len(self.outcomes) > 250_000:
            raise ValueError("at most 250000 quantile values are supported")
        if any(left >= right for left, right in pairwise(self.levels)):
            raise ValueError("levels must be strictly increasing")
        for row in self.quantiles:
            if len(row) != len(self.levels):
                raise ValueError("each quantile row must match the number of levels")
            if any(left > right for left, right in pairwise(row)):
                raise ValueError("quantile forecasts must be nondecreasing; crossings are invalid")
        return self


class Output(OutputModel):
    observation_count: int
    levels: list[float]
    mean_pinball_by_level: list[float]
    mean_pinball: float


def execute(request: Input, context: OperationContext) -> Output:
    """Return unweighted means across observations and, separately, levels."""
    scores: list[float] = []
    for column, level in enumerate(request.levels):
        losses: list[float] = []
        for outcome, row in zip(request.outcomes, request.quantiles, strict=True):
            residual = outcome - row[column]
            losses.append(level * residual if residual >= 0 else (level - 1.0) * residual)
        scores.append(fsum(losses) / len(request.outcomes))
    return Output(
        observation_count=len(request.outcomes),
        levels=request.levels,
        mean_pinball_by_level=scores,
        mean_pinball=fsum(scores) / len(scores),
    )


OPERATION = Operation(
    id="skills.score_quantiles",
    kind="skill",
    description=(
        "Compute lower-is-better pinball losses per quantile level and overall. "
        "Rejects crossing quantiles, unordered levels, invalid shapes, and nonfinite values."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
