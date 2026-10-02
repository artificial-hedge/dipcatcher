"""Proper central prediction interval score, in lower-is-better orientation.

Implements Gneiting and Raftery (2007), equation 43,
doi:10.1198/016214506000001437. Coverage is descriptive, not a proper score.
"""

from math import fsum
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100)]
Coverage = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0.000001, le=0.999999)]


class Input(InputModel):
    outcomes: list[Value] = Field(min_length=1, max_length=100_000)
    lower: list[Value] = Field(min_length=1, max_length=100_000)
    upper: list[Value] = Field(min_length=1, max_length=100_000)
    nominal_coverage: Coverage

    @model_validator(mode="after")
    def validate_alignment(self) -> Self:
        if not len(self.outcomes) == len(self.lower) == len(self.upper):
            raise ValueError("outcomes, lower, and upper must have equal lengths")
        if any(lower > upper for lower, upper in zip(self.lower, self.upper, strict=True)):
            raise ValueError("every lower bound must be less than or equal to its upper bound")
        return self


class Output(OutputModel):
    observation_count: int
    nominal_coverage: float
    interval_score: float
    mean_width: float
    mean_lower_miss_penalty: float
    mean_upper_miss_penalty: float
    empirical_coverage: float


def execute(request: Input, context: OperationContext) -> Output:
    """Score central intervals targeting the alpha/2 and 1-alpha/2 quantiles."""
    multiplier = 2.0 / (1.0 - request.nominal_coverage)
    widths: list[float] = []
    lower_penalties: list[float] = []
    upper_penalties: list[float] = []
    covered = 0
    for outcome, lower, upper in zip(request.outcomes, request.lower, request.upper, strict=True):
        widths.append(upper - lower)
        lower_penalties.append(multiplier * max(lower - outcome, 0.0))
        upper_penalties.append(multiplier * max(outcome - upper, 0.0))
        covered += int(lower <= outcome <= upper)
    count = len(request.outcomes)
    width = fsum(widths) / count
    lower_penalty = fsum(lower_penalties) / count
    upper_penalty = fsum(upper_penalties) / count
    return Output(
        observation_count=count,
        nominal_coverage=request.nominal_coverage,
        interval_score=fsum((width, lower_penalty, upper_penalty)),
        mean_width=width,
        mean_lower_miss_penalty=lower_penalty,
        mean_upper_miss_penalty=upper_penalty,
        empirical_coverage=covered / count,
    )


OPERATION = Operation(
    id="skills.score_intervals",
    kind="skill",
    description=(
        "Compute proper central interval score with width and miss-penalty decomposition. "
        "Reports empirical coverage as a diagnostic, with inclusive interval endpoints."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
