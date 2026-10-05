"""Multivariate energy scores of equally weighted empirical forecast ensembles.

Uses E||X-y|| - E||X-X'||/2 with Euclidean distances, scoring the empirical
distribution itself. There is no finite-ensemble correction or rescaling of
coordinates. Reference: Gneiting and Raftery (2007), doi:10.1198/016214506000001437.
"""

from math import dist, fsum
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Vector = Annotated[list[Value], Field(min_length=1, max_length=32)]
Ensemble = Annotated[list[Vector], Field(min_length=1, max_length=512)]


class Input(InputModel):
    outcomes: list[Vector] = Field(min_length=1, max_length=1000)
    ensembles: list[Ensemble] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def consistent_dimensions_and_budget(self) -> Self:
        if len(self.outcomes) != len(self.ensembles):
            raise ValueError("ensembles must have one row per outcome")
        dimensions = len(self.outcomes[0])
        work = 0
        for outcome, ensemble in zip(self.outcomes, self.ensembles, strict=True):
            if len(outcome) != dimensions or any(len(member) != dimensions for member in ensemble):
                raise ValueError("all outcomes and ensemble members must have equal dimensions")
            count = len(ensemble)
            work += dimensions * (count + count * (count - 1) // 2)
        if work > 2_000_000:
            raise ValueError("energy score exceeds 2000000 scalar distance-work units")
        return self


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    mean_energy_score: float
    energy_scores: list[float]
    mean_observation_distance: float
    mean_half_pairwise_distance: float
    minimum_ensemble_size: int
    maximum_ensemble_size: int
    coordinate_scaling_applied: bool = False
    finite_ensemble_correction_applied: bool = False


def execute(request: Input, context: OperationContext) -> Output:
    """Use unordered pairs once; diagonal distances are exactly zero."""
    scores: list[float] = []
    first_terms: list[float] = []
    second_terms: list[float] = []
    sizes: list[int] = []
    for outcome, ensemble in zip(request.outcomes, request.ensembles, strict=True):
        count = len(ensemble)
        first = fsum(dist(member, outcome) for member in ensemble) / count
        second = fsum(
            dist(ensemble[left], ensemble[right])
            for left in range(count)
            for right in range(left + 1, count)
        ) / (count * count)
        # Nonnegative by the triangle inequality; clamp only roundoff below zero.
        scores.append(max(0.0, fsum((first, -second))))
        first_terms.append(first)
        second_terms.append(second)
        sizes.append(count)
    return Output(
        observation_count=len(scores),
        dimension_count=len(request.outcomes[0]),
        mean_energy_score=fsum(scores) / len(scores),
        energy_scores=scores,
        mean_observation_distance=fsum(first_terms) / len(scores),
        mean_half_pairwise_distance=fsum(second_terms) / len(scores),
        minimum_ensemble_size=min(sizes),
        maximum_ensemble_size=max(sizes),
    )


OPERATION = Operation(
    id="skills.score_energy",
    kind="skill",
    description=(
        "Compute multivariate energy scores for equally weighted empirical ensembles using "
        "Euclidean distances and exact pair terms. Lower is better; coordinates are not rescaled."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
