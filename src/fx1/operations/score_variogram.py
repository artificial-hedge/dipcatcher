"""Variogram scores with explicit fixed nonnegative weights on unordered pairs.

For each i<j compute w_ij (|y_i-y_j|^p - mean_m |x_mi-x_mj|^p)^2.
The score sums each unordered pair once (half the symmetric double sum).
Fixed weights must be chosen without consulting the verifying outcome. This
proper score is not strictly proper and cannot identify all joint laws.
Reference: Scheuerer and Hamill (2015), doi:10.1175/MWR-D-14-00269.1.
"""

from itertools import combinations
from math import fsum
from typing import Annotated, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e50, le=1e50)]
Vector = Annotated[list[Value], Field(min_length=2, max_length=32)]
Ensemble = Annotated[list[Vector], Field(min_length=1, max_length=512)]


class PairWeight(InputModel):
    left: int = Field(strict=True, ge=0, le=31)
    right: int = Field(strict=True, ge=0, le=31)
    weight: float = Field(strict=True, ge=0, le=1e6)


class Input(InputModel):
    outcomes: list[Vector] = Field(min_length=1, max_length=1000)
    ensembles: list[Ensemble] = Field(min_length=1, max_length=1000)
    power: float = Field(default=0.5, strict=True, ge=0.1, le=2.0)
    pair_weights: list[PairWeight] | None = Field(default=None, min_length=1, max_length=496)

    @model_validator(mode="after")
    def consistent_shapes_and_weights(self) -> Self:
        if len(self.outcomes) != len(self.ensembles):
            raise ValueError("ensembles must have one row per outcome")
        dimensions = len(self.outcomes[0])
        for outcome, ensemble in zip(self.outcomes, self.ensembles, strict=True):
            if len(outcome) != dimensions or any(len(member) != dimensions for member in ensemble):
                raise ValueError("all outcomes and ensemble members must have equal dimensions")
        if self.pair_weights is not None:
            seen: set[tuple[int, int]] = set()
            for pair in self.pair_weights:
                if not pair.left < pair.right < dimensions:
                    raise ValueError("pair indices must satisfy 0 <= left < right < dimensions")
                key = pair.left, pair.right
                if key in seen:
                    raise ValueError("pair weights cannot repeat a coordinate pair")
                seen.add(key)
            if not any(pair.weight > 0 for pair in self.pair_weights):
                raise ValueError("at least one pair weight must be positive")
        pair_count = (
            len(self.pair_weights) if self.pair_weights else dimensions * (dimensions - 1) // 2
        )
        if pair_count * sum(len(ensemble) + 1 for ensemble in self.ensembles) > 2_000_000:
            raise ValueError("variogram score exceeds 2000000 pair-evaluation units")
        return self


class PairScore(OutputModel):
    left: int
    right: int
    weight: float
    mean_weighted_squared_error: float


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    power: float
    mean_variogram_score: float
    variogram_scores: list[float]
    pair_count: int
    total_pair_weight: float
    pair_scores: list[PairScore]
    unordered_pairs_counted_once: bool = True


def execute(request: Input, context: OperationContext) -> Output:
    pairs = (
        request.pair_weights
        if request.pair_weights is not None
        else [
            PairWeight(left=left, right=right, weight=1.0)
            for left, right in combinations(range(len(request.outcomes[0])), 2)
        ]
    )
    row_terms: list[list[float]] = [[] for _ in request.outcomes]
    pair_results: list[PairScore] = []
    for pair in pairs:
        terms: list[float] = []
        for index, (outcome, ensemble) in enumerate(
            zip(request.outcomes, request.ensembles, strict=True)
        ):
            observed = abs(outcome[pair.left] - outcome[pair.right]) ** request.power
            predicted = fsum(
                abs(member[pair.left] - member[pair.right]) ** request.power for member in ensemble
            ) / len(ensemble)
            term = pair.weight * (observed - predicted) ** 2
            terms.append(term)
            row_terms[index].append(term)
        pair_results.append(
            PairScore(
                left=pair.left,
                right=pair.right,
                weight=pair.weight,
                mean_weighted_squared_error=fsum(terms) / len(terms),
            )
        )
    scores = [fsum(terms) for terms in row_terms]
    return Output(
        observation_count=len(scores),
        dimension_count=len(request.outcomes[0]),
        power=request.power,
        mean_variogram_score=fsum(scores) / len(scores),
        variogram_scores=scores,
        pair_count=len(pairs),
        total_pair_weight=fsum(pair.weight for pair in pairs),
        pair_scores=pair_results,
    )


OPERATION = Operation(
    id="skills.score_variogram",
    kind="skill",
    description=(
        "Compute multivariate empirical-ensemble variogram scores with explicit power, "
        "fixed nonnegative pair weights and per-pair decomposition; count unordered pairs once."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
