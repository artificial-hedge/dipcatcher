"""Weighted empirical joint CDF of explicitly ranked marginal observations.

For each coordinate and tied value, let L be the mass below it and E its tied
mass. Pseudo-observations are (L+E)/W under upper ties, or (L+E/2)/W under
midpoint ties. This uses W, not an n+1 boundary correction; zero-weight rows do
not fit marginal or joint distributions but still receive transformed values.
Queries evaluate sum_i w_i*1{all(u_ij<=query_j)}/W with exact rational ranks and
threshold comparisons. Float previews of ranks need not reproduce boundary
comparisons exactly. Margins with ties are not certified to be uniform.

The implementation sorts each marginal once, retaining original source rows,
then evaluates bounded joint queries without a Cartesian grid. It neither fits
a parametric copula nor assumes coordinate or observation independence. Input
ordering/availability and selection remain caller responsibilities. Empirical
CDF-of-pseudo-observations convention (our weight/tie definition is explicit):
https://copulae.readthedocs.io/en/latest/api_reference/copulae/empirical/
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1)]
Point = Annotated[list[Value], Field(min_length=2, max_length=16)]
Query = Annotated[list[Probability], Field(min_length=2, max_length=16)]


class Input(InputModel):
    observations: list[Point] = Field(min_length=2, max_length=2048)
    weights: list[Weight] | None = Field(default=None, min_length=2, max_length=2048)
    queries: list[Query] = Field(default_factory=list, max_length=512)
    tie_policy: Literal["upper", "midpoint"] = "upper"
    max_source_witnesses: int = Field(default=5, strict=True, ge=0, le=20)

    @model_validator(mode="after")
    def rectangular_bounded_input(self) -> Self:
        dimension = len(self.observations[0])
        if any(len(point) != dimension for point in (*self.observations, *self.queries)):
            raise ValueError("all observations and queries must have the same dimension")
        if len(self.observations) * len(self.queries) * dimension > 2_000_000:
            raise ValueError("joint CDF queries exceed two million comparison cells")
        if self.weights is not None:
            if len(self.weights) != len(self.observations) or not any(self.weights):
                raise ValueError("weights must align with observations and have positive total")
            if any(0 < weight < 1e-100 for weight in self.weights):
                raise ValueError("positive empirical weights must be at least 1e-100")
        return self


class Margin(OutputModel):
    coordinate_index: int
    distinct_positive_weight_values: int
    tied_positive_weight_groups: int
    positive_weight_rows_in_ties: int
    minimum_positive_weight_pseudo_observation: float
    maximum_positive_weight_pseudo_observation: float


class QueryResult(OutputModel):
    query_index: int
    thresholds: list[float]
    empirical_joint_cdf: float
    matching_positive_weight_rows: int
    source_row_witnesses: list[int]
    omitted_source_row_witnesses: int


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    positive_weight_count: int
    weight_sum: float
    normalized_weights: list[float]
    tie_policy: str
    pseudo_observations: list[list[float]]
    margins: list[Margin]
    queries: list[QueryResult]
    query_comparison_cell_bound: int
    rank_denominator: Literal["total_positive_weight_without_n_plus_one_correction"] = (
        "total_positive_weight_without_n_plus_one_correction"
    )
    query_comparisons_use_exact_ranks: Literal[True] = True
    zero_weight_rows_affect_fit: Literal[False] = False
    uniform_margins_certified: Literal[False] = False
    independence_assumed: Literal[False] = False
    timing_verified: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    count, dimension = len(request.observations), len(request.observations[0])
    weights = (
        [Fraction(value) for value in request.weights]
        if request.weights is not None
        else [Fraction(1)] * count
    )
    total = sum(weights, Fraction())
    active = [index for index, weight in enumerate(weights) if weight]
    ranks = [[Fraction() for _ in range(dimension)] for _ in range(count)]
    margins: list[Margin] = []
    for column in range(dimension):
        groups: dict[float, list[int]] = defaultdict(list)
        for index, point in enumerate(request.observations):
            groups[point[column]].append(index)
        less = Fraction()
        distinct = tied_groups = tied_rows = 0
        for value in sorted(groups):
            indexes = groups[value]
            mass = sum((weights[index] for index in indexes), Fraction())
            positive_count = sum(bool(weights[index]) for index in indexes)
            if positive_count:
                distinct += 1
                if positive_count > 1:
                    tied_groups += 1
                    tied_rows += positive_count
            rank = (less + (mass if request.tie_policy == "upper" else mass / 2)) / total
            for index in indexes:
                ranks[index][column] = rank
            less += mass
        margins.append(
            Margin(
                coordinate_index=column,
                distinct_positive_weight_values=distinct,
                tied_positive_weight_groups=tied_groups,
                positive_weight_rows_in_ties=tied_rows,
                minimum_positive_weight_pseudo_observation=float(
                    min(ranks[index][column] for index in active)
                ),
                maximum_positive_weight_pseudo_observation=float(
                    max(ranks[index][column] for index in active)
                ),
            )
        )
    queries: list[QueryResult] = []
    for query_index, query in enumerate(request.queries):
        thresholds = [Fraction(value) for value in query]
        matching_mass = Fraction()
        matching_rows = 0
        witnesses: list[int] = []
        for index in active:
            if all(
                rank <= threshold for rank, threshold in zip(ranks[index], thresholds, strict=True)
            ):
                matching_mass += weights[index]
                matching_rows += 1
                if len(witnesses) < request.max_source_witnesses:
                    witnesses.append(index)
        queries.append(
            QueryResult(
                query_index=query_index,
                thresholds=query,
                empirical_joint_cdf=float(matching_mass / total),
                matching_positive_weight_rows=matching_rows,
                source_row_witnesses=witnesses,
                omitted_source_row_witnesses=matching_rows - len(witnesses),
            )
        )
    return Output(
        observation_count=count,
        dimension_count=dimension,
        positive_weight_count=len(active),
        weight_sum=float(total),
        normalized_weights=[float(weight / total) for weight in weights],
        tie_policy=request.tie_policy,
        pseudo_observations=[[float(value) for value in row] for row in ranks],
        margins=margins,
        queries=queries,
        query_comparison_cell_bound=count * len(request.queries) * dimension,
    )


OPERATION = Operation(
    id="features.empirical_copula",
    kind="feature",
    description="Construct exact weighted marginal pseudo-observations under explicit tie rules and evaluate their empirical joint CDF at bounded query vectors, retaining source witnesses without an independence assumption.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
