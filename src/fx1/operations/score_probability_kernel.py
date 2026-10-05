"""Gaussian-kernel proper scores of supplied weighted empirical forecasts.

For k(x,z)=exp(-sum_j((x_j-z_j)/bandwidth_j)^2/2), the loss is
1 + E[k(X,X')] - 2 E[k(X,y)]. The additive one is outcome-only; this is
twice the half-scaled convention in scoringrules' theory documentation:
https://scoringrules.readthedocs.io/en/latest/theory.html
The empirical distribution is scored including diagonal pairs, without an
unbiased finite-ensemble correction. Fixed positive bandwidths must be selected
independently of the verification outcomes for the proper-score interpretation.

Exact rational distances and weights precede floating expm1 evaluations. Using
2 E[1-k(X,y)] - E[1-k(X,X')] avoids subtracting nearly-unit kernels. Rounded
kernel dissimilarities are accumulated exactly. They can still lose positive
definiteness numerically: negative rounded losses are reported, never clamped.
Large dissimilarities that round to one are counted. No significance test,
bandwidth fitting, independence finding or calibration claim is made.
"""

from __future__ import annotations

from fractions import Fraction
from math import expm1, isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Bandwidth = Annotated[float, Field(strict=True, gt=0, le=1e100)]
Vector = Annotated[list[Value], Field(min_length=1, max_length=16)]


class Forecast(InputModel):
    members: list[Vector] = Field(min_length=1, max_length=256)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=256)


class Input(InputModel):
    outcomes: list[Vector] = Field(min_length=1, max_length=128)
    forecasts: list[Forecast] = Field(min_length=1, max_length=128)
    bandwidths: list[Bandwidth] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def dimensions_and_work(self) -> Self:
        if len(self.outcomes) != len(self.forecasts):
            raise ValueError("each outcome requires one empirical forecast")
        dimensions = len(self.bandwidths)
        work = 0
        for outcome, forecast in zip(self.outcomes, self.forecasts, strict=True):
            if len(outcome) != dimensions or any(
                len(member) != dimensions for member in forecast.members
            ):
                raise ValueError("all vector dimensions must match the bandwidth vector")
            if forecast.weights is not None and (
                len(forecast.weights) != len(forecast.members) or not any(forecast.weights)
            ):
                raise ValueError("weights must align with members and have positive total mass")
            size = len(forecast.members)
            work += dimensions * (size + size * (size - 1) // 2)
        if work > 250_000:
            raise ValueError("kernel score exceeds 250000 scalar distance-work units")
        return self


class Row(OutputModel):
    source_index: int
    kernel_score: float
    twice_observation_dissimilarity: float
    pair_dissimilarity: float
    positive_weight_members: int
    zero_weight_members: int
    evaluated_distances: int
    dissimilarities_rounded_to_one: int
    negative_roundoff_detected: bool


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    bandwidths: list[float]
    mean_kernel_score: float
    scores: list[Row]
    negative_score_count: int
    dissimilarities_rounded_to_one: int
    kernel: Literal["gaussian_diagonal_bandwidth"] = "gaussian_diagonal_bandwidth"
    score_convention: Literal["one_plus_pair_kernel_minus_twice_observation_kernel"] = (
        "one_plus_pair_kernel_minus_twice_observation_kernel"
    )
    empirical_diagonal_pairs_included: Literal[True] = True
    finite_ensemble_correction_applied: Literal[False] = False
    negative_scores_clamped: Literal[False] = False
    bandwidth_selection_validated: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} cannot be represented as a finite nonzero float")
    return result


def _dissimilarity(
    left: list[Fraction], right: list[Fraction], scales: list[Fraction]
) -> tuple[Fraction, bool]:
    exponent = (
        sum(
            (((a - b) / scale) ** 2 for a, b, scale in zip(left, right, scales, strict=True)),
            Fraction(),
        )
        / 2
    )
    # Beyond this bound exp(-x) is too small to affect rounded 1-exp(-x).
    # Avoid converting huge exact ratios to floats solely to get that result.
    if exponent > 64:
        return Fraction(1), True
    result = -expm1(-_number(exponent, "Gaussian kernel exponent"))
    if exponent and result == 0:
        raise ValueError("nonzero Gaussian dissimilarity underflows the output format")
    return Fraction(result), bool(exponent and result == 1)


def execute(request: Input, context: OperationContext) -> Output:
    scales = [Fraction(value) for value in request.bandwidths]
    rows: list[Row] = []
    exact_scores: list[Fraction] = []
    for index, (outcome, forecast) in enumerate(
        zip(request.outcomes, request.forecasts, strict=True)
    ):
        supplied_weights = forecast.weights or [1.0] * len(forecast.members)
        active = [
            ([Fraction(value) for value in member], Fraction(weight))
            for member, weight in zip(forecast.members, supplied_weights, strict=True)
            if weight > 0
        ]
        total = sum((weight for _, weight in active), Fraction())
        weights = [weight / total for _, weight in active]
        target = [Fraction(value) for value in outcome]
        observation_term = pair_term = Fraction()
        saturated = evaluated = 0
        for member_index, (member, _) in enumerate(active):
            difference, was_saturated = _dissimilarity(member, target, scales)
            evaluated += 1
            saturated += was_saturated
            observation_term += 2 * weights[member_index] * difference
            for other_index in range(member_index):
                difference, was_saturated = _dissimilarity(member, active[other_index][0], scales)
                evaluated += 1
                saturated += was_saturated
                pair_term += 2 * weights[member_index] * weights[other_index] * difference
        score = observation_term - pair_term
        exact_scores.append(score)
        rows.append(
            Row(
                source_index=index,
                kernel_score=_number(score, "kernel score"),
                twice_observation_dissimilarity=_number(observation_term, "observation term"),
                pair_dissimilarity=_number(pair_term, "pair term"),
                positive_weight_members=len(active),
                zero_weight_members=len(forecast.members) - len(active),
                evaluated_distances=evaluated,
                dissimilarities_rounded_to_one=saturated,
                negative_roundoff_detected=score < 0,
            )
        )
    return Output(
        observation_count=len(rows),
        dimension_count=len(scales),
        bandwidths=request.bandwidths,
        mean_kernel_score=_number(sum(exact_scores, Fraction()) / len(rows), "mean kernel score"),
        scores=rows,
        negative_score_count=sum(row.negative_roundoff_detected for row in rows),
        dissimilarities_rounded_to_one=sum(row.dissimilarities_rounded_to_one for row in rows),
    )


OPERATION = Operation(
    id="skills.score_probability_kernel",
    kind="skill",
    description=(
        "Score weighted multivariate empirical forecasts with a fixed-bandwidth Gaussian "
        "kernel, diagonal-inclusive pair masses and stable dissimilarities. Reports numerical "
        "saturation and negative roundoff without bandwidth fitting or calibration claims."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
