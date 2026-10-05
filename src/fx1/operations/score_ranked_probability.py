"""Ranked probability loss for explicitly ordered, mutually exclusive categories.

RPS = sum_j (F_j - 1{outcome <= j})**2 over K-1 category boundaries.
The optional normalization divides by K-1. Category spacing is not an input;
this scores ordinal categories, not distances between numeric labels. See
Gneiting and Raftery (2007), section 6, doi:10.1198/016214506000001437.
"""

from math import frexp, fsum, ldexp
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Category = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
ProbabilityRow = Annotated[list[Probability], Field(min_length=2, max_length=100)]


class Input(InputModel):
    ordered_categories: list[Category] = Field(min_length=2, max_length=100)
    outcomes: list[Category] = Field(min_length=1, max_length=10_000)
    probabilities: list[ProbabilityRow] = Field(min_length=1, max_length=10_000)
    normalization: Literal["none", "divide_by_boundaries"] = "none"
    mass_tolerance: float = Field(default=1e-12, strict=True, ge=0, le=1e-6)

    @model_validator(mode="after")
    def coherent_forecasts(self) -> Self:
        categories = set(self.ordered_categories)
        width = len(self.ordered_categories)
        if len(categories) != width:
            raise ValueError("ordered category labels must be unique")
        if len(self.outcomes) != len(self.probabilities):
            raise ValueError("each outcome requires one probability row")
        if any(outcome not in categories for outcome in self.outcomes):
            raise ValueError("every outcome must belong to the declared ordered categories")
        if len(self.outcomes) * width * width > 2_000_000:
            raise ValueError("RPS exceeds 2000000 cumulative-sum work units")
        for row in self.probabilities:
            if len(row) != width:
                raise ValueError("each probability row must match the category count")
            if abs(fsum(row) - 1.0) > self.mass_tolerance:
                raise ValueError("probability mass must be within mass_tolerance of one")
        return self


class Output(OutputModel):
    ordered_categories: list[str]
    observation_count: int
    boundary_count: int
    normalization: str
    mean_ranked_probability_score: float
    scores: list[float]
    mean_boundary_losses: list[float]
    renormalized_row_count: int
    maximum_input_mass_error: float
    probability_policy: Literal["divide_each_accepted_row_by_its_sum"] = (
        "divide_each_accepted_row_by_its_sum"
    )


def _squared_sum(values: list[float], divisor: int = 1) -> float:
    """Retain small terms until their sum and normalization are applied."""
    terms = []
    for value in values:
        if value:
            mantissa, exponent = frexp(value)
            terms.append((mantissa * mantissa, 2 * exponent))
    if not terms:
        return 0.0
    exponent = max(item[1] for item in terms)
    total = fsum(ldexp(mantissa, power - exponent) for mantissa, power in terms)
    return ldexp(total / divisor, exponent)


def execute(request: Input, context: OperationContext) -> Output:
    positions = {name: index for index, name in enumerate(request.ordered_categories)}
    boundaries = len(positions) - 1
    divisor = boundaries if request.normalization == "divide_by_boundaries" else 1
    scores: list[float] = []
    errors_by_boundary: list[list[float]] = [[] for _ in range(boundaries)]
    changed = 0
    mass_errors: list[float] = []
    for label, probabilities in zip(request.outcomes, request.probabilities, strict=True):
        total = fsum(probabilities)
        changed += int(total != 1.0)
        mass_errors.append(abs(total - 1.0))
        outcome = positions[label]
        errors: list[float] = []
        for boundary in range(boundaries):
            # Sum the error-side mass directly, avoiding 1-CDF cancellation.
            mass = fsum(
                probabilities[: boundary + 1]
                if outcome > boundary
                else probabilities[boundary + 1 :]
            )
            error = mass / total
            errors.append(error)
            errors_by_boundary[boundary].append(error)
        scores.append(_squared_sum(errors, divisor))
    return Output(
        ordered_categories=request.ordered_categories,
        observation_count=len(scores),
        boundary_count=boundaries,
        normalization=request.normalization,
        mean_ranked_probability_score=_squared_sum(
            [error for errors in errors_by_boundary for error in errors], len(scores) * divisor
        ),
        scores=scores,
        mean_boundary_losses=[
            _squared_sum(errors, len(scores) * divisor) for errors in errors_by_boundary
        ],
        renormalized_row_count=changed,
        maximum_input_mass_error=max(mass_errors),
    )


OPERATION = Operation(
    id="skills.score_ranked_probability",
    kind="skill",
    description=(
        "Score ordered categorical probability forecasts by cumulative boundary losses, "
        "with explicit mass tolerance/renormalization, optional K-1 normalization and "
        "per-boundary contributions. Lower is better; category spacing is not inferred."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
