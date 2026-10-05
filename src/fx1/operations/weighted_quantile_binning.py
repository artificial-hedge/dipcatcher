"""Fit tie-preserving bins from the inverse weighted empirical CDF.

At target j/B choose the smallest positive-weight support value whose cumulative
weight reaches j/B. Equal selected values collapse into one cut, and a cut equal
to the maximum positive-weight value is discarded to avoid an empty weighted
upper bin. Bins are (-infinity, c1], (c1, c2], ..., (c_last, infinity).
Ties are never split. Consequently neither equal row counts nor equal bin masses
are guaranteed, and fewer than B bins can result. Zero-weight rows do not fit
the CDF but receive bin IDs and contribute to unweighted bin counts.

All cumulative mass comparisons use exact fractions of supplied binary floats;
there is no interpolation, jitter, learned clipping or dependency on input order.
Queries reuse fitted cuts without changing them. Availability, leakage and
representativeness of the fitting sample remain caller responsibilities.
Weighted-CDF convention: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=10_000)
    queries: list[Value] = Field(default_factory=list, max_length=10_000)
    requested_bins: int = Field(default=10, strict=True, ge=2, le=128)

    @model_validator(mode="after")
    def aligned_weights(self) -> Self:
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError("weights must align with fitting values")
            if not any(self.weights):
                raise ValueError("at least one fitting weight must be positive")
            if any(0 < weight < 1e-100 for weight in self.weights):
                raise ValueError("positive fitting weights must be at least 1e-100")
        return self


class Boundary(OutputModel):
    target_numerator: int
    target_denominator: int
    selected_support_value: float
    cut_index: int | None
    status: Literal["retained", "duplicate_cut", "maximum_support_cut_discarded"]


class Bin(OutputModel):
    bin_id: int
    lower_exclusive: float | None
    upper_inclusive: float | None
    fitting_row_count: int
    positive_weight_row_count: int
    fitting_weight: float
    fitting_mass_fraction: float
    minimum_fitting_value: float | None
    maximum_fitting_value: float | None
    query_row_count: int


class Output(OutputModel):
    fitting_row_count: int
    query_row_count: int
    positive_weight_row_count: int
    distinct_positive_weight_values: int
    total_fitting_weight: float
    requested_bins: int
    effective_bins: int
    collapsed_boundary_count: int
    cutpoints: list[float]
    boundaries: list[Boundary]
    bins: list[Bin]
    fitting_bin_ids: list[int]
    query_bin_ids: list[int]
    interval_convention: Literal["lower_open_upper_closed_with_unbounded_outer_bins"] = (
        "lower_open_upper_closed_with_unbounded_outer_bins"
    )
    quantile_method: Literal["inverse_exact_weighted_empirical_cdf"] = (
        "inverse_exact_weighted_empirical_cdf"
    )
    ties_split: Literal[False] = False
    availability_validated: Literal[False] = False


def execute(request: Input, context: OperationContext) -> Output:
    weights = (
        [Fraction(weight) for weight in request.weights]
        if request.weights is not None
        else [Fraction(1)] * len(request.values)
    )
    grouped: dict[float, Fraction] = defaultdict(Fraction)
    for value, weight in zip(request.values, weights, strict=True):
        if weight:
            grouped[value] += weight
    support = sorted(grouped)
    total = sum(grouped.values(), Fraction())
    position = 0
    cumulative = grouped[support[0]]
    cuts: list[float] = []
    boundaries: list[Boundary] = []
    for numerator in range(1, request.requested_bins):
        target = total * Fraction(numerator, request.requested_bins)
        while cumulative < target:
            position += 1
            cumulative += grouped[support[position]]
        selected = support[position]
        status: Literal["retained", "duplicate_cut", "maximum_support_cut_discarded"]
        cut_index: int | None
        if selected == support[-1]:
            status, cut_index = "maximum_support_cut_discarded", None
        elif cuts and selected == cuts[-1]:
            status, cut_index = "duplicate_cut", len(cuts) - 1
        else:
            cuts.append(selected)
            status, cut_index = "retained", len(cuts) - 1
        boundaries.append(
            Boundary(
                target_numerator=numerator,
                target_denominator=request.requested_bins,
                selected_support_value=selected,
                cut_index=cut_index,
                status=status,
            )
        )
    fitting_ids = [bisect_left(cuts, value) for value in request.values]
    query_ids = [bisect_left(cuts, value) for value in request.queries]
    count = len(cuts) + 1
    rows = [0] * count
    positive_rows = [0] * count
    query_rows = [0] * count
    bin_weights = [Fraction() for _ in range(count)]
    minima: list[float | None] = [None] * count
    maxima: list[float | None] = [None] * count
    for value, weight, index in zip(request.values, weights, fitting_ids, strict=True):
        rows[index] += 1
        positive_rows[index] += bool(weight)
        bin_weights[index] += weight
        lower, upper = minima[index], maxima[index]
        minima[index] = value if lower is None else min(value, lower)
        maxima[index] = value if upper is None else max(value, upper)
    for index in query_ids:
        query_rows[index] += 1
    return Output(
        fitting_row_count=len(request.values),
        query_row_count=len(request.queries),
        positive_weight_row_count=sum(bool(weight) for weight in weights),
        distinct_positive_weight_values=len(support),
        total_fitting_weight=float(total),
        requested_bins=request.requested_bins,
        effective_bins=count,
        collapsed_boundary_count=request.requested_bins - count,
        cutpoints=cuts,
        boundaries=boundaries,
        bins=[
            Bin(
                bin_id=index,
                lower_exclusive=cuts[index - 1] if index else None,
                upper_inclusive=cuts[index] if index < len(cuts) else None,
                fitting_row_count=rows[index],
                positive_weight_row_count=positive_rows[index],
                fitting_weight=float(bin_weights[index]),
                fitting_mass_fraction=float(bin_weights[index] / total),
                minimum_fitting_value=minima[index],
                maximum_fitting_value=maxima[index],
                query_row_count=query_rows[index],
            )
            for index in range(count)
        ],
        fitting_bin_ids=fitting_ids,
        query_bin_ids=query_ids,
    )


OPERATION = Operation(
    id="features.weighted_quantile_binning",
    kind="feature",
    description="Fit tie-preserving right-closed bins from an exact weighted empirical CDF, report collapsed quantile boundaries and training masses, and encode separate queries without refitting.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
