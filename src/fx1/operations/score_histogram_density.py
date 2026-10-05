"""Logarithmic loss of piecewise-uniform forecast densities on explicit bins.

Each row supplies nonnegative relative bin masses; they are normalized exactly
before dividing by bin widths. Bins are [left,right), except the final right
endpoint is included. Outside support and zero-mass bins imply infinite loss;
no clipping or density floor changes those events. Continuous density log loss
can be negative and depends on the coordinate units, so compare like units.

Mass normalization, widths and weighted aggregation use exact binary-rational
arithmetic. Logarithms are rounded floating approximations; near-unit density
ratios use log1p to avoid subtracting nearly equal logarithms. Nonzero final
quantities below representable range fail explicitly. Outcome-independent row
weights are required for the usual proper-score interpretation.
Reference: Gneiting and Raftery (2007), section 4.1,
https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from itertools import pairwise
from math import isfinite, log, log1p
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Mass = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
MassRow = Annotated[list[Mass], Field(min_length=1, max_length=256)]


class Input(InputModel):
    bin_edges: list[Value] = Field(min_length=2, max_length=257)
    observations: list[Value] = Field(min_length=1, max_length=2000)
    bin_masses: list[MassRow] = Field(min_length=1, max_length=2000)
    observation_weights: list[Mass] | None = Field(default=None, min_length=1, max_length=2000)
    offset: int = Field(default=0, strict=True, ge=0, le=2000)
    limit: int = Field(default=100, strict=True, ge=1, le=500)

    @model_validator(mode="after")
    def coherent_histograms(self) -> Self:
        if any(right <= left for left, right in pairwise(self.bin_edges)):
            raise ValueError("bin edges must be strictly increasing")
        if len(self.observations) != len(self.bin_masses):
            raise ValueError("each observation requires one bin-mass row")
        width = len(self.bin_edges) - 1
        if len(self.bin_masses) * width > 100_000:
            raise ValueError("histogram forecasts exceed 100000 cells")
        if any(len(row) != width or not any(row) for row in self.bin_masses):
            raise ValueError("each bin-mass row must match the bins and have positive total mass")
        if self.observation_weights is not None and (
            len(self.observation_weights) != len(self.observations)
            or not any(self.observation_weights)
        ):
            raise ValueError("observation weights must align and have positive total mass")
        return self


class RowScore(OutputModel):
    row_index: int
    bin_index: int | None
    log_loss: float | None
    status: Literal["finite", "outside_support", "zero_bin_mass"]
    included_in_mean: bool
    total_input_mass: float


class Output(OutputModel):
    observation_count: int
    bin_count: int
    positive_weight_rows: int
    zero_weight_rows: int
    outside_support_rows: int
    zero_bin_mass_rows: int
    positive_weight_infinite_rows: int
    mean_log_loss: float | None
    mean_status: Literal["finite", "positive_infinity"]
    total_observation_weight: float
    rows: list[RowScore]
    next_offset: int | None
    mass_policy: Literal["normalize_each_row_before_dividing_by_bin_width"] = (
        "normalize_each_row_before_dividing_by_bin_width"
    )
    bin_policy: Literal["left_closed_right_open_except_final_endpoint"] = (
        "left_closed_right_open_except_final_endpoint"
    )
    log_base: Literal["natural"] = "natural"


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside the supported floating-point range")
    return result


def _log_ratio(value: Fraction) -> float:
    deviation = value - 1
    if abs(deviation) <= Fraction(1, 2):
        result = log1p(_number(deviation, "near-unit density deviation"))
    else:
        result = log(value.numerator) - log(value.denominator)
    if not isfinite(result) or (value != 1 and result == 0):
        raise ValueError("log density is outside the supported floating-point range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    """Score all rows; pagination affects row detail, never the weighted aggregate."""
    weights = request.observation_weights or [1.0] * len(request.observations)
    exact_weights = [Fraction.from_float(weight) for weight in weights]
    total_weight = sum(exact_weights, Fraction())
    widths = [Fraction(right) - Fraction(left) for left, right in pairwise(request.bin_edges)]
    outside = zero_mass = infinite = 0
    weighted_score = Fraction()
    scores: list[RowScore] = []
    for index, (observation, masses, weight) in enumerate(
        zip(request.observations, request.bin_masses, exact_weights, strict=True)
    ):
        total_mass = sum((Fraction(mass) for mass in masses), Fraction())
        bin_index: int | None = None
        loss: float | None = None
        status: Literal["finite", "outside_support", "zero_bin_mass"]
        if observation < request.bin_edges[0] or observation > request.bin_edges[-1]:
            status = "outside_support"
            outside += 1
        else:
            bin_index = min(bisect_right(request.bin_edges, observation) - 1, len(masses) - 1)
            mass = masses[bin_index]
            if mass == 0:
                status = "zero_bin_mass"
                zero_mass += 1
            else:
                status = "finite"
                loss = _log_ratio(widths[bin_index] * total_mass / Fraction(mass))
                weighted_score += weight * Fraction(loss)
        if status != "finite" and weight:
            infinite += 1
        if request.offset <= index < request.offset + request.limit:
            scores.append(
                RowScore(
                    row_index=index,
                    bin_index=bin_index,
                    log_loss=loss,
                    status=status,
                    included_in_mean=bool(weight),
                    total_input_mass=_number(total_mass, "total bin mass"),
                )
            )
    next_offset = request.offset + len(scores)
    return Output(
        observation_count=len(request.observations),
        bin_count=len(widths),
        positive_weight_rows=sum(bool(weight) for weight in exact_weights),
        zero_weight_rows=sum(not weight for weight in exact_weights),
        outside_support_rows=outside,
        zero_bin_mass_rows=zero_mass,
        positive_weight_infinite_rows=infinite,
        mean_log_loss=None if infinite else _number(weighted_score / total_weight, "mean log loss"),
        mean_status="positive_infinity" if infinite else "finite",
        total_observation_weight=_number(total_weight, "total observation weight"),
        rows=scores,
        next_offset=next_offset if next_offset < len(request.observations) else None,
    )


OPERATION = Operation(
    id="skills.score_histogram_density",
    kind="skill",
    description=(
        "Compute normalized piecewise-uniform density log loss with explicit bin endpoints, "
        "zero/outside-support infinite losses, optional observation weights and paginated details."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
