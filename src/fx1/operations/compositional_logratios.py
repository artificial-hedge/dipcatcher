"""Closure, centered/Helmert log ratios and variation of positive compositions.

Each row represents positive relative parts. Closure divides by its exact sum.
Parts are sorted and adjacent log ratios use log1p near equality. Exact sums
of these rounded increments give anchored logs; their exact mean gives CLR
coordinates. This preserves small contrasts beside very different parts.
Helmert balance j is
sqrt(j/(j+1))*(mean(log x_1..log x_j)-log x_(j+1)), j=1..D-1.
The basis depends on supplied part order; no basis or zero replacement is fitted.
Coordinate conventions: https://scikit.bio/docs/latest/generated/skbio.stats.composition.clr.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.helmert.html

The closed componentwise geometric center uses mean CLR over all rows. The
variation matrix is the population variance (divisor N) of each log(x_i/x_j),
centered against the first row's exact ratio before taking logs. Its entries
depend only on their two input parts, preserving small changes of large ratios.
Distances to that center use CLR Euclidean distance (Aitchison distance).
Returned ILR coordinates and the reported rounded basis reconstruct compositions
for diagnostics; finite rounding prevents an exact numerical isometry claim.

Limits: 1000 rows, 2..32 parts, 20000 row-part cells, 500000 row-part-squared work
cells and parts between 1e-100 and 1e100. No pseudocounts, zero replacement, units,
independence, timing, significance or forecast-performance claim is inferred.
The common part universe and availability of the whole fitting block are caller
assumptions. Logs, roots and exponentials remain numerical approximations.
"""

from __future__ import annotations

from fractions import Fraction
from math import exp, fsum, isfinite, log, log1p, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Part = Annotated[float, Field(strict=True, ge=1e-100, le=1e100)]
Composition = Annotated[list[Part], Field(min_length=2, max_length=32)]
Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class Input(InputModel):
    compositions: list[Composition] = Field(min_length=1, max_length=1000)
    part_names: list[Name] | None = Field(default=None, min_length=2, max_length=32)

    @model_validator(mode="after")
    def rectangular_bounded_input(self) -> Self:
        dimension = len(self.compositions[0])
        if any(len(row) != dimension for row in self.compositions):
            raise ValueError("all composition rows must have the same parts")
        if self.part_names is not None and (
            len(self.part_names) != dimension or len(set(self.part_names)) != dimension
        ):
            raise ValueError("part names must be distinct and align with composition columns")
        if (
            len(self.compositions) * dimension > 20_000
            or len(self.compositions) * dimension * dimension > 500_000
        ):
            raise ValueError(
                "composition work exceeds 20000 cells or 500000 squared-dimension cells"
            )
        return self


class Row(OutputModel):
    row_index: int
    input_total: float
    closed_composition: list[float]
    centered_logratios: list[float]
    isometric_logratios: list[float]
    aitchison_distance_to_geometric_center: float
    returned_clr_sum: float
    clr_ilr_norm_disagreement: float
    reconstruction_maximum_absolute_error: float


class Output(OutputModel):
    observation_count: int
    part_count: int
    part_names: list[str] | None
    helmert_basis_rows: list[list[float]]
    geometric_center: list[float]
    center_clr: list[float]
    logratio_variation_matrix: list[list[float]]
    rows: list[Row]
    variation_divisor: Literal["observation_count_population"] = "observation_count_population"
    log_base: Literal["natural"] = "natural"
    balance_convention: Literal["positive_first_j_parts_versus_negative_next_part"] = (
        "positive_first_j_parts_versus_negative_next_part"
    )
    reconstruction_uses_returned_ilr_and_basis: Literal[True] = True
    zero_replacement_applied: Literal[False] = False
    common_part_universe_verified: Literal[False] = False
    timing_verified: Literal[False] = False


def _number(value: Fraction) -> float:
    result = float(value)
    if not isfinite(result) or (value and result == 0):
        raise ValueError("composition result is outside finite nonzero binary64 range")
    return result


def _norm(values: list[Fraction]) -> float:
    squared = sum((value * value for value in values), Fraction())
    result = correctly_rounded_sqrt(squared.numerator, squared.denominator)
    if not isfinite(result) or (squared and result == 0):
        raise ValueError("composition norm is outside finite nonzero binary64 range")
    return result


def _log_ratio(numerator: Fraction, denominator: Fraction) -> Fraction:
    if numerator < denominator:
        return -_log_ratio(denominator, numerator)
    ratio = numerator / denominator
    if ratio <= 2:
        return Fraction(log1p(_number(ratio - 1)))
    # Relative ratios across rows can exceed binary64 range. Reduce exactly
    # before transcendental evaluation; retain the two rounded terms separately.
    exponent = ratio.numerator.bit_length() - ratio.denominator.bit_length()
    if ratio.numerator < ratio.denominator << exponent:
        exponent -= 1
    mantissa = Fraction(ratio.numerator, ratio.denominator << exponent)
    return Fraction(log1p(_number(mantissa - 1))) + exponent * Fraction(log(2))


def _close_logs(values: list[Fraction]) -> list[float]:
    largest = max(values)
    components = [exp(_number(value - largest)) for value in values]
    if any(not isfinite(value) or value <= 0 for value in components):
        raise ValueError("positive reconstructed composition is outside numerical range")
    total = sum((Fraction(value) for value in components), Fraction())
    return [_number(Fraction(value) / total) for value in components]


def execute(request: Input, context: OperationContext) -> Output:
    count, dimension = len(request.compositions), len(request.compositions[0])
    closed: list[list[Fraction]] = []
    original_parts: list[list[Fraction]] = []
    totals: list[Fraction] = []
    clrs: list[list[Fraction]] = []
    for values in request.compositions:
        parts = [Fraction(value) for value in values]
        original_parts.append(parts)
        total = sum(parts, Fraction())
        totals.append(total)
        closed.append([value / total for value in parts])
        logs = [Fraction() for _ in parts]
        ordering = sorted(range(dimension), key=parts.__getitem__)
        accumulated = Fraction()
        for position in range(1, dimension):
            previous, current = ordering[position - 1], ordering[position]
            accumulated += _log_ratio(parts[current], parts[previous])
            logs[current] = accumulated
        mean = sum(logs, Fraction()) / dimension
        clrs.append([value - mean for value in logs])
    center = [sum((row[column] for row in clrs), Fraction()) / count for column in range(dimension)]
    basis: list[list[float]] = []
    for width in range(1, dimension):
        coefficient = sqrt(width / (width + 1))
        basis.append(
            [coefficient / width] * width + [-coefficient] + [0.0] * (dimension - width - 1)
        )
    variation = [[0.0] * dimension for _ in range(dimension)]
    for first in range(dimension):
        for second in range(first):
            reference = original_parts[0]
            ratios = [
                _log_ratio(row[first] * reference[second], row[second] * reference[first])
                for row in original_parts
            ]
            mean_ratio = sum(ratios, Fraction()) / count
            variance = sum(((value - mean_ratio) ** 2 for value in ratios), Fraction()) / count
            variation[first][second] = variation[second][first] = _number(variance)
    rows: list[Row] = []
    for index, clr in enumerate(clrs):
        # Use precisely the reported basis, including its rounded coefficients.
        ilr = [
            _number(
                sum(
                    (
                        value * Fraction(coefficient)
                        for value, coefficient in zip(clr, balance, strict=True)
                    ),
                    Fraction(),
                )
            )
            for balance in basis
        ]
        reconstructed_clr = [
            sum(
                (
                    Fraction(ilr[axis]) * Fraction(balance[column])
                    for axis, balance in enumerate(basis)
                ),
                Fraction(),
            )
            for column in range(dimension)
        ]
        reconstruction = _close_logs(reconstructed_clr)
        reported_clr = [_number(value) for value in clr]
        norm_disagreement = abs(
            _norm([Fraction(value) for value in reported_clr])
            - _norm([Fraction(value) for value in ilr])
        )
        rows.append(
            Row(
                row_index=index,
                input_total=_number(totals[index]),
                closed_composition=[_number(value) for value in closed[index]],
                centered_logratios=reported_clr,
                isometric_logratios=ilr,
                aitchison_distance_to_geometric_center=_norm(
                    [value - location for value, location in zip(clr, center, strict=True)]
                ),
                returned_clr_sum=fsum(reported_clr),
                clr_ilr_norm_disagreement=norm_disagreement,
                reconstruction_maximum_absolute_error=_number(
                    max(
                        abs(Fraction(value) - original)
                        for value, original in zip(reconstruction, closed[index], strict=True)
                    )
                ),
            )
        )
    return Output(
        observation_count=count,
        part_count=dimension,
        part_names=request.part_names,
        helmert_basis_rows=basis,
        geometric_center=_close_logs(center),
        center_clr=[_number(value) for value in center],
        logratio_variation_matrix=variation,
        rows=rows,
    )


OPERATION = Operation(
    id="features.compositional_logratios",
    kind="feature",
    description="Transform strictly positive compositions into closure, CLR and ordered Helmert balances; compute the geometric center and log-ratio variation matrix with returned-coordinate reconstruction diagnostics.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
