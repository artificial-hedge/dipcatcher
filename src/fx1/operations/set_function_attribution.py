"""Exact coalition transforms and attribution of an explicitly supplied set function.

values[mask] contains v(S), with bit i selecting names[i]. Every coalition must
be present; missing model evaluations are never inferred. The empty coalition
may have a nonzero baseline. Own subset differencing computes Möbius dividends;
the inverse subset sum is checked against every supplied value.

Shapley attribution averages marginal contributions over uniform permutations;
Banzhaf attribution averages them over uniformly chosen subsets. Pair Shapley
interaction is sum of dividends containing both players divided by |S|-1.
Pair interactions are not additive pieces to sum with individual Shapley values.
Monotonicity checks compare all one-player additions, including negative values.
These are algebraic properties of the supplied table, not causal importance,
model faithfulness, feature independence or uncertainty estimates.
Reference: Grabisch et al., Equivalent Representations of Set Functions (2000),
https://pubsonline.informs.org/doi/10.1287/moor.25.2.157.12225
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class Input(InputModel):
    names: list[Name] = Field(min_length=1, max_length=12)
    values: list[Value] = Field(min_length=2, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=4096)
    limit: int = Field(default=50, strict=True, ge=1, le=128)
    max_monotonicity_findings: int = Field(default=20, strict=True, ge=0, le=128)

    @model_validator(mode="after")
    def complete_table(self) -> Self:
        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique")
        if len(self.values) != 1 << len(self.names):
            raise ValueError("values must contain exactly 2**len(names) coalition entries")
        return self


class Number(OutputModel):
    numerator: str
    denominator: str
    value: float
    preview_underflow: bool


class Attribution(OutputModel):
    player_index: int
    name: str
    shapley: Number
    banzhaf: Number
    standalone_gain: Number
    grand_coalition_removal_loss: Number
    null_player: bool
    negative_marginal_count: int


class Pair(OutputModel):
    first_player: int
    second_player: int
    shapley_interaction: Number


class Coalition(OutputModel):
    mask: int
    member_indices: list[int]
    value: float
    mobius_dividend: Number


class MonotonicityFinding(OutputModel):
    base_mask: int
    added_player: int
    marginal: Number


class Output(OutputModel):
    player_count: int
    coalition_count: int
    baseline: Number
    grand_coalition_value: Number
    grand_coalition_gain: Number
    attributions: list[Attribution]
    pair_interactions: list[Pair]
    nonzero_dividend_count_by_order: list[int]
    dividend_sum_by_order: list[Number]
    monotone: bool
    negative_marginal_count: int
    monotonicity_findings: list[MonotonicityFinding]
    omitted_monotonicity_findings: int
    coalitions: list[Coalition]
    offset: int
    has_more: bool
    exact_shapley_efficiency_checked: Literal[True] = True
    exact_transform_reconstruction_checked: Literal[True] = True
    model_or_causal_importance_verified: Literal[False] = False
    table_availability_verified: Literal[False] = False


def _number(value: Fraction) -> Number:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 12_000:
        raise ValueError("set-function result exceeds the 12000-bit serialization budget")
    preview = float(value)
    return Number(
        numerator=str(value.numerator),
        denominator=str(value.denominator),
        value=preview,
        preview_underflow=bool(value) and preview == 0,
    )


def execute(request: Input, context: OperationContext) -> Output:
    dimension = len(request.names)
    size = len(request.values)
    values = [Fraction(value) for value in request.values]
    dividends = values.copy()
    for player in range(dimension):
        bit = 1 << player
        for mask in range(size):
            if mask & bit:
                dividends[mask] -= dividends[mask ^ bit]
    reconstructed = dividends.copy()
    for player in range(dimension):
        bit = 1 << player
        for mask in range(size):
            if mask & bit:
                reconstructed[mask] += reconstructed[mask ^ bit]
    if reconstructed != values:
        raise ArithmeticError("subset transform failed exact reconstruction")
    shapley = [Fraction() for _ in request.names]
    banzhaf = [Fraction() for _ in request.names]
    interaction = {(i, j): Fraction() for i in range(dimension) for j in range(i + 1, dimension)}
    order_counts = [0] * (dimension + 1)
    order_sums = [Fraction() for _ in range(dimension + 1)]
    for mask, dividend in enumerate(dividends):
        order = mask.bit_count()
        order_counts[order] += bool(dividend)
        order_sums[order] += dividend
        if not mask or not dividend:
            continue
        members = [player for player in range(dimension) if mask & (1 << player)]
        for player in members:
            shapley[player] += dividend / order
            banzhaf[player] += dividend / (1 << (order - 1))
        for offset, first in enumerate(members):
            for second in members[offset + 1 :]:
                interaction[first, second] += dividend / (order - 1)
    gain = values[-1] - values[0]
    if sum(shapley, Fraction()) != gain:
        raise ArithmeticError("Shapley attributions failed exact baseline-adjusted efficiency")
    negative = [0] * dimension
    null_players = [True] * dimension
    findings: list[MonotonicityFinding] = []
    for player in range(dimension):
        bit = 1 << player
        for mask in range(size):
            if mask & bit:
                continue
            marginal = values[mask | bit] - values[mask]
            null_players[player] &= marginal == 0
            if marginal < 0:
                negative[player] += 1
                if len(findings) < request.max_monotonicity_findings:
                    findings.append(
                        MonotonicityFinding(
                            base_mask=mask, added_player=player, marginal=_number(marginal)
                        )
                    )
    return Output(
        player_count=dimension,
        coalition_count=size,
        baseline=_number(values[0]),
        grand_coalition_value=_number(values[-1]),
        grand_coalition_gain=_number(gain),
        attributions=[
            Attribution(
                player_index=player,
                name=name,
                shapley=_number(shapley[player]),
                banzhaf=_number(banzhaf[player]),
                standalone_gain=_number(values[1 << player] - values[0]),
                grand_coalition_removal_loss=_number(
                    values[-1] - values[(size - 1) ^ (1 << player)]
                ),
                null_player=null_players[player],
                negative_marginal_count=negative[player],
            )
            for player, name in enumerate(request.names)
        ],
        pair_interactions=[
            Pair(first_player=i, second_player=j, shapley_interaction=_number(value))
            for (i, j), value in interaction.items()
        ],
        nonzero_dividend_count_by_order=order_counts,
        dividend_sum_by_order=[_number(value) for value in order_sums],
        monotone=not any(negative),
        negative_marginal_count=sum(negative),
        monotonicity_findings=findings,
        omitted_monotonicity_findings=sum(negative) - len(findings),
        coalitions=[
            Coalition(
                mask=mask,
                member_indices=[player for player in range(dimension) if mask & (1 << player)],
                value=request.values[mask],
                mobius_dividend=_number(dividends[mask]),
            )
            for mask in range(request.offset, min(request.offset + request.limit, size))
        ],
        offset=request.offset,
        has_more=request.offset + request.limit < size,
    )


OPERATION = Operation(
    id="features.set_function_attribution",
    kind="feature",
    description="Decompose a complete supplied coalition table by exact subset inversion into Shapley/Banzhaf attributions, pair interactions and monotonicity diagnostics with no causal or model-evaluation claim.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
