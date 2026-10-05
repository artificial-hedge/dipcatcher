"""A weighted sum of binary Brier losses over a supplied rooted class tree.

Each leaf is a mutually exclusive outcome. Nonnegative supplied leaf masses
are normalized, then summed up the tree. A node's target is one precisely when
the realized leaf descends from it. Score=sum_v(w_v*(p_v-y_v)^2)/sum_v(w_v),
excluding the root. All nonroot weights are strictly positive and fixed across
observations, so positive leaf losses identify the full leaf distribution.
This constructs a strictly proper score as a positive sum of proper binary
scores, provided the hierarchy/weights are selected independently of outcomes.
It does not claim equivalence to every published hierarchical loss.

Inputs provide leaf masses only; internal coherence is constructed, not repaired
from contradictory internal forecasts. Binary-float masses and weights enter
exact rational aggregation, with rounding only at the output boundary. Each
cross-row accumulator and final fraction is limited to 131072 numerator or
denominator bits; inputs exceeding that arithmetic budget fail explicitly.
There is no fitted hierarchy, node weighting, calibration or independence assessment.
Binary proper-score reference: Gneiting and Raftery (2007), section 4.2:
https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
Mass = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Weight = Annotated[float, Field(strict=True, gt=0, le=1e100)]
_MAX_FRACTION_BITS = 131_072


class Node(InputModel):
    node_id: Name
    parent_id: Name | None


class Forecast(InputModel):
    leaf_masses: dict[Name, Mass] = Field(min_length=2, max_length=128)


class Input(InputModel):
    nodes: list[Node] = Field(min_length=3, max_length=256)
    outcomes: list[Name] = Field(min_length=1, max_length=1000)
    forecasts: list[Forecast] = Field(min_length=1, max_length=1000)
    node_weights: dict[Name, Weight] | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def aligned_bounded_rows(self) -> Self:
        if len(self.outcomes) != len(self.forecasts):
            raise ValueError("forecasts must align with realized leaf outcomes")
        if len(self.nodes) * len(self.outcomes) > 100_000:
            raise ValueError("hierarchical scoring exceeds 100000 node-observation cells")
        return self


class NodeSummary(OutputModel):
    node_id: str
    parent_id: str
    depth: int
    descendant_leaf_count: int
    weight: float
    normalized_weight: float
    observed_descendant_count: int
    mean_probability: float
    mean_binary_brier: float
    mean_weighted_contribution: float


class Output(OutputModel):
    root_id: str
    leaf_ids: list[str]
    maximum_depth: int
    observation_count: int
    mean_hierarchical_brier: float
    scores: list[float]
    nodes: list[NodeSummary]
    normalized_input_mass_rows: int
    minimum_input_mass: float
    maximum_input_mass: float
    score_convention: Literal["positive_weighted_nonroot_binary_brier_average"] = (
        "positive_weighted_nonroot_binary_brier_average"
    )
    internal_probabilities: Literal["sum_of_normalized_descendant_leaf_masses"] = (
        "sum_of_normalized_descendant_leaf_masses"
    )
    hierarchy_and_weight_selection_validated: Literal[False] = False
    calibration_certified: Literal[False] = False
    maximum_fraction_bits: Literal[131072] = 131072


def _bounded(value: Fraction, label: str) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > _MAX_FRACTION_BITS:
        raise ValueError(f"{label} exceeds the 131072-bit exact arithmetic budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value, label)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside representable finite range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    names = [node.node_id for node in request.nodes]
    positions = {name: index for index, name in enumerate(names)}
    if len(positions) != len(names):
        raise ValueError("hierarchy node IDs must be distinct")
    roots = [index for index, node in enumerate(request.nodes) if node.parent_id is None]
    if len(roots) != 1:
        raise ValueError("hierarchy must contain exactly one root")
    root = roots[0]
    children: list[list[int]] = [[] for _ in names]
    parents: list[int | None] = []
    parent: int | None
    for index, node in enumerate(request.nodes):
        if node.parent_id is None:
            parents.append(None)
        elif node.parent_id not in positions or node.parent_id == node.node_id:
            raise ValueError("each nonroot parent must name a different declared node")
        else:
            parent = positions[node.parent_id]
            parents.append(parent)
            children[parent].append(index)
    order: list[int] = []
    depths = [0] * len(names)
    pending = deque([root])
    while pending:
        current = pending.popleft()
        order.append(current)
        for child in children[current]:
            depths[child] = depths[current] + 1
            if depths[child] > 64:
                raise ValueError("hierarchy exceeds 64 edges from root to leaf")
            pending.append(child)
    if len(order) != len(names):
        raise ValueError("hierarchy contains a cycle or nodes disconnected from its root")
    leaves = [index for index in range(len(names)) if not children[index]]
    leaf_names = [names[index] for index in leaves]
    if not 2 <= len(leaves) <= 128:
        raise ValueError("hierarchy must contain 2 through 128 leaves")
    expected_leaves = set(leaf_names)
    nonroot_names = set(names) - {names[root]}
    if request.node_weights is not None and set(request.node_weights) != nonroot_names:
        raise ValueError("node_weights must supply every nonroot node exactly once")
    if any(outcome not in expected_leaves for outcome in request.outcomes):
        raise ValueError("outcomes must name declared leaves")
    weights = [
        Fraction()
        if index == root
        else Fraction(1 if request.node_weights is None else request.node_weights[name])
        for index, name in enumerate(names)
    ]
    total_weight = sum(weights, Fraction())
    descendants = [int(index in leaves) for index in range(len(names))]
    for index in reversed(order):
        parent = parents[index]
        if parent is not None:
            descendants[parent] += descendants[index]
    probability_sums = [Fraction() for _ in names]
    loss_sums = [Fraction() for _ in names]
    observed = [0] * len(names)
    totals: list[Fraction] = []
    losses: list[Fraction] = []
    aggregate_loss = Fraction()
    for outcome, forecast in zip(request.outcomes, request.forecasts, strict=True):
        if set(forecast.leaf_masses) != expected_leaves:
            raise ValueError("each forecast must supply all and only declared leaf masses")
        total = sum((Fraction(value) for value in forecast.leaf_masses.values()), Fraction())
        if total <= 0:
            raise ValueError("each forecast needs positive total leaf mass")
        totals.append(total)
        probabilities = [Fraction() for _ in names]
        targets = [0] * len(names)
        targets[positions[outcome]] = 1
        for index in leaves:
            probabilities[index] = Fraction(forecast.leaf_masses[names[index]]) / total
        for index in reversed(order):
            parent = parents[index]
            if parent is not None:
                probabilities[parent] += probabilities[index]
                targets[parent] += targets[index]
        score = Fraction()
        for index in range(len(names)):
            loss = (probabilities[index] - targets[index]) ** 2
            probability_sums[index] = _bounded(
                probability_sums[index] + probabilities[index], "node probability accumulator"
            )
            observed[index] += targets[index]
            loss_sums[index] = _bounded(loss_sums[index] + loss, "node loss accumulator")
            score += weights[index] * loss / total_weight
        losses.append(score)
        aggregate_loss = _bounded(aggregate_loss + score, "hierarchy loss accumulator")
    count = len(losses)
    return Output(
        root_id=names[root],
        leaf_ids=leaf_names,
        maximum_depth=max(depths),
        observation_count=count,
        mean_hierarchical_brier=_number(aggregate_loss / count, "mean hierarchy score"),
        scores=[_number(loss, "hierarchy score") for loss in losses],
        nodes=[
            NodeSummary(
                node_id=name,
                parent_id=names[parent],
                depth=depths[index],
                descendant_leaf_count=descendants[index],
                weight=_number(weights[index], "node weight"),
                normalized_weight=_number(weights[index] / total_weight, "normalized node weight"),
                observed_descendant_count=observed[index],
                mean_probability=_number(probability_sums[index] / count, "node probability"),
                mean_binary_brier=_number(loss_sums[index] / count, "node Brier score"),
                mean_weighted_contribution=_number(
                    loss_sums[index] * weights[index] / (count * total_weight), "node contribution"
                ),
            )
            for index, name in enumerate(names)
            if (parent := parents[index]) is not None
        ],
        normalized_input_mass_rows=sum(total != 1 for total in totals),
        minimum_input_mass=_number(min(totals), "minimum input mass"),
        maximum_input_mass=_number(max(totals), "maximum input mass"),
    )


OPERATION = Operation(
    id="skills.score_hierarchical_probabilities",
    kind="skill",
    description=(
        "Aggregate leaf masses through a validated class tree and score each nonroot event "
        "with a positive fixed-weight Brier loss, retaining node contributions and explicit "
        "normalization. Hierarchy and weight selection assumptions remain unverified."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
