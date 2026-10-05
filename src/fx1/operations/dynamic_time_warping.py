"""Endpoint-constrained DTW with exact weighted squared-Euclidean costs.

For a coordinate weight vector w normalized to sum one, local cost(i,j) is
sum_d w_d*(left[i,d]-right[j,d])**2. Minimize the unnormalized sum of visited
cell costs, including both endpoints. Steps are (1,1), (1,0), (0,1), in that
predecessor tie priority; every visited cell has unit step weight. An optional
absolute index band permits exactly abs(i-j)<=band, without length rescaling.
This is squared-cost DTW, not a metric, minimum-average-cost alignment,
interpolation, edit distance, subsequence search or a temporal-causality claim.

Coordinates and weights become common integer units, so cost accumulation,
path comparisons and ties are exact for the supplied binary64 inputs. A
reported cost per path cell is descriptive and does not change optimization.
Nonzero float-output underflow fails. Bounds are 256 points per side, eight
dimensions, 16384 grid cells and 65536 grid-coordinate cells. Caller supplies
ordered, compatible coordinate units and enforces point-in-time selection.
Standard step/endpoints/recurrence reference:
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Point = Annotated[list[Value], Field(min_length=1, max_length=8)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Step = Literal["start", "diagonal", "advance_left", "advance_right"]


class Input(InputModel):
    left: list[Point] = Field(min_length=1, max_length=256)
    right: list[Point] = Field(min_length=1, max_length=256)
    coordinate_weights: list[Weight] | None = Field(default=None, min_length=1, max_length=8)
    index_band: int | None = Field(default=None, strict=True, ge=0, le=255)

    @model_validator(mode="after")
    def aligned_dimensions(self) -> Self:
        dimensions = len(self.left[0])
        if any(len(point) != dimensions for point in (*self.left, *self.right)):
            raise ValueError("all points must have the same dimension")
        if self.coordinate_weights is not None and (
            len(self.coordinate_weights) != dimensions or not any(self.coordinate_weights)
        ):
            raise ValueError("weights must align with coordinates and contain a positive value")
        cells = len(self.left) * len(self.right)
        if cells > 16_384 or cells * dimensions > 65_536:
            raise ValueError("DTW exceeds 16384 grid cells or 65536 grid-coordinate cells")
        return self


class PathCell(OutputModel):
    left_index: int
    right_index: int
    step: Step
    local_squared_cost: float


class Output(OutputModel):
    left_count: int
    right_count: int
    dimension_count: int
    normalized_coordinate_weights: list[float]
    positive_weight_dimensions: int
    index_band: int | None
    status: Literal["aligned", "no_feasible_path"]
    total_squared_cost: float | None
    total_cost_exact_numerator: str | None
    total_cost_exact_denominator: str | None
    cost_per_path_cell: float | None
    path: list[PathCell]
    path_length: int
    diagonal_steps: int
    advance_left_steps: int
    advance_right_steps: int
    permitted_grid_cells: int
    reachable_grid_cells: int
    maximum_path_index_separation: int | None
    objective: Literal["sum_of_normalized_weighted_squared_distances"] = (
        "sum_of_normalized_weighted_squared_distances"
    )
    tie_priority: list[Step] = ["diagonal", "advance_left", "advance_right"]
    endpoint_rule: Literal["both_sequences_first_and_last_indices"] = (
        "both_sequences_first_and_last_indices"
    )
    timing_verified: Literal[False] = False


def _units(values: list[float]) -> tuple[list[int], int]:
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return (
        [numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios],
        places,
    )


def _number(value: Fraction, label: str) -> float:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("DTW output arithmetic exceeds the 16384-bit budget")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} exceeds finite binary64 range") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    rows, columns = len(request.left), len(request.right)
    dimensions = len(request.left[0])
    weights, _ = _units(request.coordinate_weights or [1.0] * dimensions)
    weight_sum = sum(weights)
    normalized_weights = [
        _number(Fraction(weight, weight_sum), "normalized weight") for weight in weights
    ]
    values, places = _units([value for point in (*request.left, *request.right) for value in point])
    points = [values[start : start + dimensions] for start in range(0, len(values), dimensions)]
    left, right = points[:rows], points[rows:]
    denominator = weight_sum << (2 * places)
    best: list[list[int | None]] = [[None] * columns for _ in range(rows)]
    predecessors = [[0] * columns for _ in range(rows)]
    permitted = reachable = 0

    def local_cost(i: int, j: int) -> int:
        return sum(
            weight * (x - y) ** 2 for weight, x, y in zip(weights, left[i], right[j], strict=True)
        )

    for i in range(rows):
        for j in range(columns):
            if request.index_band is not None and abs(i - j) > request.index_band:
                continue
            permitted += 1
            if i == j == 0:
                best[i][j] = local_cost(i, j)
                reachable += 1
                continue
            candidates: list[tuple[int, int]] = []
            for earlier_i, earlier_j, direction in (
                (i - 1, j - 1, 1),
                (i - 1, j, 2),
                (i, j - 1, 3),
            ):
                if earlier_i >= 0 and earlier_j >= 0:
                    prefix = best[earlier_i][earlier_j]
                    if prefix is not None:
                        candidates.append((prefix, direction))
            if candidates:
                prefix, direction = min(candidates)
                best[i][j] = prefix + local_cost(i, j)
                predecessors[i][j] = direction
                reachable += 1
    total_units = best[-1][-1]
    cells: list[tuple[int, int, int]] = []
    if total_units is not None:
        i, j = rows - 1, columns - 1
        while True:
            direction = predecessors[i][j]
            cells.append((i, j, direction))
            if i == j == 0:
                break
            if direction == 1:
                i, j = i - 1, j - 1
            elif direction == 2:
                i -= 1
            elif direction == 3:
                j -= 1
            else:
                raise ValueError("invalid DTW backpointer")
        cells.reverse()
        if sum(local_cost(i, j) for i, j, _ in cells) != total_units:
            raise ValueError("reconstructed DTW path does not match its objective")
    names: tuple[Step, ...] = ("start", "diagonal", "advance_left", "advance_right")
    path = [
        PathCell(
            left_index=i,
            right_index=j,
            step=names[direction],
            local_squared_cost=_number(
                Fraction(local_cost(i, j), denominator), "local squared cost"
            ),
        )
        for i, j, direction in cells
    ]
    total = Fraction(total_units, denominator) if total_units is not None else None
    return Output(
        left_count=rows,
        right_count=columns,
        dimension_count=dimensions,
        normalized_coordinate_weights=normalized_weights,
        positive_weight_dimensions=sum(weight > 0 for weight in weights),
        index_band=request.index_band,
        status="aligned" if total is not None else "no_feasible_path",
        total_squared_cost=_number(total, "total squared cost") if total is not None else None,
        total_cost_exact_numerator=str(total.numerator) if total is not None else None,
        total_cost_exact_denominator=str(total.denominator) if total is not None else None,
        cost_per_path_cell=_number(total / len(path), "cost per path cell")
        if total is not None
        else None,
        path=path,
        path_length=len(path),
        diagonal_steps=sum(direction == 1 for _, _, direction in cells),
        advance_left_steps=sum(direction == 2 for _, _, direction in cells),
        advance_right_steps=sum(direction == 3 for _, _, direction in cells),
        permitted_grid_cells=permitted,
        reachable_grid_cells=reachable,
        maximum_path_index_separation=max((abs(i - j) for i, j, _ in cells), default=None),
    )


OPERATION = Operation(
    id="features.dynamic_time_warping",
    kind="feature",
    description="Align bounded multivariate sequences by exact squared-cost dynamic time warping.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
