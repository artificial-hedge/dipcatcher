"""Unweighted biased empirical Euclidean distance covariance and correlation.

For paired vector rows, A_ij=|x_i-x_j|-row_mean_i-row_mean_j+grand_mean;
B is defined similarly. Include diagonal entries after double centering and
divide all matrix inner products by n^2: Vxy2=mean(A*B), Vxx2=mean(A*A),
Vyy2=mean(B*B). The returned correlation is
(Vxy2^2/(Vxx2*Vyy2))^(1/4), undefined when either marginal distance variance
is zero. This is the biased V-statistic, not U-centering or an independence
test. In particular two nonconstant paired observations give correlation one.

Pairwise norms use exact differences and correctly rounded square roots.
Their rounded distances are converted to common integer units; centering and
inner products are then exact. A negative cross product from the approximate
distance matrices fails instead of being silently clipped. A direct integer
fourth-root rounding step avoids underflow in squared-correlation intermediates.
Unrepresentable positive physical moments retain zero with an explicit underflow
flag. No matrix is emitted, and sample/availability assumptions remain unchecked.
Formula reference: Szekely, Rizzo and Bakirov (2007), definitions 4 and 5:
https://arxiv.org/abs/0803.4101
"""

from fractions import Fraction
from math import isfinite, isqrt, ldexp
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Row = Annotated[list[Value], Field(min_length=1, max_length=8)]


class Input(InputModel):
    x: list[Row] = Field(min_length=2, max_length=128)
    y: list[Row] = Field(min_length=2, max_length=128)

    @model_validator(mode="after")
    def aligned_vectors(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError("x and y must contain paired rows")
        for rows in (self.x, self.y):
            if any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("each marginal must have a fixed vector dimension")
        return self


class Magnitude(OutputModel):
    value: float
    positive_underflow: bool


class Output(OutputModel):
    observation_count: int
    x_dimension: int
    y_dimension: int
    unordered_pair_count: int
    distance_covariance: Magnitude
    squared_distance_covariance: Magnitude
    squared_distance_variance_x: Magnitude
    squared_distance_variance_y: Magnitude
    distance_correlation: float | None
    correlation_status: Literal["defined", "zero_marginal_distance_variance", "positive_underflow"]
    x_constant: bool
    y_constant: bool
    maximum_distance_x: float
    maximum_distance_y: float
    exact_centered_row_sums_zero: bool
    estimator: Literal["unweighted_biased_double_centered_v_statistic"] = (
        "unweighted_biased_double_centered_v_statistic"
    )
    independence_test_performed: Literal[False] = False
    timing_verified: Literal[False] = False


def _distances(rows: list[list[float]]) -> tuple[list[list[int]], int, float]:
    count = len(rows)
    exact = [[Fraction(value) for value in row] for row in rows]
    matrix = [[0.0] * count for _ in rows]
    for i in range(count):
        for j in range(i):
            squared = sum(
                ((left - right) ** 2 for left, right in zip(exact[i], exact[j], strict=True)),
                Fraction(),
            )
            distance = correctly_rounded_sqrt(squared.numerator, squared.denominator)
            if not isfinite(distance) or (squared and distance == 0):
                raise ValueError("pairwise distance is outside finite binary64 range")
            matrix[i][j] = matrix[j][i] = distance
    ratios = [[value.as_integer_ratio() for value in row] for row in matrix]
    places = max(denominator.bit_length() - 1 for row in ratios for _, denominator in row)
    units = [
        [numerator << (places - denominator.bit_length() + 1) for numerator, denominator in row]
        for row in ratios
    ]
    return units, places, max(max(row) for row in matrix)


def _center(matrix: list[list[int]]) -> list[list[int]]:
    count = len(matrix)
    row_sums = [sum(row) for row in matrix]
    total = sum(row_sums)
    return [
        [
            count * count * value - count * (row_sums[i] + row_sums[j]) + total
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(matrix)
    ]


def _magnitude(value: Fraction, *, square_root: bool = False) -> Magnitude:
    try:
        result = (
            correctly_rounded_sqrt(value.numerator, value.denominator)
            if square_root
            else float(value)
        )
    except OverflowError as error:
        raise ValueError("distance moment exceeds binary64 range") from error
    if not isfinite(result):
        raise ValueError("distance moment exceeds binary64 range")
    return Magnitude(value=result, positive_underflow=bool(value and result == 0))


def _fourth_root(numerator: int, denominator: int) -> float:
    """Round the positive fourth root directly, using exact midpoint comparisons."""
    if not numerator:
        return 0.0
    if numerator < 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("squared matrix cosine must lie in [0, 1]")
    if max(numerator.bit_length(), denominator.bit_length()) > 16_384:
        raise ValueError("distance-correlation ratio exceeds 16384-bit budget")
    exponent = numerator.bit_length() - denominator.bit_length()
    if (
        numerator < denominator << exponent
        if exponent >= 0
        else numerator << -exponent < denominator
    ):
        exponent -= 1
    unit_exponent = max(exponent // 4 - 52, -1074)
    scaled_numerator = numerator << (-4 * unit_exponent)
    root = isqrt(isqrt(scaled_numerator // denominator))
    midpoint_left = 16 * scaled_numerator
    midpoint_right = denominator * (2 * root + 1) ** 4
    if midpoint_left > midpoint_right or (midpoint_left == midpoint_right and root % 2):
        root += 1
    return ldexp(float(root), unit_exponent)


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.x)
    x_matrix, x_places, x_maximum = _distances(request.x)
    y_matrix, y_places, y_maximum = _distances(request.y)
    x_centered, y_centered = _center(x_matrix), _center(y_matrix)
    cross = x_squares = y_squares = 0
    for x_row, y_row in zip(x_centered, y_centered, strict=True):
        for left, right in zip(x_row, y_row, strict=True):
            cross += left * right
            x_squares += left * left
            y_squares += right * right
    if cross < 0:
        raise ValueError("rounded Euclidean distances produced a negative distance covariance")
    common = count**6
    covariance = Fraction(cross, common << (x_places + y_places))
    correlation: float | None = None
    status: Literal["defined", "zero_marginal_distance_variance", "positive_underflow"] = (
        "zero_marginal_distance_variance"
    )
    if x_squares and y_squares:
        correlation = _fourth_root(cross * cross, x_squares * y_squares)
        status = "positive_underflow" if cross and correlation == 0 else "defined"
    return Output(
        observation_count=count,
        x_dimension=len(request.x[0]),
        y_dimension=len(request.y[0]),
        unordered_pair_count=count * (count - 1) // 2,
        distance_covariance=_magnitude(covariance, square_root=True),
        squared_distance_covariance=_magnitude(covariance),
        squared_distance_variance_x=_magnitude(Fraction(x_squares, common << (2 * x_places))),
        squared_distance_variance_y=_magnitude(Fraction(y_squares, common << (2 * y_places))),
        distance_correlation=correlation,
        correlation_status=status,
        x_constant=not x_squares,
        y_constant=not y_squares,
        maximum_distance_x=x_maximum,
        maximum_distance_y=y_maximum,
        exact_centered_row_sums_zero=all(sum(row) == 0 for row in (*x_centered, *y_centered)),
    )


OPERATION = Operation(
    id="features.distance_correlation",
    kind="feature",
    description=(
        "Compute biased multivariate Euclidean distance covariance and correlation with exact "
        "double centering of rounded distances, explicit constant-input and underflow diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
