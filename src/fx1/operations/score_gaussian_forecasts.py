"""Multivariate Gaussian negative log density from exact covariance factorization.

For d coordinates, loss=(d*log(2*pi)+log(det(Sigma))+
(y-mu)'*inverse(Sigma)*(y-mu))/2. Lower is better; losses may be negative and
depend on coordinate units. A covariance must be exactly symmetric and positive
definite for the supplied binary-float entries. No tolerance, jitter, covariance
repair, matrix inverse, or fitted parameter estimation is used.

An independently implemented unpivoted LDL' factorization over Fractions checks
positive pivots and solves the triangular system. Mahalanobis accumulation and
the determinant product are exact; logarithms are floating approximations. Very
small nonzero reported values that cannot be represented cause an explicit
error. Proper density scoring assumes supplied forecasts precede verification.
Density reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
"""

from __future__ import annotations

from fractions import Fraction
from math import isfinite, log, log1p, pi
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Vector = Annotated[list[Value], Field(min_length=1, max_length=12)]


class Forecast(InputModel):
    mean: Vector
    covariance: list[Vector] = Field(min_length=1, max_length=12)


class Input(InputModel):
    outcomes: list[Vector] = Field(min_length=1, max_length=128)
    forecasts: list[Forecast] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def common_dimensions(self) -> Self:
        if len(self.outcomes) != len(self.forecasts):
            raise ValueError("each outcome must have one Gaussian forecast")
        size = len(self.outcomes[0])
        if len(self.outcomes) * size**3 > 100_000:
            raise ValueError("Gaussian scoring exceeds 100000 cubic dimension-work units")
        for outcome, forecast in zip(self.outcomes, self.forecasts, strict=True):
            if (
                len(outcome) != size
                or len(forecast.mean) != size
                or len(forecast.covariance) != size
                or any(len(row) != size for row in forecast.covariance)
            ):
                raise ValueError("all vectors and covariance matrices must share one dimension")
            if any(
                forecast.covariance[i][j] != forecast.covariance[j][i]
                for i in range(size)
                for j in range(i)
            ):
                raise ValueError("covariance matrices must be exactly symmetric")
        return self


class Row(OutputModel):
    source_index: int
    negative_log_density: float
    log_covariance_determinant: float
    squared_mahalanobis_distance: float
    minimum_ldl_pivot: float
    maximum_ldl_pivot: float


class Output(OutputModel):
    observation_count: int
    dimension_count: int
    mean_negative_log_density: float
    scores: list[Row]
    logarithm_base: Literal["natural"] = "natural"
    covariance_validation: Literal["exact_symmetry_and_positive_rational_ldl_pivots"] = (
        "exact_symmetry_and_positive_rational_ldl_pivots"
    )
    covariance_jitter_applied: Literal[False] = False
    covariance_fitted: Literal[False] = False
    forecast_timing_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds finite floating-point range") from exc
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"{label} is outside representable finite range")
    return result


def _log_fraction(value: Fraction) -> float:
    deviation = value - 1
    if abs(deviation) <= Fraction(1, 2):
        result = log1p(_number(deviation, "log-determinant deviation"))
    else:
        exponent = value.numerator.bit_length() - value.denominator.bit_length()
        scaled = (
            Fraction(value.numerator, value.denominator << exponent)
            if exponent >= 0
            else Fraction(value.numerator << -exponent, value.denominator)
        )
        result = log(float(scaled)) + exponent * log(2)
    if not isfinite(result) or (value != 1 and result == 0):
        raise ValueError("nonzero log determinant is outside supported precision")
    return result


def execute(request: Input, context: OperationContext) -> Output:
    size = len(request.outcomes[0])
    constant = Fraction(size * log(2 * pi))
    rows: list[Row] = []
    losses: list[Fraction] = []
    for row_index, (outcome, forecast) in enumerate(
        zip(request.outcomes, request.forecasts, strict=True)
    ):
        covariance = [[Fraction(value) for value in row] for row in forecast.covariance]
        lower = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
        pivots: list[Fraction] = []
        determinant = Fraction(1)
        for column in range(size):
            pivot = covariance[column][column] - sum(
                (lower[column][k] ** 2 * pivots[k] for k in range(column)), Fraction()
            )
            if pivot <= 0:
                raise ValueError(
                    f"forecast {row_index} covariance is not positive definite at pivot {column}"
                )
            pivots.append(pivot)
            determinant *= pivot
            for row in range(column + 1, size):
                previous = sum(
                    (lower[row][k] * lower[column][k] * pivots[k] for k in range(column)),
                    Fraction(),
                )
                lower[row][column] = (covariance[row][column] - previous) / pivot
        transformed: list[Fraction] = []
        for row in range(size):
            residual = Fraction(outcome[row]) - Fraction(forecast.mean[row])
            residual -= sum(
                (lower[row][column] * transformed[column] for column in range(row)), Fraction()
            )
            transformed.append(residual)
        mahalanobis = sum(
            (value * value / pivot for value, pivot in zip(transformed, pivots, strict=True)),
            Fraction(),
        )
        log_determinant = _log_fraction(determinant)
        loss = (constant + Fraction(log_determinant) + mahalanobis) / 2
        losses.append(loss)
        rows.append(
            Row(
                source_index=row_index,
                negative_log_density=_number(loss, "negative log density"),
                log_covariance_determinant=log_determinant,
                squared_mahalanobis_distance=_number(mahalanobis, "squared Mahalanobis distance"),
                minimum_ldl_pivot=_number(min(pivots), "minimum LDL pivot"),
                maximum_ldl_pivot=_number(max(pivots), "maximum LDL pivot"),
            )
        )
    return Output(
        observation_count=len(rows),
        dimension_count=size,
        mean_negative_log_density=_number(sum(losses, Fraction()) / len(rows), "mean log loss"),
        scores=rows,
    )


OPERATION = Operation(
    id="skills.score_gaussian_forecasts",
    kind="skill",
    description=(
        "Score multivariate Gaussian forecasts with exact positive-definite LDL factorization "
        "and Mahalanobis accumulation, approximate log determinants, and explicit singularity "
        "or numerical-range errors. Covariances are supplied and never repaired or fitted."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
