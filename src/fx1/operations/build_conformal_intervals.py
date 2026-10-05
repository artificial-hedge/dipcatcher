"""Absolute-residual split-conformal intervals for supplied held-out predictions.

Reference: Angelopoulos and Bates, https://arxiv.org/abs/2107.07511, section 1.1.
The marginal-coverage theorem requires exchangeable calibration/test scores and
a predictor fitted independently of calibration labels. This operation cannot
establish those conditions from numeric arrays, especially for time series.
"""

from __future__ import annotations

from fractions import Fraction
from math import inf, isfinite, nextafter
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class Input(InputModel):
    calibration_predictions: list[Finite] = Field(min_length=1, max_length=10_000)
    calibration_outcomes: list[Finite] = Field(min_length=1, max_length=10_000)
    predictions: list[Finite] = Field(min_length=1, max_length=1000)
    alpha: float = Field(strict=True, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_pairs(self) -> Self:
        if len(self.calibration_predictions) != len(self.calibration_outcomes):
            raise ValueError("calibration predictions and outcomes must have equal lengths")
        return self


class Interval(OutputModel):
    prediction_index: int
    prediction: float
    lower: float | None
    upper: float | None
    status: Literal["finite", "all_real_numbers"]


class Output(OutputModel):
    calibration_count: int
    prediction_count: int
    alpha: float
    order_statistic_rank: int
    radius: float | None
    radius_status: Literal["finite", "positive_infinity"]
    calibration_covered_count: int
    calibration_coverage: float
    calibration_coverage_is_in_sample: Literal[True] = True
    intervals: list[Interval]
    boundary_policy: Literal["inclusive_outward_rounded"] = "inclusive_outward_rounded"
    alpha_rank_arithmetic: Literal["shortest_decimal_spelling"] = "shortest_decimal_spelling"
    assumptions_verified: Literal[False] = False
    coverage_certified: Literal[False] = False
    required_assumptions: list[str]


def _rounded_bound(value: Fraction, *, upward: bool) -> float:
    """Round a rational expression in supplied binary floats outward, or fail."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError("conformal radius or endpoint exceeds floating-point range") from exc
    if not isfinite(rounded):
        raise ValueError("conformal radius or endpoint exceeds floating-point range")
    represented = Fraction.from_float(rounded)
    if (upward and represented < value) or (not upward and represented > value):
        rounded = nextafter(rounded, inf if upward else -inf)
    if not isfinite(rounded):
        raise ValueError("outward-rounded conformal endpoint exceeds floating-point range")
    return rounded


def execute(request: Input, context: OperationContext) -> Output:
    """Use the k-th ordered residual, extending the order statistics with infinity."""
    size = len(request.calibration_outcomes)
    # JSON alpha has a decimal spelling. Use its shortest round-trip spelling
    # explicitly rather than letting binary multiplication cross a ceil boundary.
    rank_value = (size + 1) * (1 - Fraction(str(request.alpha)))
    rank = (rank_value.numerator + rank_value.denominator - 1) // rank_value.denominator
    radius: float | None = None
    covered = size
    intervals: list[Interval] = []
    if rank <= size:
        # Exact differences preserve ordering for nearly equal residuals and
        # opposite-signed extreme inputs. Only emitted bounds need binary floats.
        residuals = sorted(
            abs(Fraction.from_float(outcome) - Fraction.from_float(prediction))
            for outcome, prediction in zip(
                request.calibration_outcomes, request.calibration_predictions, strict=True
            )
        )
        threshold = residuals[rank - 1]
        radius = _rounded_bound(threshold, upward=True)
        rounded_radius = Fraction.from_float(radius)
        covered = sum(residual <= rounded_radius for residual in residuals)
        for index, prediction in enumerate(request.predictions):
            center = Fraction.from_float(prediction)
            intervals.append(
                Interval(
                    prediction_index=index,
                    prediction=prediction,
                    lower=_rounded_bound(center - rounded_radius, upward=False),
                    upper=_rounded_bound(center + rounded_radius, upward=True),
                    status="finite",
                )
            )
    else:
        intervals = [
            Interval(
                prediction_index=index,
                prediction=prediction,
                lower=None,
                upper=None,
                status="all_real_numbers",
            )
            for index, prediction in enumerate(request.predictions)
        ]
    return Output(
        calibration_count=size,
        prediction_count=len(request.predictions),
        alpha=request.alpha,
        order_statistic_rank=rank,
        radius=radius,
        radius_status="finite" if radius is not None else "positive_infinity",
        calibration_covered_count=covered,
        calibration_coverage=covered / size,
        intervals=intervals,
        required_assumptions=[
            "Calibration and future scores are exchangeable conditional on the fitted predictor.",
            "The predictor was fitted without using these calibration labels.",
            "The score definition and requested alpha were not tuned against these residuals.",
        ],
    )


OPERATION = Operation(
    id="skills.build_conformal_intervals",
    kind="skill",
    description=(
        "Construct inclusive split-conformal intervals from held-out absolute residuals using "
        "a finite-sample order statistic. Insufficient calibration size returns all-real sets; "
        "exchangeability and independence assumptions are reported but not verified."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
