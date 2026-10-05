"""Weighted directional moments and the shortest arc containing positive mass.

Angles are reduced exactly modulo 360 degrees or the binary64 constant tau
in radians. Trigonometric evaluation remains approximate; exact quarter turns
use exact unit vectors. Weights normalize to probabilities; zero weights are
excluded from both moments and the covering arc. The center is undefined when
the resultant length is at most the supplied degeneracy tolerance.

For small spreads, compute q=1-R^2 as 4*sum(i<j,p_i*p_j*sin(delta_ij/2)^2),
then variance=q/(1+R), avoiding subtraction of a rounded R from one. R itself
comes from the mean vector, avoiding loss near zero when q is close to one.
These two trigonometric calculations agree only up to floating roundoff. Angular
deviation is sqrt(2*variance), converted to the requested angle units; this is
not the logarithmic circular standard deviation. A q just above one within
64 binary64 epsilons is clipped and disclosed. Larger violations fail.

The closed shortest covering arc is the complement of the largest empty gap
between distinct positive-mass angles. Equal largest gaps select the lowest
normalized arc start. Exact rational endpoint/width fields define that arc;
their float previews can lose precision, including rounding across the seam
to zero. Coverage applies to the exact fields, not the previews. It is a
support summary, not a confidence interval.
Caller must enforce availability and sample selection. Moment conventions:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html
"""

from fractions import Fraction
from math import atan2, cos, isfinite, pi, sin, tau
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Angle = Annotated[float, Field(strict=True, ge=-1e12, le=1e12)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
Unit = Literal["degrees", "radians"]


class Input(InputModel):
    angles: list[Angle] = Field(min_length=1, max_length=256)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=256)
    unit: Unit = "radians"
    degeneracy_tolerance: float = Field(default=1e-12, strict=True, ge=0, le=1e-3)

    @model_validator(mode="after")
    def aligned_weights(self) -> Self:
        if self.weights is not None and (
            len(self.weights) != len(self.angles) or not any(self.weights)
        ):
            raise ValueError("weights must align with angles and have positive total mass")
        return self


class Output(OutputModel):
    observation_count: int
    positive_weight_count: int
    distinct_positive_angles: int
    unit: Unit
    period: float
    weight_sum: float
    normalized_weights: list[float]
    mean_cosine: float
    mean_sine: float
    resultant_length: float
    circular_variance: float
    angular_deviation: float
    mean_direction: float | None
    center_status: Literal["defined", "resultant_at_or_below_tolerance"]
    degeneracy_tolerance: float
    covering_arc_start: float
    covering_arc_end: float
    covering_arc_width: float
    covering_arc_start_numerator: str
    covering_arc_start_denominator: str
    covering_arc_end_numerator: str
    covering_arc_end_denominator: str
    covering_arc_width_numerator: str
    covering_arc_width_denominator: str
    covering_arc_float_endpoints_exact: bool
    covering_arc_wraps: bool
    covering_arc_start_source_index: int
    covering_arc_end_source_index: int
    equally_short_covering_arcs: int
    pair_count: int
    resultant_roundoff_clipped: bool
    arc_contains: Literal["all_positive_weight_angles_using_exact_closed_endpoints"] = (
        "all_positive_weight_angles_using_exact_closed_endpoints"
    )
    angular_deviation_definition: Literal["sqrt_2_times_circular_variance_in_requested_units"] = (
        "sqrt_2_times_circular_variance_in_requested_units"
    )
    timing_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _root(value: Fraction, label: str) -> float:
    result = correctly_rounded_sqrt(value.numerator, value.denominator)
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _direction(turn: Fraction) -> tuple[float, float]:
    quarters = turn * 4
    if quarters.denominator == 1:
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quarters.numerator % 4]
    # Evaluate around zero so directions just below a full turn retain their small angle.
    reduced = turn if turn <= Fraction(1, 2) else turn - 1
    phase = _number(reduced * Fraction(tau), "trigonometric phase")
    return cos(phase), sin(phase)


def execute(request: Input, context: OperationContext) -> Output:
    period = Fraction(360 if request.unit == "degrees" else tau)
    raw_weights = request.weights if request.weights is not None else [1.0] * len(request.angles)
    weights = [Fraction(weight) for weight in raw_weights]
    total = sum(weights, Fraction())
    active = [index for index, weight in enumerate(weights) if weight]
    angles = [Fraction(angle) % period for angle in request.angles]
    directions = {index: _direction(angles[index] / period) for index in active}
    cosine = sum((weights[i] * Fraction(directions[i][0]) for i in active), Fraction()) / total
    sine = sum((weights[i] * Fraction(directions[i][1]) for i in active), Fraction()) / total
    deficit_sum = Fraction()
    for offset, left in enumerate(active):
        for right in active[:offset]:
            delta = abs(angles[left] - angles[right])
            delta = min(delta, period - delta)
            if delta == period / 2:
                half_sine = 1.0
            elif not delta:
                half_sine = 0.0
            else:
                half_sine = sin(_number(delta * Fraction(pi) / period, "half-angle phase"))
            deficit_sum += weights[left] * weights[right] * Fraction(half_sine) ** 2
    deficit = 4 * deficit_sum / (total * total)
    clipped = deficit > 1
    if deficit > 1 + Fraction(64, 2**52):
        raise ValueError("pairwise resultant deficit exceeds its mathematical bound")
    deficit = min(deficit, Fraction(1))
    resultant = _root(cosine * cosine + sine * sine, "resultant length")
    if resultant > 1 + 64 / 2**52:
        raise ValueError("mean resultant exceeds its mathematical bound")
    clipped = clipped or resultant > 1
    resultant = min(resultant, 1.0)
    variance = deficit / (1 + Fraction(resultant))
    angular_deviation = _root(2 * variance, "angular deviation")
    mean_cosine = _number(cosine, "mean cosine")
    mean_sine = _number(sine, "mean sine")
    direction: float | None = None
    if resultant > request.degeneracy_tolerance:
        if not mean_cosine and not mean_sine:
            raise ValueError("directional components vanished despite a nondegenerate resultant")
        phase = Fraction(atan2(mean_sine, mean_cosine))
        direction = _number((phase / Fraction(tau) * period) % period, "mean direction")
        if direction == float(period):
            direction = 0.0
    unique: dict[Fraction, int] = {}
    for index in active:
        unique.setdefault(angles[index], index)
    ordered = sorted(unique)
    gaps = [
        (ordered[(index + 1) % len(ordered)] - angle) % period
        for index, angle in enumerate(ordered)
    ]
    if len(ordered) == 1:
        gaps[0] = period
    largest_gap = max(gaps)
    candidates = [index for index, gap in enumerate(gaps) if gap == largest_gap]
    selected = min(candidates, key=lambda index: ordered[(index + 1) % len(ordered)])
    start, end = ordered[(selected + 1) % len(ordered)], ordered[selected]
    width = period - largest_gap
    start_float = _number(start, "arc start")
    end_float = _number(end, "arc end")
    start_float = 0.0 if start_float == float(period) else start_float
    end_float = 0.0 if end_float == float(period) else end_float
    return Output(
        observation_count=len(request.angles),
        positive_weight_count=len(active),
        distinct_positive_angles=len(unique),
        unit=request.unit,
        period=float(period),
        weight_sum=_number(total, "total weight"),
        normalized_weights=[_number(weight / total, "normalized weight") for weight in weights],
        mean_cosine=mean_cosine,
        mean_sine=mean_sine,
        resultant_length=resultant,
        circular_variance=_number(variance, "circular variance"),
        angular_deviation=_number(
            Fraction(angular_deviation) * period / Fraction(tau), "angular deviation in units"
        ),
        mean_direction=direction,
        center_status="defined" if direction is not None else "resultant_at_or_below_tolerance",
        degeneracy_tolerance=request.degeneracy_tolerance,
        covering_arc_start=start_float,
        covering_arc_end=end_float,
        covering_arc_width=_number(width, "arc width"),
        covering_arc_start_numerator=str(start.numerator),
        covering_arc_start_denominator=str(start.denominator),
        covering_arc_end_numerator=str(end.numerator),
        covering_arc_end_denominator=str(end.denominator),
        covering_arc_width_numerator=str(width.numerator),
        covering_arc_width_denominator=str(width.denominator),
        covering_arc_float_endpoints_exact=(
            Fraction(start_float) == start and Fraction(end_float) == end
        ),
        covering_arc_wraps=end < start,
        covering_arc_start_source_index=unique[start],
        covering_arc_end_source_index=unique[end],
        equally_short_covering_arcs=len(candidates),
        pair_count=len(active) * (len(active) - 1) // 2,
        resultant_roundoff_clipped=clipped,
    )


OPERATION = Operation(
    id="features.circular_summary",
    kind="feature",
    description=(
        "Summarize weighted circular direction and stable pairwise dispersion, with an explicit "
        "unit, degeneracy tolerance and shortest closed arc containing positive-weight angles."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
