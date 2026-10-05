"""Exact-mass sorted transport for one-dimensional weighted empirical measures.

Each side is normalized independently to unit mass. Zero-weight observations
are ignored; equal support locations are coalesced. The monotone transport
merge computes C_p=sum(mass*abs(x-y)^p), and W_p=C_p^(1/p), for p=1 or 2.
No binning, interpolation, statistical significance test, or calibration is
applied. The caller controls sample selection and point-in-time availability.

Masses, support differences and powered costs use exact integer units derived
from binary64 inputs. A shared integer mass denominator prevents subtraction
from losing tiny remaining masses. Only output conversion and the p=2 square
root round. A positive final distance below binary64 range raises. A p=2 cost
may be below that range while its root is representable: its scalar field is
then null and its binary mantissa/exponent remain available. Transport-plan
mass uses the same explicit scaled representation when its scalar underflows.

Reference: Ramdas, Garcia Trillos and Cuturi, proposition 1 (monotone quantile
transport), https://arxiv.org/abs/1509.02237.
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Input(InputModel):
    left_values: list[Value] = Field(min_length=1, max_length=2_048)
    right_values: list[Value] = Field(min_length=1, max_length=2_048)
    left_weights: list[Weight] | None = Field(default=None, min_length=1, max_length=2_048)
    right_weights: list[Weight] | None = Field(default=None, min_length=1, max_length=2_048)
    p: int = Field(default=1, strict=True, ge=1, le=2)
    plan_limit: int = Field(default=32, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        for name, values, weights in (
            ("left", self.left_values, self.left_weights),
            ("right", self.right_values, self.right_weights),
        ):
            if weights is not None:
                if len(weights) != len(values):
                    raise ValueError(f"{name}_weights must align with {name}_values")
                if not any(weights):
                    raise ValueError(f"{name}_weights must contain a positive weight")
        return self


class TransportStep(OutputModel):
    left_value: Value
    right_value: Value
    absolute_distance: Nonnegative
    probability_mass: float | None = Field(gt=0, le=1, allow_inf_nan=False)
    mass_binary_mantissa: float = Field(ge=0.5, lt=1, allow_inf_nan=False)
    mass_binary_exponent: int
    mass_below_binary64: bool


class Output(OutputModel):
    p: Literal[1, 2]
    distance: Nonnegative
    power_cost: Nonnegative | None
    power_cost_status: Literal["zero", "finite", "below_binary64"]
    cost_binary_mantissa: float = Field(ge=0, lt=1, allow_inf_nan=False)
    cost_binary_exponent: int
    left_observation_count: int
    right_observation_count: int
    left_positive_weight_count: int
    right_positive_weight_count: int
    left_support_count: int
    right_support_count: int
    left_weight_sum: float = Field(gt=0, allow_inf_nan=False)
    right_weight_sum: float = Field(gt=0, allow_inf_nan=False)
    transport_step_count: int
    transport_plan: list[TransportStep] = Field(max_length=200)
    omitted_transport_steps: int


def _integer_units(values: list[float]) -> tuple[list[int], int]:
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ], places


def _binary_scale(value: Fraction) -> tuple[float, int]:
    if value == 0:
        return 0.0, 0
    numerator, denominator = value.numerator, value.denominator
    floor_exponent = numerator.bit_length() - denominator.bit_length()
    below = (
        numerator < (denominator << floor_exponent)
        if floor_exponent >= 0
        else (numerator << -floor_exponent) < denominator
    )
    exponent = floor_exponent + 1 - int(below)
    scaled = (
        Fraction(numerator, denominator << exponent)
        if exponent >= 0
        else Fraction(numerator << -exponent, denominator)
    )
    mantissa = float(scaled)
    if mantissa == 1.0:
        return 0.5, exponent + 1
    return mantissa, exponent


def execute(request: Input, context: OperationContext) -> Output:
    """Sort supports and consume exact integer mass with a two-pointer merge."""
    left_weights = (
        request.left_weights
        if request.left_weights is not None
        else [1.0] * len(request.left_values)
    )
    right_weights = (
        request.right_weights
        if request.right_weights is not None
        else [1.0] * len(request.right_values)
    )
    integer_weights, weight_places = _integer_units(left_weights + right_weights)
    left_units = integer_weights[: len(left_weights)]
    right_units = integer_weights[len(left_weights) :]
    left_total, right_total = sum(left_units), sum(right_units)
    supports: list[list[tuple[float, int]]] = []
    for values, weights in ((request.left_values, left_units), (request.right_values, right_units)):
        grouped: dict[float, int] = {}
        for value, weight in zip(values, weights, strict=True):
            if weight:
                grouped[value] = grouped.get(value, 0) + weight
        supports.append(sorted(grouped.items()))
    left, right = supports
    coordinate_units, coordinate_places = _integer_units(
        [value for value, _ in left] + [value for value, _ in right]
    )
    left_coordinates = coordinate_units[: len(left)]
    right_coordinates = coordinate_units[len(left) :]
    mass_denominator = left_total * right_total
    left_index = right_index = step_count = 0
    left_remaining = left[0][1] * right_total
    right_remaining = right[0][1] * left_total
    cost_numerator = 0
    plan: list[TransportStep] = []
    while left_index < len(left) and right_index < len(right):
        mass = min(left_remaining, right_remaining)
        distance_units = abs(left_coordinates[left_index] - right_coordinates[right_index])
        cost_numerator += mass * distance_units**request.p
        step_count += 1
        if len(plan) < request.plan_limit:
            probability = Fraction(mass, mass_denominator)
            scalar_mass = float(probability)
            mantissa, exponent = _binary_scale(probability)
            plan.append(
                TransportStep(
                    left_value=left[left_index][0],
                    right_value=right[right_index][0],
                    absolute_distance=float(Fraction(distance_units, 1 << coordinate_places)),
                    probability_mass=scalar_mass if scalar_mass else None,
                    mass_binary_mantissa=mantissa,
                    mass_binary_exponent=exponent,
                    mass_below_binary64=scalar_mass == 0.0,
                )
            )
        left_remaining -= mass
        right_remaining -= mass
        if left_remaining == 0:
            left_index += 1
            if left_index < len(left):
                left_remaining = left[left_index][1] * right_total
        if right_remaining == 0:
            right_index += 1
            if right_index < len(right):
                right_remaining = right[right_index][1] * left_total

    cost = Fraction(cost_numerator, mass_denominator << (coordinate_places * request.p))
    cost_mantissa, cost_exponent = _binary_scale(cost)
    scalar_cost = float(cost)
    if request.p == 1:
        distance = scalar_cost
    else:
        distance = correctly_rounded_sqrt(cost.numerator, cost.denominator)
    if not isfinite(distance) or (cost > 0 and distance == 0.0):
        raise ValueError("the nonzero Wasserstein distance is outside finite binary64 range")
    status: Literal["zero", "finite", "below_binary64"] = (
        "zero" if cost == 0 else "finite" if scalar_cost else "below_binary64"
    )
    return Output(
        p=1 if request.p == 1 else 2,
        distance=distance,
        power_cost=scalar_cost if status != "below_binary64" else None,
        power_cost_status=status,
        cost_binary_mantissa=cost_mantissa,
        cost_binary_exponent=cost_exponent,
        left_observation_count=len(request.left_values),
        right_observation_count=len(request.right_values),
        left_positive_weight_count=sum(weight > 0 for weight in left_units),
        right_positive_weight_count=sum(weight > 0 for weight in right_units),
        left_support_count=len(left),
        right_support_count=len(right),
        left_weight_sum=float(Fraction(left_total, 1 << weight_places)),
        right_weight_sum=float(Fraction(right_total, 1 << weight_places)),
        transport_step_count=step_count,
        transport_plan=plan,
        omitted_transport_steps=step_count - len(plan),
    )


OPERATION = Operation(
    id="features.empirical_wasserstein",
    kind="feature",
    description=(
        "Compute weighted one-dimensional Wasserstein-1 or Wasserstein-2 distance through "
        "an exact-mass monotone transport merge, independently normalized supports and a "
        "bounded transport-plan prefix. Preserves tiny costs using explicit binary scaling."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
