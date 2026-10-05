"""Evaluate the characteristic function of an explicitly weighted empirical law.

At each supplied angular frequency t, return sum_j p_j exp(i*t*x_j), with
p_j=w_j/sum(w), nonnegative weights and at least one positive weight. Frequencies
retain their input order and may repeat. Zero-weight observations are ignored.
An effective weight count 1/sum(p_j**2) describes concentration, not independence.
The caller must select observations available at the intended decision time.

Phase products use exact binary rationals before rounding and must have absolute
value at most 1e6 radians. This explicit safety limit avoids evaluating enormous
arguments; it is not a rigorous transcendental error bound. Nonzero phases that
round to zero are rejected. Platform math.sin/cos evaluate the rounded phase;
their results are accumulated with exact integer weighted sums, then rounded.
A nonzero normalized weight or component that rounds to zero is rejected.
The returned modulus is the correctly rounded norm of the returned components;
it can exceed one by floating point error, which is reported without clamping.
No distribution fit, goodness-of-fit test, or inference is performed.

The unweighted ECF definition is given by Zeng and Zimmerman, section 2:
https://arxiv.org/html/2504.07946v2
Here equal mass 1/n is explicitly generalized to supplied empirical masses p_j.
"""

from fractions import Fraction
from math import cos, isfinite, sin
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=2_048)
    frequencies: list[Value] = Field(min_length=1, max_length=128)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=2_048)

    @model_validator(mode="after")
    def bound_work(self) -> Self:
        if len(self.values) * len(self.frequencies) > 131_072:
            raise ValueError("at most 131072 observation-frequency pairs are supported")
        if self.weights is not None:
            if len(self.weights) != len(self.values):
                raise ValueError("weights must align with values")
            if not any(self.weights):
                raise ValueError("at least one weight must be positive")
        return self


class Evaluation(OutputModel):
    frequency_index: int
    frequency: float = Field(allow_inf_nan=False)
    real: float = Field(ge=-1, le=1, allow_inf_nan=False)
    imaginary: float = Field(ge=-1, le=1, allow_inf_nan=False)
    modulus: float = Field(ge=0, le=1.000000000001, allow_inf_nan=False)
    modulus_above_one_from_rounding: bool
    maximum_absolute_evaluated_phase: float = Field(ge=0, le=1e6, allow_inf_nan=False)


class Output(OutputModel):
    observation_count: int
    positive_weight_count: int
    evaluated_phase_count: int
    weight_sum: float = Field(gt=0, allow_inf_nan=False)
    normalized_weights: list[Probability] = Field(min_length=1, max_length=2_048)
    effective_weight_count: float = Field(ge=1, le=2_048, allow_inf_nan=False)
    phase_limit_radians: Literal[1_000_000] = 1_000_000
    frequency_units: Literal["radians_per_value_unit"] = "radians_per_value_unit"
    evaluations: list[Evaluation] = Field(min_length=1, max_length=128)


def _units(values: list[float]) -> tuple[list[int], int]:
    ratios = [value.as_integer_ratio() for value in values]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    return [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ], places


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"nonzero {label} cannot be represented as finite binary64")
    return result


def _weighted_mean(values: list[float], weights: list[int], total: int, label: str) -> float:
    units, places = _units(values)
    numerator = sum(value * weight for value, weight in zip(units, weights, strict=True))
    return _number(Fraction(numerator, total << places), label)


def execute(request: Input, context: OperationContext) -> Output:
    """Evaluate bounded phases and exactly accumulate their weighted trig components."""
    supplied_weights = (
        request.weights if request.weights is not None else [1.0] * len(request.values)
    )
    weights, places = _units(supplied_weights)
    total = sum(weights)
    active = [index for index, weight in enumerate(weights) if weight]
    active_weights = [weights[index] for index in active]
    values = [Fraction(request.values[index]) for index in active]
    probabilities = [_number(Fraction(weight, total), "normalized weight") for weight in weights]
    evaluations: list[Evaluation] = []
    for frequency_index, frequency in enumerate(request.frequencies):
        rational_frequency = Fraction(frequency)
        real_terms: list[float] = []
        imaginary_terms: list[float] = []
        max_phase = 0.0
        for source_index, value in zip(active, values, strict=True):
            exact_phase = value * rational_frequency
            if abs(exact_phase) > 1_000_000:
                raise ValueError(
                    f"phase exceeds 1e6 radians at value {source_index}, frequency {frequency_index}"
                )
            phase = _number(exact_phase, "phase")
            real_terms.append(cos(phase))
            imaginary_terms.append(sin(phase))
            max_phase = max(max_phase, abs(phase))
        real = _weighted_mean(real_terms, active_weights, total, "real component")
        imaginary = _weighted_mean(imaginary_terms, active_weights, total, "imaginary component")
        squared_norm = Fraction(real) ** 2 + Fraction(imaginary) ** 2
        modulus = correctly_rounded_sqrt(squared_norm.numerator, squared_norm.denominator)
        evaluations.append(
            Evaluation(
                frequency_index=frequency_index,
                frequency=frequency,
                real=real,
                imaginary=imaginary,
                modulus=modulus,
                modulus_above_one_from_rounding=modulus > 1,
                maximum_absolute_evaluated_phase=max_phase,
            )
        )
    return Output(
        observation_count=len(weights),
        positive_weight_count=len(active),
        evaluated_phase_count=len(active) * len(request.frequencies),
        weight_sum=_number(Fraction(total, 1 << places), "weight sum"),
        normalized_weights=probabilities,
        effective_weight_count=_number(
            Fraction(total * total, sum(w * w for w in weights)), "effective weight count"
        ),
        evaluations=evaluations,
    )


OPERATION = Operation(
    id="features.empirical_characteristic_function",
    kind="feature",
    description=(
        "Evaluate a weighted empirical characteristic function at explicit angular frequencies "
        "with exact phase-product checks, bounded trig evaluation and exact weighted accumulation."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
