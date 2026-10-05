"""Burg forward/backward lattice estimation of an ordered real-valued block.

The optional demean mode subtracts the exact binary-rational sample mean.
Start forward and backward errors at the centered series. At stage m pair
forward[1:] with backward[:-1], and set k=-2*sum(f*b)/sum(f*f+b*b). Update
f'=f+k*b, b'=b+k*f and a'_j=a_j+k*a_(m-j), with a'_m=k and a_0=1.
Returned AR coefficients are phi_j=-a_j, so the fitted convention is
x_t=intercept+sum(phi_j*x_(t-j))+error, intercept=mean*(1-sum(phi)).

Errors are divided by a common maximum magnitude initially and after each
stage. Each ratio, dot product and update is computed exactly from the current
binary64 errors before normalization/rounding. Reflection updates use the exact
ratio, not its returned rounded value. This bounds integer sizes and preserves
small differences within the current lattice; intermediate lattice values and
polynomial coefficients still round between stages. Nonzero quantities that
round to zero are rejected. A nonunit reflection that rounds to +/-1 stops
with numerical_boundary instead of being labeled perfect prediction.

Innovation variance follows E_0=mean(centered_x^2), E_m=E_(m-1)*(1-k_m^2),
with exact rational accumulation, not degrees-of-freedom correction or final
forward-residual MSE. Zero initial energy, singular paired errors, or exact
unit reflection in the current rounded lattice stop explicitly. The latter
does not certify exact prediction of the original unrounded data.
This is a whole-block fit. The caller must choose ordered, available samples;
no forecasting, stationarity or statistical model-validity claim is supplied.

Real Burg/lattice and coefficient conventions, equations(5)-(8):
https://link.springer.com/article/10.1007/s00779-024-01806-8
Innovation-product convention in Spectrum's public arburg implementation:
https://pyspectrum.readthedocs.io/en/latest/_modules/spectrum/burg.html
"""

from fractions import Fraction
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Centering = Literal["demean", "none"]
Status = Literal[
    "fitted",
    "zero_energy_input",
    "singular_lattice",
    "perfect_prediction_in_lattice",
    "numerical_boundary",
]


class Input(InputModel):
    values: list[Value] = Field(min_length=2, max_length=4_096)
    order: int = Field(default=1, strict=True, ge=0, le=64)
    centering: Centering = "demean"

    @model_validator(mode="after")
    def bound_work(self) -> Self:
        if self.order >= len(self.values):
            raise ValueError("order must be smaller than the number of observations")
        if self.order * len(self.values) > 131_072:
            raise ValueError("at most 131072 observation-order cells are supported")
        return self


class Stage(OutputModel):
    order: int
    paired_error_count: int
    reflection_coefficient: float = Field(ge=-1, le=1, allow_inf_nan=False)
    innovation_variance: float = Field(ge=0, allow_inf_nan=False)


class Output(OutputModel):
    observation_count: int
    requested_order: int
    fitted_order: int
    centering: Centering
    centering_offset: Finite
    constant_input: bool
    ar_coefficients: list[Finite] = Field(max_length=64)
    polynomial_coefficients: list[Finite] = Field(min_length=1, max_length=65)
    intercept: Finite
    initial_innovation_variance: float = Field(ge=0, allow_inf_nan=False)
    innovation_variance: float = Field(ge=0, allow_inf_nan=False)
    fit_status: Status
    stopped_at_order: int | None
    stopping_reflection_coefficient: Finite | None
    stages: list[Stage] = Field(max_length=64)
    innovation_convention: Literal["initial_mse_times_reflection_factors"] = (
        "initial_mse_times_reflection_factors"
    )
    whole_block_fit: Literal[True] = True
    model_validity_established: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value != 0 and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _normalize(values: list[Fraction], scale: Fraction) -> list[float]:
    return [_number(value / scale, "normalized lattice error") for value in values]


def _reflection(forward: list[float], backward: list[float]) -> Fraction | None:
    # A shared power-of-two denominator cancels from the reflection ratio.
    ratios = [value.as_integer_ratio() for value in (*forward, *backward)]
    places = max(denominator.bit_length() - 1 for _, denominator in ratios)
    units = [
        numerator << (places - denominator.bit_length() + 1) for numerator, denominator in ratios
    ]
    count = len(forward)
    left, right = units[:count], units[count:]
    denominator = sum(value * value for value in units)
    if not denominator:
        return None
    numerator = -2 * sum(f * b for f, b in zip(left, right, strict=True))
    return Fraction(numerator, denominator)


def execute(request: Input, context: OperationContext) -> Output:
    """Run independently implemented Burg recursion with bounded normalized errors."""
    size = len(request.values)
    exact = [Fraction(value) for value in request.values]
    mean = sum(exact, Fraction()) / size if request.centering == "demean" else Fraction()
    centered = [value - mean for value in exact]
    energy = sum((value * value for value in centered), Fraction()) / size
    initial_energy = _number(energy, "initial innovation variance")
    coefficients = [1.0]
    stages: list[Stage] = []
    status: Status = "fitted" if energy else "zero_energy_input"
    stopped_at = None
    stopping_reflection = None
    if energy and request.order:
        scale = max(map(abs, centered))
        forward = _normalize(centered, scale)
        backward = forward.copy()
        for order in range(1, request.order + 1):
            paired_forward, paired_backward = forward[1:], backward[:-1]
            reflection = _reflection(paired_forward, paired_backward)
            if reflection is None:
                status, stopped_at = "singular_lattice", order
                break
            if abs(reflection) > 1:
                raise ValueError("reflection magnitude exceeds its exact mathematical bound")
            rounded_reflection = _number(reflection, "reflection coefficient")
            if abs(reflection) < 1 and abs(rounded_reflection) == 1:
                status, stopped_at = "numerical_boundary", order
                stopping_reflection = rounded_reflection
                break
            previous = [Fraction(value) for value in coefficients]
            coefficients = (
                [1.0]
                + [
                    _number(
                        previous[index] + reflection * previous[order - index],
                        "polynomial coefficient",
                    )
                    for index in range(1, order)
                ]
                + [rounded_reflection]
            )
            energy *= 1 - reflection * reflection
            # With binary64 normalized errors, at most 64 stages need fewer
            # than 300000-bit numerator/denominator integers. Enforce the cap.
            if max(energy.numerator.bit_length(), energy.denominator.bit_length()) > 300_000:
                raise ValueError("innovation arithmetic exceeds the 300000-bit budget")
            stages.append(
                Stage(
                    order=order,
                    paired_error_count=size - order,
                    reflection_coefficient=rounded_reflection,
                    innovation_variance=_number(energy, "innovation variance"),
                )
            )
            if abs(reflection) == 1:
                status, stopped_at = "perfect_prediction_in_lattice", order
                stopping_reflection = rounded_reflection
                break
            if order < request.order:
                exact_forward = [Fraction(value) for value in paired_forward]
                exact_backward = [Fraction(value) for value in paired_backward]
                new_forward = [
                    f + reflection * b for f, b in zip(exact_forward, exact_backward, strict=True)
                ]
                new_backward = [
                    b + reflection * f for f, b in zip(exact_forward, exact_backward, strict=True)
                ]
                scale = max(max(map(abs, new_forward)), max(map(abs, new_backward)))
                if not scale:
                    status, stopped_at = "singular_lattice", order + 1
                    break
                forward, backward = _normalize(new_forward, scale), _normalize(new_backward, scale)
    intercept = mean * sum((Fraction(value) for value in coefficients), Fraction())
    return Output(
        observation_count=size,
        requested_order=request.order,
        fitted_order=len(coefficients) - 1,
        centering=request.centering,
        centering_offset=_number(mean, "centering offset"),
        constant_input=len(set(request.values)) == 1,
        ar_coefficients=[-value for value in coefficients[1:]],
        polynomial_coefficients=coefficients,
        intercept=_number(intercept, "intercept"),
        initial_innovation_variance=initial_energy,
        innovation_variance=_number(energy, "innovation variance"),
        fit_status=status,
        stopped_at_order=stopped_at,
        stopping_reflection_coefficient=stopping_reflection,
        stages=stages,
    )


OPERATION = Operation(
    id="features.burg_autoregression",
    kind="feature",
    description=(
        "Fit a bounded Burg forward/backward lattice with explicit centering, coefficient signs, "
        "reflection and innovation diagnostics, and singular/perfect-prediction stopping statuses."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
