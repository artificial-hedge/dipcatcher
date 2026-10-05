"""Orthonormal Haar analysis with a matched inverse and explicit boundaries.

Each pair (a,b) maps to ((a+b)/sqrt(2), (a-b)/sqrt(2)). Detail arrays run
from finest to coarsest; the final approximation completes the coefficient
set. Non-power-of-two lengths are rejected or right-padded with zeros before
analysis. No periodic or reflected samples are introduced. Level zero is the
identity. Reconstruction diagnostics use the returned, rounded coefficients
and cover the original rows only; energy covers the padded signal.

The transform summarizes the entire supplied block, not a causal feature at
each row. The caller selects its observation and availability cutoff.
Reference: BYU ACME, Intro to Wavelets, orthonormal Haar filter bank and inverse:
https://labs.acme.byu.edu/Volume2/Wavelets/Wavelets.html
"""

from math import frexp, fsum, hypot, ldexp, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
LengthPolicy = Literal["reject", "zero_pad"]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=4_096)
    levels: int | None = Field(default=None, strict=True, ge=0, le=12)
    non_power_of_two: LengthPolicy = "reject"
    reconstruction_limit: int = Field(default=32, strict=True, ge=0, le=256)

    @model_validator(mode="after")
    def validate_length(self) -> Self:
        size = len(self.values)
        if size & (size - 1) and self.non_power_of_two == "reject":
            raise ValueError("values length must be a power of two, or select zero_pad")
        padded_size = 1 << (size - 1).bit_length()
        if self.levels is not None and self.levels > padded_size.bit_length() - 1:
            raise ValueError("levels exceeds log2 of the padded input length")
        return self


class DetailLevel(OutputModel):
    level: int = Field(ge=1, le=12)
    coefficients: list[Finite] = Field(max_length=2_048)
    energy: Nonnegative
    energy_fraction: Nonnegative | None
    energy_rounded_to_zero: bool


class Output(OutputModel):
    original_length: int
    transformed_length: int
    padded_count: int
    levels: int
    non_power_of_two: LengthPolicy
    approximation: list[Finite] = Field(max_length=4_096)
    details_finest_first: list[DetailLevel] = Field(max_length=12)
    input_energy: Nonnegative
    coefficient_energy: Nonnegative
    approximation_energy: Nonnegative
    relative_energy_error: Nonnegative | None
    energy_rounded_to_zero: bool
    coefficient_underflow_count: int
    reconstructed_prefix: list[Finite] = Field(max_length=256)
    omitted_reconstructed_count: int
    maximum_reconstruction_error: Nonnegative
    reconstruction_rmse: Nonnegative
    relative_reconstruction_l2_error: Nonnegative | None


def _physical_energy(values: list[float], scale_exponent: int) -> tuple[float, bool]:
    # hypot avoids squaring tiny detail coefficients before the original units
    # are restored; such squared terms can still be representable in those units.
    norm = hypot(*values)
    mantissa, exponent = frexp(norm)
    result = ldexp(mantissa * mantissa, 2 * (exponent + scale_exponent))
    return result, norm != 0.0 and result == 0.0


def execute(request: Input, context: OperationContext) -> Output:
    """Analyze and invert bounded blocks using power-of-two internal scaling."""
    original_size = len(request.values)
    size = 1 << (original_size - 1).bit_length()
    levels = request.levels if request.levels is not None else size.bit_length() - 1
    _, exponent = frexp(max(abs(value) for value in request.values))
    normalized = [ldexp(value, -exponent) for value in request.values]
    if any(
        value != 0.0 and scaled == 0.0
        for value, scaled in zip(request.values, normalized, strict=True)
    ):
        raise ValueError("input dynamic range loses a nonzero value during Haar scaling")
    normalized.extend([0.0] * (size - original_size))
    approximation = normalized[:]
    detail_arrays: list[list[float]] = []
    root_two = sqrt(2.0)
    for _ in range(levels):
        pairs = zip(approximation[::2], approximation[1::2], strict=True)
        next_approximation: list[float] = []
        detail: list[float] = []
        for left, right in pairs:
            pair_sum = fsum((left, right))
            pair_difference = fsum((left, -right))
            average = pair_sum / root_two
            difference = pair_difference / root_two
            if (pair_sum != 0.0 and average == 0.0) or (
                pair_difference != 0.0 and difference == 0.0
            ):
                raise ValueError("input dynamic range underflows a normalized Haar coefficient")
            next_approximation.append(average)
            detail.append(difference)
        detail_arrays.append(detail)
        approximation = next_approximation

    # Re-import the emitted binary64 coefficients before computing the inverse:
    # subnormal coefficient rounding must be reflected in reconstruction error.
    emitted_approximation = [ldexp(value, exponent) for value in approximation]
    emitted_details = [[ldexp(value, exponent) for value in row] for row in detail_arrays]
    restored_approximation = [ldexp(value, -exponent) for value in emitted_approximation]
    restored_details = [[ldexp(value, -exponent) for value in row] for row in emitted_details]
    underflows = sum(
        before != 0.0 and after == 0.0
        for before, after in zip(approximation, emitted_approximation, strict=True)
    ) + sum(
        before != 0.0 and after == 0.0
        for original, emitted in zip(detail_arrays, emitted_details, strict=True)
        for before, after in zip(original, emitted, strict=True)
    )
    reconstructed = restored_approximation[:]
    for detail in reversed(restored_details):
        previous: list[float] = []
        for average, difference in zip(reconstructed, detail, strict=True):
            previous.extend(
                (fsum((average, difference)) / root_two, fsum((average, -difference)) / root_two)
            )
        reconstructed = previous
    emitted_reconstruction = [ldexp(value, exponent) for value in reconstructed[:original_size]]
    errors = [
        fsum((ldexp(value, -exponent), -expected))
        for value, expected in zip(emitted_reconstruction, normalized[:original_size], strict=True)
    ]
    input_norm = hypot(*normalized)
    input_square_sum = input_norm * input_norm
    approximation_norm = hypot(*restored_approximation)
    approximation_square_sum = approximation_norm * approximation_norm
    detail_norms = [hypot(*row) for row in restored_details]
    detail_square_sums = [norm * norm for norm in detail_norms]
    coefficient_square_sum = fsum((approximation_square_sum, *detail_square_sums))
    input_energy = ldexp(input_square_sum, 2 * exponent)
    coefficient_energy = ldexp(coefficient_square_sum, 2 * exponent)
    approximation_energy, approximation_energy_underflow = _physical_energy(
        restored_approximation, exponent
    )
    detail_energies = [_physical_energy(row, exponent) for row in restored_details]
    error_norm = hypot(*errors)
    return Output(
        original_length=original_size,
        transformed_length=size,
        padded_count=size - original_size,
        levels=levels,
        non_power_of_two=request.non_power_of_two,
        approximation=emitted_approximation,
        details_finest_first=[
            DetailLevel(
                level=index + 1,
                coefficients=coefficients,
                energy=detail_energies[index][0],
                energy_fraction=(norm / input_norm) ** 2 if input_norm else None,
                energy_rounded_to_zero=detail_energies[index][1],
            )
            for index, (coefficients, norm) in enumerate(
                zip(emitted_details, detail_norms, strict=True)
            )
        ],
        input_energy=input_energy,
        coefficient_energy=coefficient_energy,
        approximation_energy=approximation_energy,
        relative_energy_error=(
            abs(coefficient_square_sum - input_square_sum) / input_square_sum
            if input_square_sum
            else None
        ),
        energy_rounded_to_zero=(
            (input_square_sum > 0 and input_energy == 0.0)
            or (coefficient_square_sum > 0 and coefficient_energy == 0.0)
            or approximation_energy_underflow
            or any(underflow for _, underflow in detail_energies)
        ),
        coefficient_underflow_count=underflows,
        reconstructed_prefix=emitted_reconstruction[: request.reconstruction_limit],
        omitted_reconstructed_count=max(0, original_size - request.reconstruction_limit),
        maximum_reconstruction_error=ldexp(max(map(abs, errors)), exponent),
        reconstruction_rmse=ldexp(error_norm / sqrt(original_size), exponent),
        relative_reconstruction_l2_error=(error_norm / input_norm if input_norm else None),
    )


OPERATION = Operation(
    id="features.haar_decomposition",
    kind="feature",
    description=(
        "Compute bounded multilevel orthonormal Haar coefficients with reject/zero-pad length "
        "policy, energy accounting and reconstruction diagnostics from returned coefficients. "
        "This summarizes a supplied block; coefficients are not per-row causal estimates."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
