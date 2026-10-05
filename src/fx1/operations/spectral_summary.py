"""Boxcar one-sided periodogram density of an evenly sampled real series.

For unnormalized DFT X[k], density is dt*|X[k]|²/n with interior bins doubled;
DC and even-length Nyquist bins are not doubled. No padding or taper is used.
Peak and Shannon entropy use positive-frequency bins only, excluding DC;
normalized entropy divides by log(number of positive-frequency bins), including
zero-power bins. A single positive bin has undefined normalized entropy.
Constant input has undefined positive-frequency summaries, even without mean
removal. Caller must enforce even spacing, ordering and PIT source selection.
FFT inputs are never scaled down, so global normalization cannot erase small
observations. Squared coefficient magnitudes retain binary exponents until
reported physical powers are formed. A float64 FFT can still lose information
through cancellation; underflow flags concern nonzero computed coefficients.

Conventions: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
"""

from math import frexp, fsum, hypot, ldexp, log, log1p
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Detrend = Literal["none", "mean"]


class Input(InputModel):
    values: list[Value] = Field(min_length=2, max_length=16_384)
    sample_interval_seconds: float = Field(strict=True, ge=1e-9, le=31_536_000, allow_inf_nan=False)
    detrend: Detrend = "mean"
    offset: int = Field(default=0, strict=True, ge=0, le=8_193)
    limit: int = Field(default=100, strict=True, ge=1, le=256)


class SpectralBin(OutputModel):
    index: int
    frequency_hz: Nonnegative
    density_value_squared_per_hz: Nonnegative
    density_underflow: bool


class Output(OutputModel):
    observation_count: int
    sample_interval_seconds: float
    detrend: Detrend
    frequency_resolution_hz: float
    total_bin_count: int
    positive_frequency_bin_count: int
    integrated_total_power: Nonnegative
    integrated_positive_frequency_power: Nonnegative
    total_power_underflow: bool
    positive_power_underflow: bool
    density_underflow_bin_count: int
    entropy_probability_underflow_bin_count: int
    peak_frequency_hz: Nonnegative | None
    peak_period_seconds: Nonnegative | None
    spectral_entropy_nats: Nonnegative | None
    normalized_spectral_entropy: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    spectral_status: Literal["constant_input", "no_positive_frequency_power", "finite"]
    entropy_status: Literal["no_positive_frequency_power", "single_positive_bin", "finite"]
    bins: list[SpectralBin] = Field(max_length=256)
    offset: int
    has_more: bool


Power = tuple[float, int]


def _power(magnitude: float, *, input_shift: int = 0, doubled: bool = False) -> Power:
    """Represent a nonnegative physical power as normalized mantissa * 2**exponent."""
    if magnitude == 0.0:
        return 0.0, 0
    mantissa, exponent = frexp(magnitude)
    squared_mantissa, squared_exponent = frexp(mantissa * mantissa)
    return squared_mantissa, 2 * (exponent - input_shift) + squared_exponent + int(doubled)


def _physical_value(power: Power, factor: float) -> float:
    """Apply scale before the final ldexp so subnormal intermediate squares survive."""
    mantissa, exponent = power
    if mantissa == 0.0 or factor == 0.0:
        return 0.0
    factor_mantissa, factor_exponent = frexp(factor)
    # Input and dt bounds keep positive exponents well below float overflow.
    return ldexp(mantissa * factor_mantissa, exponent + factor_exponent)


def _sum_powers(powers: list[Power]) -> Power:
    nonzero = [(mantissa, exponent) for mantissa, exponent in powers if mantissa]
    if not nonzero:
        return 0.0, 0
    largest_exponent = max(exponent for _, exponent in nonzero)
    return (
        fsum(ldexp(mantissa, exponent - largest_exponent) for mantissa, exponent in nonzero),
        largest_exponent,
    )


def execute(request: Input, context: OperationContext) -> Output:
    """Retain coefficient powers and shape summaries when physical powers underflow."""
    count = len(request.values)
    anchor = request.values[0]
    constant = all(value == anchor for value in request.values)
    # Remove a common level only when Sterbenz's interval guarantees every
    # subtraction is exact. This protects near-constant inputs without erasing
    # weak components of mixed-sign or very wide-range inputs. Constant shifts
    # affect only DC, which is assigned explicitly below.
    exact_anchor = (
        anchor > 0 and all(anchor / 2 <= value <= 2 * anchor for value in request.values)
    ) or (anchor < 0 and all(2 * anchor <= value <= anchor / 2 for value in request.values))
    values = [value - anchor for value in request.values] if exact_anchor else request.values
    largest_value = max(map(abs, values))
    input_shift = max(0, -frexp(largest_value)[1]) if largest_value else 0
    fft_values = [ldexp(value, input_shift) for value in values]
    transformed = np.fft.rfft(np.asarray(fft_values, dtype=np.float64), norm="backward")
    # Mean removal is exactly a DC removal in the DFT. Subtracting a rounded
    # sample mean from every value can contaminate weak non-DC components.
    dc = abs(fsum(request.values)) if request.detrend == "none" else 0.0
    weighted_power: list[Power] = [_power(dc)]
    for index in range(1, len(transformed)):
        magnitude = (
            0.0
            if constant
            else hypot(float(transformed[index].real), float(transformed[index].imag))
        )
        weighted_power.append(
            _power(
                magnitude,
                input_shift=input_shift,
                doubled=count % 2 != 0 or index != len(transformed) - 1,
            )
        )
    positive_sum = _sum_powers(weighted_power[1:])
    total_sum = _sum_powers(weighted_power)
    densities = [
        _physical_value(power, request.sample_interval_seconds / count) for power in weighted_power
    ]
    density_underflow = [
        power[0] > 0 and density == 0.0
        for power, density in zip(weighted_power, densities, strict=True)
    ]
    total_power = _physical_value(total_sum, 1.0 / count**2)
    positive_power = _physical_value(positive_sum, 1.0 / count**2)
    resolution = 1.0 / (count * request.sample_interval_seconds)
    positive_bins = len(weighted_power) - 1
    peak_frequency: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None
    entropy_status: Literal["no_positive_frequency_power", "single_positive_bin", "finite"] = (
        "no_positive_frequency_power"
    )
    probability_underflows = 0
    if positive_sum[0] > 0.0:
        peak_index = max(
            (index for index in range(1, len(weighted_power)) if weighted_power[index][0]),
            key=lambda index: (weighted_power[index][1], weighted_power[index][0]),
        )
        peak_frequency = peak_index * resolution
        scaled_sum, largest_exponent = positive_sum
        contributions: list[float] = []
        peak_mantissa, peak_exponent = weighted_power[peak_index]
        # Retain a tiny entropy contribution from the dominant bin even when
        # summing all masses rounds total mass back to the dominant mass.
        other_mass = fsum(
            ldexp(mantissa, exponent - largest_exponent)
            for index, (mantissa, exponent) in enumerate(weighted_power)
            if index > 0 and index != peak_index and mantissa
        )
        peak_mass = ldexp(peak_mantissa, peak_exponent - largest_exponent)
        for index, (mantissa, exponent) in enumerate(weighted_power[1:], start=1):
            if not mantissa:
                continue
            relative_power = mantissa, exponent - largest_exponent
            probability = _physical_value(relative_power, 1.0 / scaled_sum)
            probability_underflows += int(probability == 0.0)
            negative_log_probability = (
                log1p(other_mass / peak_mass)
                if index == peak_index
                else fsum((log(scaled_sum / mantissa), (largest_exponent - exponent) * log(2.0)))
            )
            contributions.append(
                _physical_value(relative_power, max(0.0, negative_log_probability) / scaled_sum)
            )
        entropy = fsum(contributions)
        if positive_bins > 1:
            normalized_entropy = min(1.0, entropy / log(positive_bins))
            entropy_status = "finite"
        else:
            entropy_status = "single_positive_bin"
    bins = [
        SpectralBin(
            index=index,
            frequency_hz=index * resolution,
            density_value_squared_per_hz=densities[index],
            density_underflow=density_underflow[index],
        )
        for index in range(request.offset, min(len(weighted_power), request.offset + request.limit))
    ]
    return Output(
        observation_count=count,
        sample_interval_seconds=request.sample_interval_seconds,
        detrend=request.detrend,
        frequency_resolution_hz=resolution,
        total_bin_count=len(weighted_power),
        positive_frequency_bin_count=positive_bins,
        integrated_total_power=total_power,
        integrated_positive_frequency_power=positive_power,
        total_power_underflow=total_sum[0] > 0.0 and total_power == 0.0,
        positive_power_underflow=positive_sum[0] > 0.0 and positive_power == 0.0,
        density_underflow_bin_count=sum(density_underflow),
        entropy_probability_underflow_bin_count=probability_underflows,
        peak_frequency_hz=peak_frequency,
        peak_period_seconds=1.0 / peak_frequency if peak_frequency is not None else None,
        spectral_entropy_nats=entropy,
        normalized_spectral_entropy=normalized_entropy,
        spectral_status=(
            "constant_input"
            if constant
            else "finite"
            if positive_sum[0] > 0
            else "no_positive_frequency_power"
        ),
        entropy_status=entropy_status,
        bins=bins,
        offset=request.offset,
        has_more=request.offset + len(bins) < len(weighted_power),
    )


OPERATION = Operation(
    id="features.spectral_summary",
    kind="feature",
    description=(
        "Compute a boxcar one-sided FFT density periodogram with none/mean detrending, "
        "correct DC/Nyquist weights, positive-frequency peak and entropy, constant/underflow "
        "statuses and paginated bins. Caller guarantees even spacing and PIT input selection."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
