"""Boxcar one-sided periodogram density of an evenly sampled real series.

For unnormalized DFT X[k], density is dt*|X[k]|²/n with interior bins doubled;
DC and even-length Nyquist bins are not doubled. No padding or taper is used.
Peak and Shannon entropy use positive-frequency bins only, excluding DC;
normalized entropy divides by log(number of positive-frequency bins), including
zero-power bins. A single positive bin has undefined normalized entropy.
Constant input has undefined positive-frequency summaries, even without mean
removal. Caller must enforce even spacing, ordering and PIT source selection.

Conventions: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
and https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
"""

from math import fsum, log
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


class Output(OutputModel):
    observation_count: int
    sample_interval_seconds: float
    detrend: Detrend
    frequency_resolution_hz: float
    total_bin_count: int
    positive_frequency_bin_count: int
    integrated_total_power: Nonnegative
    integrated_positive_frequency_power: Nonnegative
    positive_power_underflow: bool
    peak_frequency_hz: Nonnegative | None
    peak_period_seconds: Nonnegative | None
    spectral_entropy_nats: Nonnegative | None
    normalized_spectral_entropy: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    spectral_status: Literal["constant_input", "no_positive_frequency_power", "finite"]
    entropy_status: Literal["no_positive_frequency_power", "single_positive_bin", "finite"]
    bins: list[SpectralBin] = Field(max_length=256)
    offset: int
    has_more: bool


def execute(request: Input, context: OperationContext) -> Output:
    """Use normalized FFT inputs and retain shape summaries through power underflow."""
    count = len(request.values)
    constant = all(value == request.values[0] for value in request.values)
    if request.detrend == "mean":
        anchor = request.values[0]
        offsets = [value - anchor for value in request.values]
        scale = max(abs(value) for value in offsets)
        normalized = [value / scale for value in offsets] if scale else [0.0] * count
        mean = fsum(normalized) / count
        normalized = [value - mean for value in normalized]
    else:
        scale = max(abs(value) for value in request.values)
        normalized = [value / scale for value in request.values] if scale else [0.0] * count
    transformed = np.fft.rfft(np.asarray(normalized, dtype=np.float64), norm="backward")
    weighted_power = [float(value.real**2 + value.imag**2) for value in transformed]
    for index in range(1, len(weighted_power)):
        if count % 2 != 0 or index != len(weighted_power) - 1:
            weighted_power[index] *= 2.0
    if constant:
        # Constant signals have exactly zero positive-frequency power; exclude
        # FFT roundoff from producing a spurious periodicity or entropy.
        weighted_power[1:] = [0.0] * (len(weighted_power) - 1)
    positive_sum = fsum(weighted_power[1:])
    total_sum = fsum(weighted_power)
    density_factor = request.sample_interval_seconds / count
    densities = [(value * density_factor * scale) * scale for value in weighted_power]
    total_power = ((total_sum / count**2) * scale) * scale
    positive_power = ((positive_sum / count**2) * scale) * scale
    resolution = 1.0 / (count * request.sample_interval_seconds)
    positive_bins = len(weighted_power) - 1
    peak_frequency: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None
    entropy_status: Literal["no_positive_frequency_power", "single_positive_bin", "finite"] = (
        "no_positive_frequency_power"
    )
    if positive_sum > 0.0:
        peak_index = max(range(1, len(weighted_power)), key=lambda index: weighted_power[index])
        peak_frequency = peak_index * resolution
        probabilities = [value / positive_sum for value in weighted_power[1:] if value > 0.0]
        entropy = max(0.0, -fsum(probability * log(probability) for probability in probabilities))
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
        positive_power_underflow=positive_sum > 0.0 and positive_power == 0.0,
        peak_frequency_hz=peak_frequency,
        peak_period_seconds=1.0 / peak_frequency if peak_frequency is not None else None,
        spectral_entropy_nats=entropy,
        normalized_spectral_entropy=normalized_entropy,
        spectral_status=(
            "constant_input"
            if constant
            else "finite"
            if positive_sum > 0
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
