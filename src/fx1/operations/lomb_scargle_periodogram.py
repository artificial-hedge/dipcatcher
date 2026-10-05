"""Unweighted irregular-time harmonic least squares on a supplied frequency grid.

At each f fit a*cos(2*pi*f*(t-t0))+b*sin(2*pi*f*(t-t0))+c. In floating mode,
c is fitted jointly by centering the harmonic columns and observations. In
fixed_sample_mean mode c is fixed at the sample mean; in zero mode c is zero.
The baseline is respectively the sample-mean or zero prediction. Power is the
fractional reduction of baseline mean squared error, with divisor N throughout.
No sampling-rate heuristic, frequency search, amplitude normalization, taper,
noise estimate, false-alarm probability or significance claim is supplied.

Times must increase strictly. Exact binary-rational time differences and cycle
products are reduced modulo one before trigonometric evaluation; the total
span must be at most one million cycles at every requested frequency. Exact
quarter cycles use exact harmonic values. Normal equations for the rounded
trigonometric design are accumulated and solved as Fractions, capped at 65536
bits. A zero determinant or determinant/trace^2 at or below rank_tolerance
leaves that frequency undefined rather than applying a pseudoinverse.
Nonzero reported quantities outside binary64 range cause an explicit error.

The normalized spectrum and ideal fit MSE use the exact rational solution to
that numerical design. Separately report MSE recalculated from the rounded
coefficients actually emitted, including its possibly negative improvement.
The caller is responsible for time units, availability and source selection.
Least-squares/floating-mean convention: Zechmeister and Kurster (2009):
https://arxiv.org/abs/0901.2573
"""

from fractions import Fraction
from math import atan2, cos, isfinite, sin, tau
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations._numeric import correctly_rounded_sqrt
from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Frequency = Annotated[float, Field(strict=True, ge=1e-12, le=1e6)]
MeanModel = Literal["floating", "fixed_sample_mean", "zero"]
BinStatus = Literal["fitted", "zero_baseline_energy", "rank_deficient", "ill_conditioned"]


class Input(InputModel):
    times_seconds: list[Value] = Field(min_length=3, max_length=4_096)
    values: list[Value] = Field(min_length=3, max_length=4_096)
    frequencies_hz: list[Frequency] = Field(min_length=1, max_length=512)
    mean_model: MeanModel = "floating"
    rank_tolerance: float = Field(default=1e-12, strict=True, ge=1e-15, le=1e-3)

    @model_validator(mode="after")
    def ordered_bounded_grid(self) -> Self:
        if len(self.times_seconds) != len(self.values):
            raise ValueError("values must align with observation times")
        if any(
            after <= before
            for before, after in zip(self.times_seconds[:-1], self.times_seconds[1:], strict=True)
        ):
            raise ValueError("times_seconds must be strictly increasing")
        if any(
            after <= before
            for before, after in zip(self.frequencies_hz[:-1], self.frequencies_hz[1:], strict=True)
        ):
            raise ValueError("frequencies_hz must be strictly increasing")
        if len(self.values) * len(self.frequencies_hz) > 131_072:
            raise ValueError("harmonic grid exceeds 131072 observation-frequency cells")
        span = Fraction(self.times_seconds[-1]) - Fraction(self.times_seconds[0])
        if span * Fraction(self.frequencies_hz[-1]) > 1_000_000:
            raise ValueError("observation span exceeds one million cycles at the highest frequency")
        return self


class FrequencyFit(OutputModel):
    frequency_index: int
    frequency_hz: float
    status: BinStatus
    normalized_power: float | None
    gram_determinant_over_trace_squared: float | None
    cosine_coefficient: float | None
    sine_coefficient: float | None
    intercept: float | None
    amplitude_from_returned_coefficients: float | None
    phase_radians: float | None
    ideal_fit_mean_squared_error: float | None
    returned_coefficients_mean_squared_error: float | None
    returned_coefficients_fractional_improvement: float | None


class Output(OutputModel):
    observation_count: int
    frequency_count: int
    fitted_frequency_count: int
    mean_model: MeanModel
    time_origin_seconds: float
    time_span_seconds: float
    sample_mean: float
    baseline_mean_squared_error: float
    constant_input: bool
    rank_tolerance: float
    peak_frequency_index: int | None
    peak_frequency_hz: float | None
    peak_period_seconds: float | None
    peak_normalized_power: float | None
    frequencies: list[FrequencyFit]
    observation_weights: Literal["uniform"] = "uniform"
    power_normalization: Literal[
        "fraction_of_baseline_mse_removed_by_exact_numerical_design_fit"
    ] = "fraction_of_baseline_mse_removed_by_exact_numerical_design_fit"
    phase_convention: Literal["amplitude_cos_2pi_f_t_minus_origin_minus_phase"] = (
        "amplitude_cos_2pi_f_t_minus_origin_minus_phase"
    )
    frequency_selection_validated: Literal[False] = False
    significance_test_performed: Literal[False] = False
    timing_verified: Literal[False] = False


def _bounded(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 65_536:
        raise ValueError("harmonic least-squares arithmetic exceeds 65536-bit budget")
    return value


def _number(value: Fraction, label: str) -> float:
    _bounded(value)
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _harmonics(cycles: Fraction) -> tuple[Fraction, Fraction]:
    turn = cycles % 1
    quarter = 4 * turn
    if quarter.denominator == 1:
        cosine, sine = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarter.numerator % 4]
        return Fraction(cosine), Fraction(sine)
    turn = turn if turn <= Fraction(1, 2) else turn - 1
    phase = _number(turn * Fraction(tau), "harmonic phase")
    return Fraction(cos(phase)), Fraction(sin(phase))


def _dot(left: list[Fraction], right: list[Fraction]) -> Fraction:
    return _bounded(sum((a * b for a, b in zip(left, right, strict=True)), Fraction()) / len(left))


def _undefined(
    index: int, frequency: float, status: BinStatus, rank_ratio: float | None
) -> FrequencyFit:
    return FrequencyFit(
        frequency_index=index,
        frequency_hz=frequency,
        status=status,
        normalized_power=None,
        gram_determinant_over_trace_squared=rank_ratio,
        cosine_coefficient=None,
        sine_coefficient=None,
        intercept=None,
        amplitude_from_returned_coefficients=None,
        phase_radians=None,
        ideal_fit_mean_squared_error=None,
        returned_coefficients_mean_squared_error=None,
        returned_coefficients_fractional_improvement=None,
    )


def execute(request: Input, context: OperationContext) -> Output:
    count = len(request.values)
    values = [Fraction(value) for value in request.values]
    mean = sum(values, Fraction()) / count
    offset = mean if request.mean_model != "zero" else Fraction()
    centered = [value - offset for value in values]
    baseline = _dot(centered, centered)
    origin = Fraction(request.times_seconds[0])
    times = [Fraction(value) - origin for value in request.times_seconds]
    fits: list[FrequencyFit] = []
    peak_index: int | None = None
    peak_power: Fraction | None = None
    for index, frequency in enumerate(request.frequencies_hz):
        if not baseline:
            fits.append(_undefined(index, frequency, "zero_baseline_energy", None))
            continue
        harmonics = [_harmonics(Fraction(frequency) * time) for time in times]
        cosine = [pair[0] for pair in harmonics]
        sine = [pair[1] for pair in harmonics]
        cosine_mean = (
            sum(cosine, Fraction()) / count if request.mean_model == "floating" else Fraction()
        )
        sine_mean = (
            sum(sine, Fraction()) / count if request.mean_model == "floating" else Fraction()
        )
        cosine_centered = [value - cosine_mean for value in cosine]
        sine_centered = [value - sine_mean for value in sine]
        cc, ss = _dot(cosine_centered, cosine_centered), _dot(sine_centered, sine_centered)
        cs = _dot(cosine_centered, sine_centered)
        yc, ys = _dot(centered, cosine_centered), _dot(centered, sine_centered)
        determinant = _bounded(cc * ss - cs * cs)
        if determinant < 0:
            raise ValueError("exact harmonic Gram determinant is negative")
        if not determinant:
            fits.append(_undefined(index, frequency, "rank_deficient", 0.0))
            continue
        rank_ratio = _bounded(determinant / ((cc + ss) ** 2))
        if rank_ratio <= Fraction(request.rank_tolerance):
            fits.append(
                _undefined(index, frequency, "ill_conditioned", _number(rank_ratio, "rank ratio"))
            )
            continue
        a = _bounded((yc * ss - ys * cs) / determinant)
        b = _bounded((ys * cc - yc * cs) / determinant)
        c = _bounded(offset - a * cosine_mean - b * sine_mean)
        explained = _bounded(a * yc + b * ys)
        if not 0 <= explained <= baseline:
            raise ValueError("exact least-squares improvement violates its projection bounds")
        power = _bounded(explained / baseline)
        a_float, b_float, c_float = (
            _number(a, "cosine coefficient"),
            _number(b, "sine coefficient"),
            _number(c, "intercept"),
        )
        returned_a, returned_b, returned_c = Fraction(a_float), Fraction(b_float), Fraction(c_float)
        residuals = [
            value - (returned_c + returned_a * cos_value + returned_b * sin_value)
            for value, cos_value, sin_value in zip(values, cosine, sine, strict=True)
        ]
        returned_mse = _dot(residuals, residuals)
        amplitude_squared = returned_a * returned_a + returned_b * returned_b
        amplitude = correctly_rounded_sqrt(
            amplitude_squared.numerator, amplitude_squared.denominator
        )
        if not isfinite(amplitude) or (amplitude_squared and amplitude == 0):
            raise ValueError("harmonic amplitude is outside finite binary64 range")
        fits.append(
            FrequencyFit(
                frequency_index=index,
                frequency_hz=frequency,
                status="fitted",
                normalized_power=_number(power, "normalized power"),
                gram_determinant_over_trace_squared=_number(rank_ratio, "rank ratio"),
                cosine_coefficient=a_float,
                sine_coefficient=b_float,
                intercept=c_float,
                amplitude_from_returned_coefficients=amplitude,
                phase_radians=atan2(b_float, a_float) if amplitude else None,
                ideal_fit_mean_squared_error=_number(baseline - explained, "ideal fit MSE"),
                returned_coefficients_mean_squared_error=_number(
                    returned_mse, "returned coefficient MSE"
                ),
                returned_coefficients_fractional_improvement=_number(
                    1 - returned_mse / baseline, "returned coefficient improvement"
                ),
            )
        )
        if peak_power is None or power > peak_power:
            peak_index, peak_power = index, power
    peak_frequency = request.frequencies_hz[peak_index] if peak_index is not None else None
    return Output(
        observation_count=count,
        frequency_count=len(fits),
        fitted_frequency_count=sum(fit.status == "fitted" for fit in fits),
        mean_model=request.mean_model,
        time_origin_seconds=request.times_seconds[0],
        time_span_seconds=_number(times[-1], "time span"),
        sample_mean=_number(mean, "sample mean"),
        baseline_mean_squared_error=_number(baseline, "baseline MSE"),
        constant_input=all(value == request.values[0] for value in request.values),
        rank_tolerance=request.rank_tolerance,
        peak_frequency_index=peak_index,
        peak_frequency_hz=peak_frequency,
        peak_period_seconds=_number(1 / Fraction(peak_frequency), "peak period")
        if peak_frequency is not None
        else None,
        peak_normalized_power=_number(peak_power, "peak power") if peak_power is not None else None,
        frequencies=fits,
    )


OPERATION = Operation(
    id="features.lomb_scargle_periodogram",
    kind="feature",
    description=(
        "Evaluate an irregular-time harmonic least-squares spectrum on explicit frequencies, "
        "with mean-model choices, exact numerical-design solves, rank checks and returned-fit diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
