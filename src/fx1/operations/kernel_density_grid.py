"""Evaluate a univariate Gaussian kernel mixture at supplied query locations.

With p_i=w_i/sum(w), bandwidth h is the kernel standard deviation in the input
units: f(q)=sum_i p_i*phi((q-x_i)/h)/h, F(q)=sum_i p_i*Phi((q-x_i)/h).
Bandwidth is mandatory and never estimated; query order and repeated locations
are preserved. Zero-weight observations do not contribute. This fits a mixture,
not a proper forecast score, calibration finding or evidence of independence.

Differences and standardized quadratic exponents begin as exact fractions.
Log-sum-exp retains density and tail logs when ordinary values underflow.
Normal tails use erfc below |z|=12 and the alternating Mills expansion beyond,
stopping at a term below 1e-18, at increasing terms, or after 64 terms. Logs and
special functions are approximate. The smaller mixture tail is evaluated in
log space and its complement is formed with log1p/expm1. Boundary probabilities,
zero densities and tiny quadratic terms are explicitly diagnosed. Distances
above one million bandwidths fail; no kernel is truncated or silently discarded.

Gaussian mixture convention and normal tail expansion references:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
https://dlmf.nist.gov/7.12
"""

from fractions import Fraction
from math import erfc, exp, expm1, fsum, isfinite, log, log1p, pi, sqrt
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100)]
Weight = Annotated[float, Field(strict=True, ge=0, le=1e100)]
_LOG_NORMALIZER = 0.5 * log(2 * pi)
_SQRT_TWO = sqrt(2)


class Input(InputModel):
    samples: list[Value] = Field(min_length=1, max_length=1_024)
    queries: list[Value] = Field(min_length=1, max_length=512)
    bandwidth: float = Field(strict=True, ge=1e-100, le=1e100)
    weights: list[Weight] | None = Field(default=None, min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def bounded_grid(self) -> Self:
        if len(self.samples) * len(self.queries) > 65_536:
            raise ValueError("KDE grid exceeds 65536 sample-query cells")
        if self.weights is not None:
            if len(self.weights) != len(self.samples) or not any(self.weights):
                raise ValueError("weights must align with samples and have positive total mass")
            if any(0 < weight < 1e-100 for weight in self.weights):
                raise ValueError("positive weights must be at least 1e-100")
        return self


class Query(OutputModel):
    query_index: int
    location: float
    density: float
    log_density: float
    density_underflow: bool
    cdf: float
    survival_probability: float
    log_cdf: float
    log_survival_probability: float
    cdf_rounded_to_boundary: bool
    survival_rounded_to_boundary: bool
    log_cdf_rounded_to_zero: bool
    log_survival_rounded_to_zero: bool
    complement_roundoff_adjusted: bool
    quadratic_underflow_components: int
    asymptotic_tail_components: int
    maximum_absolute_standardized_distance: float


class Output(OutputModel):
    sample_count: int
    positive_weight_count: int
    query_count: int
    bandwidth: float
    weight_sum: float
    normalized_weights: list[float]
    effective_weight_count: float
    queries: list[Query]
    bandwidth_definition: Literal["gaussian_kernel_standard_deviation_in_input_units"] = (
        "gaussian_kernel_standard_deviation_in_input_units"
    )
    kernel_truncated: Literal[False] = False
    bandwidth_fitted: Literal[False] = False
    timing_verified: Literal[False] = False


def _number(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} overflows binary64") from error
    if not isfinite(result) or (value and result == 0):
        raise ValueError(f"nonzero {label} is outside finite binary64 range")
    return result


def _log_probability(value: Fraction) -> float:
    if value > Fraction(1, 2):
        return log1p(_number(value - 1, "weight deviation"))
    return log(_number(value, "normalized weight"))


def _logsumexp(values: list[float]) -> float:
    largest = max(values)
    return fsum((largest, log(fsum(exp(value - largest) for value in values))))


def _normal_logs(z: float, quadratic: float) -> tuple[float, float]:
    """Return log Phi(z), log Phi(-z), with a direct lower-tail expansion."""
    absolute = abs(z)
    if absolute < 12:
        log_tail = log(0.5 * erfc(absolute / _SQRT_TWO))
    else:
        inverse_square = 1 / (absolute * absolute)
        term = 1.0
        terms = [term]
        for index in range(1, 65):
            following = -term * (2 * index - 1) * inverse_square
            if abs(following) >= abs(term):
                break
            terms.append(following)
            term = following
            if abs(term) <= 1e-18:
                break
        correction = fsum(terms)
        if correction <= 0:
            raise ValueError("normal-tail expansion lost positivity")
        log_tail = fsum((-quadratic, -log(absolute), -_LOG_NORMALIZER, log(correction)))
    log_complement = log1p(-exp(log_tail))
    return (log_tail, log_complement) if z <= 0 else (log_complement, log_tail)


def execute(request: Input, context: OperationContext) -> Output:
    weights = [
        Fraction(value)
        for value in (
            request.weights if request.weights is not None else [1.0] * len(request.samples)
        )
    ]
    total = sum(weights, Fraction())
    probabilities = [weight / total for weight in weights]
    active = [index for index, weight in enumerate(weights) if weight]
    log_weights = {index: _log_probability(probabilities[index]) for index in active}
    samples = [Fraction(value) for value in request.samples]
    bandwidth = Fraction(request.bandwidth)
    log_bandwidth = log(request.bandwidth)
    results: list[Query] = []
    for query_index, query in enumerate(request.queries):
        exact_query = Fraction(query)
        density_terms: list[float] = []
        cdf_terms: list[float] = []
        survival_terms: list[float] = []
        tiny_quadratics = asymptotic = 0
        maximum_distance = 0.0
        for index in active:
            standardized = (exact_query - samples[index]) / bandwidth
            if abs(standardized) > 1_000_000:
                raise ValueError("query/sample separation exceeds one million bandwidths")
            z = _number(standardized, "standardized distance")
            exact_quadratic = standardized * standardized / 2
            quadratic = float(exact_quadratic)
            tiny_quadratics += bool(exact_quadratic and quadratic == 0)
            asymptotic += abs(z) >= 12
            maximum_distance = max(maximum_distance, abs(z))
            lower, upper = _normal_logs(z, quadratic)
            density_terms.append(fsum((log_weights[index], -quadratic)))
            cdf_terms.append(fsum((log_weights[index], lower)))
            survival_terms.append(fsum((log_weights[index], upper)))
        log_density = fsum((_logsumexp(density_terms), -log_bandwidth, -_LOG_NORMALIZER))
        density = exp(log_density)
        raw_log_cdf, raw_log_survival = _logsumexp(cdf_terms), _logsumexp(survival_terms)
        cdf_is_smaller = raw_log_cdf <= raw_log_survival
        log_small = min(raw_log_cdf, raw_log_survival)
        small = exp(log_small)
        if small > 0.5 + 64 / 2**52:
            raise ValueError("normal-mixture tail probabilities lost complementary normalization")
        adjusted = small > 0.5
        if adjusted:
            small, log_small = 0.5, log(0.5)
        large = -expm1(log_small)
        log_large = log1p(-small)
        cdf, survival = (small, large) if cdf_is_smaller else (large, small)
        log_cdf, log_survival = (log_small, log_large) if cdf_is_smaller else (log_large, log_small)
        if not all(isfinite(value) for value in (density, log_density, log_cdf, log_survival)):
            raise ValueError("kernel density or tail logarithm is outside finite range")
        results.append(
            Query(
                query_index=query_index,
                location=query,
                density=density,
                log_density=log_density,
                density_underflow=density == 0,
                cdf=cdf,
                survival_probability=survival,
                log_cdf=log_cdf,
                log_survival_probability=log_survival,
                cdf_rounded_to_boundary=cdf in (0, 1),
                survival_rounded_to_boundary=survival in (0, 1),
                log_cdf_rounded_to_zero=log_cdf == 0,
                log_survival_rounded_to_zero=log_survival == 0,
                complement_roundoff_adjusted=adjusted,
                quadratic_underflow_components=tiny_quadratics,
                asymptotic_tail_components=asymptotic,
                maximum_absolute_standardized_distance=maximum_distance,
            )
        )
    return Output(
        sample_count=len(request.samples),
        positive_weight_count=len(active),
        query_count=len(request.queries),
        bandwidth=request.bandwidth,
        weight_sum=_number(total, "weight sum"),
        normalized_weights=[_number(value, "normalized weight") for value in probabilities],
        effective_weight_count=_number(
            total * total / sum((weight * weight for weight in weights), Fraction()),
            "effective weight count",
        ),
        queries=results,
    )


OPERATION = Operation(
    id="features.kernel_density_grid",
    kind="feature",
    description=(
        "Evaluate an explicit-bandwidth weighted Gaussian KDE on supplied queries, including "
        "density/CDF/survival logarithms, stable tail evaluation and boundary-rounding diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
