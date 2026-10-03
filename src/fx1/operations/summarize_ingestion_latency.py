"""Describe signed ingestion lags; negative lags remain visible as clock anomalies.

All observations remain in signed summaries. A separate nonnegative summary
describes admissible lags without relabeling negatives as zero or removing their
counts. Quantiles use linear interpolation at index (n-1)*p; bins are [left,right).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from math import fsum, sqrt
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
FiniteSeconds = Annotated[float, Field(strict=True, ge=0, le=315_537_897_599)]
Probability = Annotated[float, Field(strict=True, ge=0, le=1)]


def _aware_clock(value: object) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ValueError("clock must be an explicit timezone-aware ISO datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("clock must include a timezone offset")
    try:
        return parsed.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("clock cannot be represented in UTC") from error


class Observation(InputModel):
    source: Name
    available_time: AwareDatetime
    ingested_time: AwareDatetime

    @field_validator("available_time", "ingested_time", mode="before")
    @classmethod
    def validate_clocks(cls, value: object) -> datetime:
        return _aware_clock(value)


class Input(InputModel):
    observations: list[Observation] = Field(min_length=1, max_length=10_000)
    quantiles: list[Probability] = Field(
        default_factory=lambda: [0.0, 0.5, 0.9, 0.95, 0.99, 1.0], min_length=1, max_length=21
    )
    nonnegative_bin_edges_seconds: list[FiniteSeconds] = Field(
        default_factory=lambda: [0.0, 1.0, 10.0, 60.0, 300.0, 3600.0, 86400.0],
        min_length=1,
        max_length=32,
    )
    source_offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    source_limit: int = Field(default=100, strict=True, ge=1, le=200)

    @model_validator(mode="after")
    def ordered_probabilities_and_edges(self) -> Self:
        if any(right <= left for left, right in pairwise(self.quantiles)):
            raise ValueError("quantiles must be strictly increasing without duplicates")
        edges = self.nonnegative_bin_edges_seconds
        if edges[0] != 0:
            raise ValueError("nonnegative bin edges must start at zero")
        if any(right <= left for left, right in pairwise(edges)):
            raise ValueError("nonnegative bin edges must be strictly increasing")
        return self


class Quantile(OutputModel):
    probability: float
    seconds: float


class Distribution(OutputModel):
    count: int
    minimum_seconds: float | None
    maximum_seconds: float | None
    mean_seconds: float | None
    population_std_seconds: float | None
    quantiles: list[Quantile]


class Bin(OutputModel):
    lower_seconds_inclusive: float | None
    upper_seconds_exclusive: float | None
    count: int


class Summary(OutputModel):
    observation_count: int
    negative_lag_count: int
    zero_lag_count: int
    positive_lag_count: int
    signed: Distribution
    nonnegative: Distribution
    histogram: list[Bin]


class SourceSummary(OutputModel):
    source: str
    summary: Summary


class Output(OutputModel):
    observation_count: int
    source_count: int
    lag_definition: Literal["ingested_time_minus_available_time_utc_seconds"] = (
        "ingested_time_minus_available_time_utc_seconds"
    )
    quantile_method: Literal["linear_at_n_minus_one_times_probability"] = (
        "linear_at_n_minus_one_times_probability"
    )
    overall: Summary
    source_offset: int
    next_source_offset: int | None
    sources: list[SourceSummary]


def _distribution(values: list[float], probabilities: list[float]) -> Distribution:
    if not values:
        return Distribution(
            count=0,
            minimum_seconds=None,
            maximum_seconds=None,
            mean_seconds=None,
            population_std_seconds=None,
            quantiles=[],
        )
    ordered = sorted(values)
    mean = fsum(ordered) / len(ordered)
    variance = fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    quantiles = []
    for probability in probabilities:
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        quantiles.append(
            Quantile(
                probability=probability,
                seconds=(1.0 - fraction) * ordered[lower] + fraction * ordered[upper],
            )
        )
    return Distribution(
        count=len(ordered),
        minimum_seconds=ordered[0],
        maximum_seconds=ordered[-1],
        mean_seconds=mean,
        population_std_seconds=sqrt(variance),
        quantiles=quantiles,
    )


def _summarize(values: list[float], request: Input) -> Summary:
    negative = sum(value < 0 for value in values)
    zeros = sum(value == 0 for value in values)
    nonnegative = [value for value in values if value >= 0]
    edges = request.nonnegative_bin_edges_seconds
    counts = [0] * len(edges)
    for value in nonnegative:
        counts[bisect_right(edges, value) - 1] += 1
    histogram = [Bin(lower_seconds_inclusive=None, upper_seconds_exclusive=0.0, count=negative)]
    histogram.extend(
        Bin(
            lower_seconds_inclusive=edge,
            upper_seconds_exclusive=edges[index + 1] if index + 1 < len(edges) else None,
            count=counts[index],
        )
        for index, edge in enumerate(edges)
    )
    return Summary(
        observation_count=len(values),
        negative_lag_count=negative,
        zero_lag_count=zeros,
        positive_lag_count=len(values) - negative - zeros,
        signed=_distribution(values, request.quantiles),
        nonnegative=_distribution(nonnegative, request.quantiles),
        histogram=histogram,
    )


def execute(request: Input, context: OperationContext) -> Output:
    by_source: dict[str, list[float]] = defaultdict(list)
    all_lags: list[float] = []
    for observation in request.observations:
        delta = observation.ingested_time.astimezone(UTC) - observation.available_time.astimezone(
            UTC
        )
        lag = delta.total_seconds()
        by_source[observation.source].append(lag)
        all_lags.append(lag)
    source_names = sorted(by_source)
    page_names = source_names[request.source_offset : request.source_offset + request.source_limit]
    stop = request.source_offset + len(page_names)
    return Output(
        observation_count=len(all_lags),
        source_count=len(source_names),
        overall=_summarize(all_lags, request),
        source_offset=request.source_offset,
        next_source_offset=stop if stop < len(source_names) else None,
        sources=[
            SourceSummary(source=name, summary=_summarize(by_source[name], request))
            for name in page_names
        ],
    )


OPERATION = Operation(
    id="skills.summarize_ingestion_latency",
    kind="skill",
    description=(
        "Describe timezone-aware ingestion-minus-availability lags overall and by source, "
        "with signed and nonnegative distributions, explicit negative-lag counts, "
        "configurable quantiles and bins, and bounded source pagination."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
