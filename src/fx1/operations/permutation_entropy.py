"""Shannon entropy of delayed ordinal-pattern frequencies, divided by log(m!).

All starting positions are used: embedding i contains x[i+j*delay], j=0..m-1.
Stable ties put earlier embedding positions first; drop/reject policies are
also explicit. The sample threshold is a caller-selected count diagnostic,
not a guarantee of estimation accuracy or independent observations.
The caller must enforce sampling order and point-in-time source availability.

References: Bandt and Pompe, Phys. Rev. Lett. 88, 174102 (2002),
doi:10.1103/PhysRevLett.88.174102; delayed embeddings and log(m!) scaling:
https://link.aps.org/accepted/10.1103/PhysRevE.91.023101, equations 1-2.
"""

from collections import Counter
from math import factorial, fsum, log
from typing import Annotated, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Value = Annotated[float, Field(strict=True, ge=-1e100, le=1e100, allow_inf_nan=False)]
TiePolicy = Literal["stable", "drop", "reject"]
SampleStatus = Literal[
    "no_embeddings", "no_eligible_patterns", "below_threshold", "meets_threshold"
]


class Input(InputModel):
    values: list[Value] = Field(min_length=1, max_length=10_000)
    embedding_dimension: int = Field(default=3, strict=True, ge=2, le=7)
    delay: int = Field(default=1, strict=True, ge=1, le=1_000)
    tie_policy: TiePolicy = "stable"
    minimum_patterns: int = Field(default=20, strict=True, ge=1, le=10_000)
    pattern_limit: int = Field(default=50, strict=True, ge=1, le=200)


class PatternFrequency(OutputModel):
    ordinal_positions: list[int] = Field(min_length=2, max_length=7)
    count: int = Field(ge=1)
    probability: float = Field(gt=0, le=1, allow_inf_nan=False)


class Output(OutputModel):
    embedding_dimension: int
    delay: int
    tie_policy: TiePolicy
    total_embeddings: int
    tied_embeddings: int
    discarded_embeddings: int
    used_embeddings: int
    observed_pattern_count: int
    possible_pattern_count: int
    entropy_nats: float | None = Field(ge=0, allow_inf_nan=False)
    normalized_entropy: float | None = Field(ge=0, le=1, allow_inf_nan=False)
    minimum_patterns: int
    sample_status: SampleStatus
    patterns: list[PatternFrequency] = Field(max_length=200)
    omitted_pattern_count: int


def execute(request: Input, context: OperationContext) -> Output:
    """Count bounded ordinal patterns; reporting limits do not alter the entropy."""
    dimension = request.embedding_dimension
    embeddings = max(0, len(request.values) - (dimension - 1) * request.delay)
    counts: Counter[tuple[int, ...]] = Counter()
    tied = discarded = 0
    for start in range(embeddings):
        values = [request.values[start + index * request.delay] for index in range(dimension)]
        has_ties = len(set(values)) != dimension
        if has_ties:
            tied += 1
            if request.tie_policy == "reject":
                raise ValueError(
                    f"ordinal embedding starting at index {start} contains tied values"
                )
            if request.tie_policy == "drop":
                discarded += 1
                continue
        pattern = tuple(sorted(range(dimension), key=lambda position: (values[position], position)))
        counts[pattern] += 1
    used = sum(counts.values())
    possible = factorial(dimension)
    entropy = (
        -fsum((count / used) * log(count / used) for count in counts.values()) if used else None
    )
    normalized = min(1.0, max(0.0, entropy / log(possible))) if entropy is not None else None
    status: SampleStatus
    if embeddings == 0:
        status = "no_embeddings"
    elif used == 0:
        status = "no_eligible_patterns"
    elif used < request.minimum_patterns:
        status = "below_threshold"
    else:
        status = "meets_threshold"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    patterns = [
        PatternFrequency(ordinal_positions=list(pattern), count=count, probability=count / used)
        for pattern, count in ranked[: request.pattern_limit]
    ]
    return Output(
        embedding_dimension=dimension,
        delay=request.delay,
        tie_policy=request.tie_policy,
        total_embeddings=embeddings,
        tied_embeddings=tied,
        discarded_embeddings=discarded,
        used_embeddings=used,
        observed_pattern_count=len(counts),
        possible_pattern_count=possible,
        entropy_nats=entropy,
        normalized_entropy=normalized,
        minimum_patterns=request.minimum_patterns,
        sample_status=status,
        patterns=patterns,
        omitted_pattern_count=max(0, len(counts) - request.pattern_limit),
    )


OPERATION = Operation(
    id="features.permutation_entropy",
    kind="feature",
    description=(
        "Measure Shannon entropy of delayed ordinal patterns with explicit stable/drop/reject "
        "tie policies, log(m!) normalization, count-threshold diagnostics and bounded frequency "
        "output. The threshold does not establish statistical adequacy or market evidence."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
