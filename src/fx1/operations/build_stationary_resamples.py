"""Politis-Romano stationary-bootstrap indices with geometric restart blocks.

Draw the first source index uniformly. At each subsequent output position,
restart at an independent uniform source index with probability p=1/L;
otherwise continue to (previous+1) mod N. Thus untruncated block lengths have
geometric support 1,2,... and mean L, including noninteger L. L may exceed N;
a block can circle the source repeatedly. The final block is right-censored
by the requested output length. A restart is recorded even if its random
index equals the circular continuation index; boundaries cannot be inferred
from the indices alone.

The binary64 L is interpreted exactly as a rational. Each Bernoulli draw is
randrange(p.denominator)<p.numerator, avoiding floating uniform-threshold bias.
A fresh local Random(seed) consumes draws in replicate/position order: initial
index, then restart decision and, on success, a new source index. This includes
a decision draw when p=1. Repeatability is scoped to the reported Python RNG
implementation/version, without a cross-version or cryptographic guarantee.

The caller selects block length and supplies an appropriately ordered,
availability-selected stationary source. Indices do not validate dependence,
stationarity, uncertainty coverage or suitability of the circular seam.
Reference: Politis and Romano (1994), The Stationary Bootstrap:
https://doi.org/10.1080/01621459.1994.10476870
RNG semantics: https://docs.python.org/3/library/random.html
"""

import platform
from fractions import Fraction
from random import Random
from typing import Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    source_length: int = Field(strict=True, ge=1, le=10_000)
    expected_block_length: float = Field(strict=True, ge=1, le=1e6, allow_inf_nan=False)
    resample_length: int | None = Field(default=None, strict=True, ge=1, le=10_000)
    replicate_count: int = Field(default=10, strict=True, ge=1, le=200)
    seed: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)

    @model_validator(mode="after")
    def bound_output(self) -> Self:
        length = self.resample_length if self.resample_length is not None else self.source_length
        if length * self.replicate_count > 100_000:
            raise ValueError("at most 100000 total resampled indices are supported")
        return self


class Replicate(OutputModel):
    replicate_index: int
    indices: list[int] = Field(min_length=1, max_length=10_000)
    block_start_offsets: list[int] = Field(min_length=1, max_length=10_000)
    observed_block_lengths: list[int] = Field(min_length=1, max_length=10_000)
    restart_count: int
    circular_continuation_wraps: int
    unique_source_indices: int
    final_block_end_observed: Literal[False] = False


class Output(OutputModel):
    source_length: int
    resample_length: int
    replicate_count: int
    expected_block_length: float
    restart_probability: float
    restart_probability_numerator: int
    restart_probability_denominator: int
    expected_blocks_per_replicate: float
    total_restart_decisions: int
    total_uniform_source_draws: int
    total_realized_blocks: int
    total_index_count: int
    seed: int
    rng: Literal["python.random.Random(MT19937).randrange"] = (
        "python.random.Random(MT19937).randrange"
    )
    python_implementation: str
    python_version: str
    replicates: list[Replicate] = Field(min_length=1, max_length=200)


def execute(request: Input, context: OperationContext) -> Output:
    """Sample exact-rational restart decisions and retain realized block boundaries."""
    length = (
        request.resample_length if request.resample_length is not None else request.source_length
    )
    probability = 1 / Fraction(request.expected_block_length)
    generator = Random(request.seed)
    replicates: list[Replicate] = []
    total_blocks = 0
    for replicate_index in range(request.replicate_count):
        indices = [generator.randrange(request.source_length)]
        starts = [0]
        wraps = 0
        for position in range(1, length):
            if generator.randrange(probability.denominator) < probability.numerator:
                starts.append(position)
                indices.append(generator.randrange(request.source_length))
            else:
                wraps += indices[-1] == request.source_length - 1
                indices.append((indices[-1] + 1) % request.source_length)
        stops = starts[1:] + [length]
        block_lengths = [stop - start for start, stop in zip(starts, stops, strict=True)]
        total_blocks += len(starts)
        replicates.append(
            Replicate(
                replicate_index=replicate_index,
                indices=indices,
                block_start_offsets=starts,
                observed_block_lengths=block_lengths,
                restart_count=len(starts) - 1,
                circular_continuation_wraps=wraps,
                unique_source_indices=len(set(indices)),
            )
        )
    return Output(
        source_length=request.source_length,
        resample_length=length,
        replicate_count=request.replicate_count,
        expected_block_length=request.expected_block_length,
        restart_probability=float(probability),
        restart_probability_numerator=probability.numerator,
        restart_probability_denominator=probability.denominator,
        expected_blocks_per_replicate=float(1 + (length - 1) * probability),
        total_restart_decisions=request.replicate_count * (length - 1),
        total_uniform_source_draws=total_blocks,
        total_realized_blocks=total_blocks,
        total_index_count=request.replicate_count * length,
        seed=request.seed,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        replicates=replicates,
    )


OPERATION = Operation(
    id="skills.build_stationary_resamples",
    kind="skill",
    description=(
        "Generate seeded stationary-bootstrap indices with exact-rational geometric restart "
        "decisions, circular continuation, realized block boundaries and bounded output."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
