"""Seeded circular moving-block index resamples of a supplied source length.

For each replicate, draw independent uniform block starts in [0,n), append
(start+j) mod n for j=0..block_length-1, and truncate the final block to the
requested sample length. Block length cannot exceed n. Source rows are never
reordered inside a block except for the explicit last-to-first circular seam.

A fresh local Python random.Random(seed) instance uses integer seeding and
randrange(n), with draws consumed in replicate order, then block order. Global
random state is untouched. Repeatability is scoped to the same Python random
implementation and version, which are reported; no cross-version bitstream
guarantee or cryptographic randomness is claimed.

This generates indices only. Bootstrap inference needs appropriate stationarity,
dependence and block-length assumptions, which are not checked or selected here.
The circular seam can create an artificial adjacency. The caller must preserve
the source's temporal order and enforce sample availability before resampling.
References: arch CircularBlockBootstrap's fixed-length wrap-around convention:
https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html
Python RNG semantics: https://docs.python.org/3/library/random.html
"""

import platform
from random import Random
from typing import Literal, Self

from pydantic import Field, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    source_length: int = Field(strict=True, ge=1, le=10_000)
    block_length: int = Field(strict=True, ge=1, le=10_000)
    replicate_count: int = Field(default=10, strict=True, ge=1, le=200)
    resample_length: int | None = Field(default=None, strict=True, ge=1, le=10_000)
    seed: int = Field(default=0, strict=True, ge=0, le=2**64 - 1)

    @model_validator(mode="after")
    def bound_generation(self) -> Self:
        if self.block_length > self.source_length:
            raise ValueError("block_length cannot exceed source_length")
        length = self.resample_length if self.resample_length is not None else self.source_length
        if length * self.replicate_count > 100_000:
            raise ValueError("at most 100000 resampled indices are supported")
        return self


class Replicate(OutputModel):
    replicate_index: int
    indices: list[int] = Field(min_length=1, max_length=10_000)
    block_starts: list[int] = Field(min_length=1, max_length=10_000)
    final_block_used_length: int
    blocks_wrapping_in_returned_indices: int
    unique_source_indices: int
    omitted_source_indices: int


class Output(OutputModel):
    source_length: int
    block_length: int
    replicate_count: int
    resample_length: int
    seed: int
    total_index_count: int
    total_block_draws: int
    rng: Literal["python.random.Random(MT19937).randrange"] = (
        "python.random.Random(MT19937).randrange"
    )
    python_implementation: str
    python_version: str
    resampling: Literal["circular_fixed_length_blocks"] = "circular_fixed_length_blocks"
    replicates: list[Replicate] = Field(min_length=1, max_length=200)


def execute(request: Input, context: OperationContext) -> Output:
    """Draw block starts and construct complete bounded index replicates."""
    length = (
        request.resample_length if request.resample_length is not None else request.source_length
    )
    generator = Random(request.seed)
    replicates: list[Replicate] = []
    total_draws = 0
    for replicate_index in range(request.replicate_count):
        indices: list[int] = []
        starts: list[int] = []
        wraps = final_length = 0
        while len(indices) < length:
            start = generator.randrange(request.source_length)
            starts.append(start)
            final_length = min(request.block_length, length - len(indices))
            indices.extend(
                (start + offset) % request.source_length for offset in range(final_length)
            )
            wraps += start + final_length > request.source_length
        unique = len(set(indices))
        total_draws += len(starts)
        replicates.append(
            Replicate(
                replicate_index=replicate_index,
                indices=indices,
                block_starts=starts,
                final_block_used_length=final_length,
                blocks_wrapping_in_returned_indices=wraps,
                unique_source_indices=unique,
                omitted_source_indices=request.source_length - unique,
            )
        )
    return Output(
        source_length=request.source_length,
        block_length=request.block_length,
        replicate_count=request.replicate_count,
        resample_length=length,
        seed=request.seed,
        total_index_count=request.replicate_count * length,
        total_block_draws=total_draws,
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        replicates=replicates,
    )


OPERATION = Operation(
    id="skills.build_block_resamples",
    kind="skill",
    description=(
        "Generate seeded circular moving-block index resamples with independently drawn "
        "starts, explicit final-block truncation, bounded total output and Python RNG "
        "provenance. Does not select a block length or establish bootstrap validity."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
