"""Read bounded RIFF/WAVE PCM or IEEE samples without audio playback.

Little-endian RIFF only, exactly one fmt before one data chunk. PCM uses 8/16/
24/32-bit containers (8-bit unsigned, others signed); IEEE uses 32/64 bits.
Formats 1/3 and extensible PCM/IEEE GUIDs are supported. Extensible valid bits
must equal container bits; nonzero speaker masks must use known bits and match
the channel count. Bare formats allow only mono/stereo. No normalization,
clipping, resampling or inferred units; integers are exact decimal strings and
finite floats are float.hex strings, including signed zero. Nonfinite samples
are rejected even outside the requested frame page. Frame positions are exact
sample-rate fractions; offsets and channel ordinals are zero based.

Chunk extents/padding and RIFF size must consume the full source; odd chunk pad
bytes must be zero. LIST/RIFF/RIFX/wavl/slnt containers are unsupported. Other
ancillary chunk payloads are opaque, hashed and never executed/interpreted.
A fact chunk, if present, must have exactly four bytes; its unsigned count and
agreement with stored frame count are observations, not an inferred convention.
Fact is required for IEEE in this subset. RF64/BW64 and compressed codecs fail.
Bare IEEE fmt requires the 18-byte WAVEFORMATEX including zero cbSize; PCM
accepts the 16-byte legacy form or 18 bytes with zero cbSize.

Limits: 8 MB source, 256 chunks, 32 channels, sample rate 1..768000 Hz, 1M scalar
samples; output at most 200 frames, 6400 scalars and 512 KB JSON. Every stored
sample is validated before output. Duration is stored frames/rate, not playlist
playback duration; ancillary playback instructions are not followed.
References: Microsoft/IBM Multimedia Programming Interface and Data
Specifications (1991), RIFF/WAVE; Microsoft WAVEFORMATEX/WAVEFORMATEXTENSIBLE.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=1_000_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Chunk(OutputModel):
    chunk_id: str
    header_byte_start: int
    payload_byte_start: int
    payload_bytes: int
    padding_bytes: int
    payload_sha256: str
    payload_interpreted: bool


class Sample(OutputModel):
    channel_index: int
    source_byte_start: int
    raw_hex: str
    value: str


class Frame(OutputModel):
    frame_index: int
    source_byte_start: int
    time_numerator: int
    samples: list[Sample]


class Output(OutputModel):
    format_code: int
    encoding: Literal["pcm_unsigned_8", "pcm_signed_integer", "ieee_float"]
    extensible: bool
    channel_count: int
    channel_mask: int | None
    speaker_bit_indices: list[int] | None
    sample_rate: int
    bits_per_sample: int
    block_align: int
    byte_rate: int
    frame_count: int
    scalar_sample_count: int
    has_frames: bool
    duration_numerator: int
    time_denominator: int
    fact_declared_sample_count: int | None
    fact_count_equals_frame_count: bool | None
    fact_count_semantics_verified: Literal[False] = False
    chunks: list[Chunk]
    frames: list[Frame]
    offset: int
    has_more: bool
    normalization_applied: Literal[False] = False
    ancillary_semantics_validated: Literal[False] = False
    full_framing_and_samples_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".wav", ".wave"), max_bytes=8_000_000)
    if len(source) < 12 or source[:4] != b"RIFF" or source[8:12] != b"WAVE":
        raise ValueError("WAVE requires little-endian RIFF/WAVE; RF64/BW64/RIFX are unsupported")
    if int.from_bytes(source[4:8], "little") + 8 != len(source):
        raise ValueError("WAVE RIFF size does not consume exactly the source")
    position = 12
    chunks: list[Chunk] = []
    special: dict[bytes, tuple[int, int]] = {}
    while position < len(source):
        if len(chunks) >= 256 or position + 8 > len(source):
            raise ValueError("WAVE chunk limit exceeded or chunk header truncated")
        identity = source[position : position + 4]
        if any(byte < 32 or byte > 126 for byte in identity):
            raise ValueError("WAVE chunk IDs require four printable ASCII characters")
        size = int.from_bytes(source[position + 4 : position + 8], "little")
        start, end = position + 8, position + 8 + size
        padded = end + (size & 1)
        if padded > len(source) or size & 1 and source[end] != 0:
            raise ValueError("WAVE chunk is truncated or odd-byte padding is nonzero")
        if identity in {b"LIST", b"RIFF", b"RIFX", b"wavl", b"slnt"}:
            raise ValueError("WAVE nested/segmented audio containers are unsupported")
        if identity in {b"fmt ", b"data", b"fact"}:
            if identity in special or identity == b"data" and b"fmt " not in special:
                raise ValueError("WAVE fmt/data/fact must be unique and fmt must precede data")
            special[identity] = (start, size)
        chunks.append(
            Chunk(
                chunk_id=identity.decode("ascii"),
                header_byte_start=position,
                payload_byte_start=start,
                payload_bytes=size,
                padding_bytes=size & 1,
                payload_sha256=hashlib.sha256(source[start:end]).hexdigest(),
                payload_interpreted=identity in {b"fmt ", b"data", b"fact"},
            )
        )
        position = padded
    if b"fmt " not in special or b"data" not in special:
        raise ValueError("WAVE needs exactly one fmt and data chunk")
    fmt_start, fmt_size = special[b"fmt "]
    if fmt_size not in {16, 18, 40}:
        raise ValueError("WAVE fmt size must be 16, 18 or extensible 40 bytes")
    fmt = source[fmt_start : fmt_start + fmt_size]
    tag, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", fmt[:16])
    mask: int | None = None
    codec = tag
    if tag == 65534:
        if fmt_size != 40 or int.from_bytes(fmt[16:18], "little") != 22:
            raise ValueError("WAVE extensible format requires exactly 22 extension bytes")
        valid = int.from_bytes(fmt[18:20], "little")
        mask = int.from_bytes(fmt[20:24], "little")
        codec = int.from_bytes(fmt[24:28], "little")
        if fmt[28:40] != bytes.fromhex("00001000800000aa00389b71") or codec not in {1, 3}:
            raise ValueError("WAVE extensible GUID must identify PCM or IEEE float")
        if valid != bits or mask & ~0x3FFFF or mask and mask.bit_count() != channels:
            raise ValueError("WAVE reduced valid bits or inconsistent/reserved speaker masks fail")
    elif tag not in {1, 3} or channels > 2:
        raise ValueError("WAVE bare format supports mono/stereo PCM or IEEE only")
    elif fmt_size == 40 or fmt_size == 18 and fmt[16:18] != b"\0\0":
        raise ValueError("WAVE bare fmt extensions must be empty")
    elif tag == 3 and fmt_size != 18:
        raise ValueError("WAVE bare IEEE requires WAVEFORMATEX with zero cbSize")
    if not 1 <= channels <= 32 or not 1 <= rate <= 768_000:
        raise ValueError("WAVE channel count/sample rate is outside its bound")
    if (codec == 1 and bits not in {8, 16, 24, 32}) or (codec == 3 and bits not in {32, 64}):
        raise ValueError("WAVE sample precision is unsupported")
    width = bits // 8
    if align != channels * width or byte_rate != rate * align:
        raise ValueError("WAVE block alignment/byte rate differs from sample format")
    data_start, data_size = special[b"data"]
    if data_size % align or data_size // width > 1_000_000:
        raise ValueError("WAVE data has a partial frame or exceeds 1M scalar samples")
    frame_count = data_size // align
    fact: int | None = None
    if b"fact" in special:
        fact_start, fact_size = special[b"fact"]
        if fact_size != 4:
            raise ValueError("WAVE fact extensions are unsupported; exactly four bytes required")
        fact = int.from_bytes(source[fact_start : fact_start + 4], "little")
    elif codec == 3:
        raise ValueError("WAVE IEEE subset requires a fact chunk")
    frames: list[Frame] = []
    page_bytes = 0
    for frame_index in range(frame_count):
        start = data_start + frame_index * align
        selected = request.offset <= frame_index < request.offset + request.limit
        samples: list[Sample] = []
        for channel in range(channels):
            sample_start = start + channel * width
            raw = source[sample_start : sample_start + width]
            if codec == 1:
                value = str(int.from_bytes(raw, "little", signed=bits != 8))
            else:
                floating: float = struct.unpack("<f" if bits == 32 else "<d", raw)[0]
                if not math.isfinite(floating):
                    raise ValueError("WAVE nonfinite IEEE samples are unsupported")
                value = floating.hex()
            if selected:
                samples.append(
                    Sample(
                        channel_index=channel,
                        source_byte_start=sample_start,
                        raw_hex=raw.hex(),
                        value=value,
                    )
                )
        if selected:
            frame = Frame(
                frame_index=frame_index,
                source_byte_start=start,
                time_numerator=frame_index,
                samples=samples,
            )
            page_bytes += len(json.dumps(frame.model_dump()).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("WAVE encoded frame page exceeds 512 KB")
            frames.append(frame)
    return Output(
        format_code=tag,
        encoding="ieee_float"
        if codec == 3
        else "pcm_unsigned_8"
        if bits == 8
        else "pcm_signed_integer",
        extensible=tag == 65534,
        channel_count=channels,
        channel_mask=mask,
        speaker_bit_indices=[bit for bit in range(18) if mask & (1 << bit)] if mask else None,
        sample_rate=rate,
        bits_per_sample=bits,
        block_align=align,
        byte_rate=byte_rate,
        frame_count=frame_count,
        scalar_sample_count=frame_count * channels,
        has_frames=bool(frame_count),
        duration_numerator=frame_count,
        time_denominator=rate,
        fact_declared_sample_count=fact,
        fact_count_equals_frame_count=fact == frame_count if fact is not None else None,
        chunks=chunks,
        frames=frames,
        offset=request.offset,
        has_more=request.offset + len(frames) < frame_count,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_wav_pcm",
    kind="plugin",
    description="Validate bounded RIFF/WAVE PCM/IEEE framing and every sample, then return exact channel-interleaved frame pages with source offsets.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
