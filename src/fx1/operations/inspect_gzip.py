"""Validate concatenated gzip members with bounded streaming decompression.

Every byte must belong to a valid gzip member: trailing junk and zero padding
are rejected. zlib validates header integrity, DEFLATE data, CRC32 and ISIZE.
Uncompressed content is hashed incrementally and discarded, never extracted.
The output limit applies across all members, with at most 64 KiB emitted by
one decompression call. Hashes attest these bytes, not their semantic truth.
"""

from __future__ import annotations

import hashlib
import zlib
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    max_uncompressed_bytes: int = Field(default=16_000_000, strict=True, ge=0, le=64_000_000)


class Member(OutputModel):
    index: int
    compressed_offset: int
    compressed_bytes: int
    uncompressed_bytes: int
    compressed_sha256: str
    uncompressed_sha256: str


class Output(OutputModel):
    member_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    expansion_ratio: float
    max_uncompressed_bytes: int
    stream_validated: Literal[True] = True
    crc_and_size_validated: Literal[True] = True
    trailing_policy: Literal["reject_all_nonmember_bytes"] = "reject_all_nonmember_bytes"
    concatenated_members_allowed: Literal[True] = True
    members: list[Member]
    source_sha256: str
    uncompressed_sha256: str


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(
        request.path, suffixes=(".gz", ".gzip", ".tgz"), max_bytes=8_000_000
    )
    if not content:
        raise ValueError("gzip stream must contain at least one complete member")
    offset = total_output = 0
    members: list[Member] = []
    all_output_hash = hashlib.sha256()
    while offset < len(content):
        if len(members) >= 100:
            raise ValueError("gzip stream exceeds 100 members")
        if content[offset : offset + 2] != b"\x1f\x8b":
            raise ValueError(f"nonmember or trailing bytes at compressed offset {offset}")
        start = offset
        decoder = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        member_hash = hashlib.sha256()
        member_output = 0
        pending = b""
        try:
            while not decoder.eof:
                if not pending:
                    pending = content[offset : offset + 65_536]
                    offset += len(pending)
                # Probe one byte beyond the remaining allowance, including a
                # zero-byte allowance, without ever allocating the whole output.
                allowance = min(65_536, request.max_uncompressed_bytes - total_output + 1)
                chunk = decoder.decompress(pending, max_length=allowance)
                total_output += len(chunk)
                member_output += len(chunk)
                if total_output > request.max_uncompressed_bytes:
                    raise ValueError(
                        "gzip uncompressed content exceeds the requested aggregate limit"
                    )
                member_hash.update(chunk)
                all_output_hash.update(chunk)
                if decoder.eof:
                    offset -= len(decoder.unused_data)
                    pending = b""
                else:
                    next_pending = decoder.unconsumed_tail
                    if next_pending == pending and not chunk:
                        if not pending and offset >= len(content):
                            raise ValueError("truncated gzip member before its validated trailer")
                        raise ValueError("gzip decoder made no progress")
                    pending = next_pending
        except zlib.error as exc:
            raise ValueError(f"invalid gzip member {len(members)}: {exc}") from exc
        members.append(
            Member(
                index=len(members),
                compressed_offset=start,
                compressed_bytes=offset - start,
                uncompressed_bytes=member_output,
                compressed_sha256=hashlib.sha256(memoryview(content)[start:offset]).hexdigest(),
                uncompressed_sha256=member_hash.hexdigest(),
            )
        )
    return Output(
        member_count=len(members),
        compressed_bytes=len(content),
        uncompressed_bytes=total_output,
        expansion_ratio=total_output / len(content),
        max_uncompressed_bytes=request.max_uncompressed_bytes,
        members=members,
        source_sha256=hashlib.sha256(content).hexdigest(),
        uncompressed_sha256=all_output_hash.hexdigest(),
    )


OPERATION = Operation(
    id="plugins.inspect_gzip",
    kind="plugin",
    description=(
        "Stream-validate all gzip members, including CRC32 and size trailers, while hashing "
        "bounded decompressed chunks without extraction. Enforces 8 MB compressed, up to "
        "64 MB expanded and 100-member limits, and rejects all trailing nonmember bytes."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
