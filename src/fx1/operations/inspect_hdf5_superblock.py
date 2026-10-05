"""Inspect checksummed HDF5 superblocks (versions 2/3), not the object graph.

Search signature positions 0,512,1024,... within the supplied bounded file.
Validate declared offset/length widths (2/4/8 bytes), Jenkins lookup3 checksum,
superblock extent, absolute EOF and relative root/extension address bounds.
The recorded base must equal the located signature; relocated wrappers and
versions 0/1 are explicitly unsupported. Extra physical bytes after declared
HDF5 EOF are reported, not interpreted. Header-only address checks assume a
single physical file; no virtual driver mapping or superblock extension is read.

Version-2 consistency flags are uninterpreted; version-3 reserved bits fail,
and known write/SWMR flags are observations. Even zero flags do not establish a
closed, recovered, consistent or immutable file. No linked objects, datasets,
external files, filters, user blocks or object headers are decoded. Checksum is
corruption detection, not authentication, source trust or complete HDF5 validity.

Header mode accepts files through 1 TB and reads at most 1024 bytes in bounded
ranges. Full mode accepts 8 MB and additionally hashes the whole original file.
Superblock SHA-256 covers only its original bytes including stored checksum;
source SHA-256 is null in header mode. Descriptor metadata checks cannot promise
an atomic snapshot. References: HDF Group File Format Specification v3.0 II.A;
HDFGroup/hdf5 src/H5checksum.c (public-domain Bob Jenkins lookup3 algorithm).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_MASK = (1 << 32) - 1


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    read_mode: Literal["header", "full"] = "header"
    max_file_bytes: int = Field(default=1_000_000_000_000, strict=True, ge=24, le=1_000_000_000_000)


class Output(OutputModel):
    superblock_version: int
    signature_byte_start: int
    superblock_bytes: int
    signature_probes: int
    offset_width: int
    length_width: int
    observed_consistency_flags: int
    observed_write_access_flag: bool | None
    observed_swmr_write_flag: bool | None
    recorded_base_address: int
    declared_absolute_eof: int
    physical_bytes_after_declared_eof: int
    root_group_relative_address: int
    root_group_absolute_address: int
    extension_relative_address: int | None
    extension_absolute_address: int | None
    stored_lookup3_checksum_hex: str
    checksum_matches: Literal[True] = True
    superblock_fields_validated: Literal[True] = True
    address_interpretation: Literal["single_physical_file_base_plus_relative_objects"] = (
        "single_physical_file_base_plus_relative_objects"
    )
    virtual_driver_mapping_verified: Literal[False] = False
    object_graph_validated: Literal[False] = False
    full_hdf5_validity_verified: Literal[False] = False
    committed_state_verified: Literal[False] = False
    snapshot_guaranteed: Literal[False] = False
    read_mode: Literal["header", "full"]
    bytes_read: int
    superblock_sha256: str
    superblock_hash_scope: Literal["original_superblock_including_checksum"] = (
        "original_superblock_including_checksum"
    )
    source_bytes: int
    source_sha256: str | None


def _rotate(value: int, distance: int) -> int:
    return ((value << distance) | (value >> (32 - distance))) & _MASK


def _lookup3(data: bytes) -> int:
    """32-bit little-endian lookup3 with initval zero, implemented on words."""
    state = [(0xDEADBEEF + len(data)) & _MASK] * 3
    position = 0
    mix_steps = (
        (0, 2, 1, 4),
        (1, 0, 2, 6),
        (2, 1, 0, 8),
        (0, 2, 1, 16),
        (1, 0, 2, 19),
        (2, 1, 0, 4),
    )
    while len(data) - position > 12:
        for word in range(3):
            start = position + 4 * word
            state[word] = (state[word] + int.from_bytes(data[start : start + 4], "little")) & _MASK
        for target, source, other, rotation in mix_steps:
            state[target] = (
                (state[target] - state[source]) ^ _rotate(state[source], rotation)
            ) & _MASK
            state[source] = (state[source] + state[other]) & _MASK
        position += 12
    remaining = data[position:]
    for index, byte in enumerate(remaining):
        word, shift = divmod(index, 4)
        state[word] = (state[word] + (byte << (8 * shift))) & _MASK
    if remaining:
        final_steps = (
            (2, 1, 14),
            (0, 2, 11),
            (1, 0, 25),
            (2, 1, 16),
            (0, 2, 4),
            (1, 0, 14),
            (2, 1, 24),
        )
        for target, source, rotation in final_steps:
            state[target] = (
                (state[target] ^ state[source]) - _rotate(state[source], rotation)
            ) & _MASK
    return state[2]


def _header(read: Callable[[int, int], bytes], size: int) -> tuple[bytes, int, int]:
    offset = probes = 0
    while offset + 8 <= size:
        probes += 1
        if read(offset, 8) == _SIGNATURE:
            break
        offset = 512 if offset == 0 else offset * 2
    else:
        raise ValueError("HDF5 signature was not found at an allowed superblock position")
    if offset + 12 > size:
        raise ValueError("HDF5 superblock fixed header is truncated")
    prefix = read(offset, 12)
    version, offset_width, length_width = prefix[8:11]
    if version not in (2, 3):
        raise ValueError("HDF5 inspector supports checksummed superblock versions 2/3 only")
    if offset_width not in (2, 4, 8) or length_width not in (2, 4, 8):
        raise ValueError("HDF5 offset and length widths must be 2, 4 or 8 bytes")
    length = 16 + 4 * offset_width
    if offset + length > size:
        raise ValueError("HDF5 superblock address/checksum fields are truncated")
    header = read(offset, length)
    if header[:12] != prefix:
        raise ValueError("HDF5 superblock changed during inspection")
    return header, offset, probes


def execute(request: Input, context: OperationContext) -> Output:
    source_hash: str | None = None
    if request.read_mode == "full":
        content = context.read_bytes(
            request.path,
            suffixes=(".h5", ".hdf5", ".hdf"),
            max_bytes=min(request.max_file_bytes, 8_000_000),
        )
        header, offset, probes = _header(
            lambda start, length: content[start : start + length], len(content)
        )
        source_size = bytes_read = len(content)
        source_hash = hashlib.sha256(content).hexdigest()
    else:
        with context.open_ranges(
            request.path,
            suffixes=(".h5", ".hdf5", ".hdf"),
            max_file_bytes=request.max_file_bytes,
            max_read_bytes=1024,
        ) as ranges:
            header, offset, probes = _header(ranges.read_at, ranges.source_bytes)
            ranges.check_unchanged_metadata()
            source_size, bytes_read = ranges.source_bytes, ranges.bytes_read
    version, width, length_width, flags = header[8:12]
    checksum = int.from_bytes(header[-4:], "little")
    if _lookup3(header[:-4]) != checksum:
        raise ValueError("HDF5 superblock lookup3 checksum mismatch")
    if version == 3 and (flags & ~5 or flags & 4 and not flags & 1):
        raise ValueError("HDF5 version-3 consistency flags use reserved or contradictory bits")
    base, extension, eof, root = [
        int.from_bytes(header[12 + index * width : 12 + (index + 1) * width], "little")
        for index in range(4)
    ]
    undefined = (1 << (8 * width)) - 1
    if base == undefined or base != offset:
        raise ValueError(
            "HDF5 base must equal the found superblock position; relocation is unsupported"
        )
    if eof == undefined or not offset + len(header) <= eof <= source_size:
        raise ValueError("HDF5 declared absolute EOF is truncated or precedes the superblock end")
    if root == undefined or not offset + len(header) <= base + root < eof:
        raise ValueError(
            "HDF5 root object address is undefined or outside the declared data extent"
        )
    extension_address = None if extension == undefined else base + extension
    if extension_address is not None and not offset + len(header) <= extension_address < eof:
        raise ValueError("HDF5 extension address is outside the declared data extent")
    return Output(
        superblock_version=version,
        signature_byte_start=offset,
        superblock_bytes=len(header),
        signature_probes=probes,
        offset_width=width,
        length_width=length_width,
        observed_consistency_flags=flags,
        observed_write_access_flag=bool(flags & 1) if version == 3 else None,
        observed_swmr_write_flag=bool(flags & 4) if version == 3 else None,
        recorded_base_address=base,
        declared_absolute_eof=eof,
        physical_bytes_after_declared_eof=source_size - eof,
        root_group_relative_address=root,
        root_group_absolute_address=base + root,
        extension_relative_address=None if extension == undefined else extension,
        extension_absolute_address=extension_address,
        stored_lookup3_checksum_hex=f"{checksum:08x}",
        read_mode=request.read_mode,
        bytes_read=bytes_read,
        superblock_sha256=hashlib.sha256(header).hexdigest(),
        source_bytes=source_size,
        source_sha256=source_hash,
    )


OPERATION = Operation(
    id="plugins.inspect_hdf5_superblock",
    kind="plugin",
    description="Inspect checksummed HDF5 v2/v3 superblock declarations and bounded address extents, with explicit separation from unchecked object-graph and file-state validity.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
