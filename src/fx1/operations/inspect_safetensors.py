"""Inspect safetensors layout without allocating or deserializing tensor arrays.

Format: https://github.com/huggingface/safetensors#format
Dtype widths/alignment: safetensors/src/tensor.rs, Dtype::bitsize and Metadata::validate.
The unsigned 8-byte little-endian prefix bounds a strict UTF-8 JSON header.
Duplicate keys, unknown tensor fields/dtypes, invalid shapes, payload holes,
overlaps, extra trailing payload and non-byte-aligned packed tensors are rejected.
Empty dimensions and scalar shape [] are supported; zero-byte tensors must lie
at payload boundaries in sorted offset order. Tensor contents (including finite
values and BOOL byte conventions) are never interpreted.

Header mode reads at most 8 MB plus the length prefix from a contained file up
to 1 TB. Full mode reads at most 16 MB and returns a whole-file hash. Header
hashes always cover the original serialized JSON bytes, including space padding,
and are not a tensor-payload fingerprint. Metadata checks do not promise a
snapshot. Limits: 10000 tensors, rank 32, signed-63-bit shape products, 256-byte
names, 128 string metadata entries/256 KB metadata, and 200 preview tensors.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
    "C64": 64,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    read_mode: Literal["header", "full"] = "header"
    max_header_bytes: int = Field(default=1_000_000, strict=True, ge=2, le=8_000_000)
    max_file_bytes: int = Field(default=1_000_000_000_000, strict=True, ge=10, le=1_000_000_000_000)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Tensor(OutputModel):
    name: str
    dtype: str
    shape: list[int]
    elements: int
    bits_per_element: int
    payload_start: int
    payload_end: int
    absolute_file_start: int
    data_bytes: int


class Output(OutputModel):
    tensor_count: int
    scalar_tensor_count: int
    empty_tensor_count: int
    tensors: list[Tensor]
    dtype_counts: dict[str, int]
    metadata: dict[str, str]
    offset: int
    has_more: bool
    payload_bytes: int
    declared_layout_complete: Literal[True] = True
    tensor_values_validated: Literal[False] = False
    tensors_deserialized: Literal[False] = False
    snapshot_guaranteed: Literal[False] = False
    read_mode: Literal["header", "full"]
    bytes_read: int
    header_bytes: int
    header_sha256: str
    header_hash_scope: Literal["serialized_json_header_including_space_padding"] = (
        "serialized_json_header_including_space_padding"
    )
    source_bytes: int
    source_sha256: str | None


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate safetensors header JSON key")
        result[key] = value
    return result


def _integer(value: str) -> int:
    if len(value) > 19:
        raise ValueError("safetensors integer exceeds the signed-63-bit representation bound")
    number = int(value)
    if not 0 <= number <= (1 << 63) - 1:
        raise ValueError("safetensors shapes and offsets must be bounded nonnegative integers")
    return number


def _reject_number(value: str) -> None:
    raise ValueError(
        "safetensors headers support integer dimensions/offsets, not floating-point constants"
    )


def _header_size(prefix: bytes, source_bytes: int, limit: int) -> int:
    if len(prefix) != 8:
        raise ValueError("safetensors source lacks its eight-byte header length")
    length = int.from_bytes(prefix, "little")
    if not 2 <= length <= limit or length > source_bytes - 8:
        raise ValueError("safetensors header length exceeds its source or requested bound")
    return length


def _parse(header: bytes) -> dict[str, Any]:
    if not header.startswith(b"{"):
        raise ValueError("safetensors JSON header must begin with an opening object brace")
    # The only supported nested structures are tensor objects and shape/offset
    # arrays. Check nesting before JSON's recursive parser allocates them.
    quoted = escaped = False
    depth = 0
    for byte in header:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 4:
                raise ValueError("safetensors header exceeds four JSON nesting levels")
        elif byte in (93, 125):
            depth -= 1
    try:
        decoder = json.JSONDecoder(
            object_pairs_hook=_object,
            parse_int=_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
        decoded = header.decode("utf-8")
        document, end = decoder.raw_decode(decoded)
        if any(character != " " for character in decoded[end:]):
            raise ValueError("safetensors trailing header padding must consist only of spaces")
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid strict safetensors JSON header") from exc
    if not isinstance(document, dict) or len(document) > 10_001:
        raise ValueError("safetensors header exceeds 10000 tensors plus metadata")
    return document


def execute(request: Input, context: OperationContext) -> Output:
    source_hash: str | None = None
    if request.read_mode == "full":
        content = context.read_bytes(
            request.path,
            suffixes=(".safetensors",),
            max_bytes=min(request.max_file_bytes, 16_000_000),
        )
        source_bytes = len(content)
        size = _header_size(content[:8], source_bytes, request.max_header_bytes)
        header = content[8 : 8 + size]
        source_hash = hashlib.sha256(content).hexdigest()
        bytes_read = source_bytes
    else:
        with context.open_ranges(
            request.path,
            suffixes=(".safetensors",),
            max_file_bytes=request.max_file_bytes,
            max_read_bytes=request.max_header_bytes + 8,
        ) as ranges:
            source_bytes = ranges.source_bytes
            if source_bytes < 10:
                raise ValueError("safetensors source is shorter than its minimal header")
            size = _header_size(ranges.read_at(0, 8), source_bytes, request.max_header_bytes)
            header = ranges.read_at(8, size)
            ranges.check_unchanged_metadata()
            bytes_read = ranges.bytes_read
    document = _parse(header)
    metadata = document.pop("__metadata__", {})
    if not isinstance(metadata, dict) or len(metadata) > 128:
        raise ValueError("safetensors metadata must be a string map with at most 128 entries")
    metadata_bytes = 0
    for key, value in metadata.items():
        if not isinstance(value, str) or len(key) > 256 or len(value) > 4096:
            raise ValueError("safetensors metadata requires bounded string keys and values")
        metadata_bytes += len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if metadata_bytes > 256_000:
            raise ValueError("safetensors metadata exceeds 256 KB UTF-8 strings")
    if len(document) > 10_000:
        raise ValueError("safetensors header exceeds 10000 tensors")
    tensors: list[Tensor] = []
    counts: dict[str, int] = {}
    scalar_count = empty_count = 0
    payload_bytes = source_bytes - 8 - size
    for name, info in document.items():
        if not 1 <= len(name.encode("utf-8")) <= 256:
            raise ValueError("safetensors tensor names require 1 through 256 UTF-8 bytes")
        if not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("tensor metadata requires exactly dtype, shape and data_offsets")
        dtype, shape, offsets = info["dtype"], info["shape"], info["data_offsets"]
        if not isinstance(dtype, str) or dtype not in _BITS:
            raise ValueError("unsupported safetensors dtype")
        if (
            not isinstance(shape, list)
            or len(shape) > 32
            or any(type(value) is not int for value in shape)
        ):
            raise ValueError(
                "safetensors shape must be a list of at most 32 strict integer dimensions"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError("safetensors data_offsets requires exactly two strict integers")
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > (1 << 63) - 1:
                raise ValueError("safetensors shape product exceeds signed 63-bit arithmetic")
        bits = elements * _BITS[dtype]
        if bits > (1 << 63) - 1 or bits % 8:
            raise ValueError("safetensors tensor size overflows or ends within a packed byte")
        start, end = offsets
        if not 0 <= start <= end <= payload_bytes or end - start != bits // 8:
            raise ValueError("safetensors dtype/shape does not match its bounded data offsets")
        tensors.append(
            Tensor(
                name=name,
                dtype=dtype,
                shape=shape,
                elements=elements,
                bits_per_element=_BITS[dtype],
                payload_start=start,
                payload_end=end,
                absolute_file_start=8 + size + start,
                data_bytes=end - start,
            )
        )
        counts[dtype] = counts.get(dtype, 0) + 1
        scalar_count += int(not shape)
        empty_count += int(elements == 0)
    tensors.sort(key=lambda tensor: (tensor.payload_start, tensor.payload_end, tensor.name))
    cursor = 0
    for tensor in tensors:
        if tensor.payload_start != cursor:
            raise ValueError("safetensors payload has a hole, overlap or misplaced empty tensor")
        cursor = tensor.payload_end
    if cursor != payload_bytes:
        raise ValueError("safetensors declared tensors do not account for the complete payload")
    page = tensors[request.offset : request.offset + request.limit]
    return Output(
        tensor_count=len(tensors),
        scalar_tensor_count=scalar_count,
        empty_tensor_count=empty_count,
        tensors=page,
        dtype_counts=counts,
        metadata=metadata,
        offset=request.offset,
        has_more=request.offset + len(page) < len(tensors),
        payload_bytes=payload_bytes,
        read_mode=request.read_mode,
        bytes_read=bytes_read,
        header_bytes=size,
        header_sha256=hashlib.sha256(header).hexdigest(),
        source_bytes=source_bytes,
        source_sha256=source_hash,
    )


OPERATION = Operation(
    id="plugins.inspect_safetensors",
    kind="plugin",
    description="Inspect bounded safetensors JSON headers, dtypes, shapes and complete nonoverlapping payload layout without tensor deserialization; distinguish header-range hashes from optional bounded whole-file hashes.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
