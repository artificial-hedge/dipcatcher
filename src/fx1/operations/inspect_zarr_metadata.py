"""Inspect one strict Zarr v2 array metadata document without opening chunk data.

Accept a .zarray file or explicit .json export, version2 and required standard
keys, plus optional dimension_separator. Unknown keys/v3/consolidated/group
metadata are rejected. Dtypes are explicit-endian fixed-size bool, integer and
float (b1/i1,2,4,8/u1,2,4,8/f2,4,8); object/structured/string/temporal/complex
types fail. Rank1..8, nonnegative shape and positive chunk lengths are required.

Fill declarations remain exact JSON decimals/integers or booleans; float special
strings NaN/Infinity/-Infinity are recognized. Null means missing-chunk contents
are undefined. Numeric fill magnitudes must fit the declared finite type range,
but float quantization/underflow is not performed or certified. Codec/filter
objects require an id and bounded strict JSON; their canonical JSON declarations
are returned without importing codecs or validating codec-specific parameters.

Chunk coordinates are enumerated in lexicographic order, last axis fastest,
independently of C/F byte order within each chunk. Edge chunks retain full nominal
storage size; intersected logical bounds and overhang are reported. Empty axes
produce no chunks. No chunk existence, compression ratio, decoded allocation,
store completeness, attribute semantics or external-reference validity is proved.

Limits:256 KB source, JSON depth16/4096 nodes,128-digit number tokens,8192-byte
strings,16 filters/32768 bytes per codec declaration, dimensions<=1e9, nominal
chunk/array byte sizes and grid count<=2^63-1, pages<=200 chunk descriptors.
Reference: https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal
from pathlib import PurePath
from typing import Any, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=(1 << 63) - 1)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Codec(OutputModel):
    id: str
    configuration_json: str
    codec_parameters_validated: Literal[False] = False


class Fill(OutputModel):
    kind: Literal["undefined", "boolean", "integer", "decimal", "nonfinite_float"]
    exact_declaration: str | None
    dtype_quantization_performed: Literal[False] = False


class Chunk(OutputModel):
    grid_index: str
    coordinate: list[int]
    key: str
    logical_start: list[int]
    logical_stop: list[int]
    logical_shape: list[int]
    logical_elements: str
    nominal_elements: str
    overhang_elements: str


class Output(OutputModel):
    zarr_format: Literal[2] = 2
    shape: list[int]
    chunk_shape: list[int]
    dtype: str
    dtype_kind: Literal["boolean", "signed_integer", "unsigned_integer", "float"]
    item_bytes: int
    byte_order: Literal["little", "big", "not_applicable"]
    chunk_memory_order: Literal["C", "F"]
    dimension_separator: Literal[".", "/"]
    logical_elements: str
    logical_array_bytes: str
    nominal_chunk_elements: str
    nominal_chunk_bytes: str
    chunk_grid_shape: list[int]
    chunk_grid_count: str
    array_is_empty: bool
    fill: Fill
    compressor: Codec | None
    filters: list[Codec]
    chunk_page: list[Chunk]
    offset: str
    has_more: bool
    chunk_page_order: Literal["lexicographic_last_axis_fastest"] = "lexicographic_last_axis_fastest"
    chunk_files_read: Literal[False] = False
    codec_implementations_loaded: Literal[False] = False
    external_references_followed: Literal[False] = False
    store_completeness_verified: Literal[False] = False
    metadata_subset_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Zarr JSON has duplicate keys")
        result[key] = value
    return result


def _integer(token: str) -> int:
    if len(token) > 128:
        raise ValueError("Zarr integer token exceeds128 bytes")
    return int(token)


def _decimal(token: str) -> Decimal:
    if len(token) > 128:
        raise ValueError("Zarr decimal token exceeds128 bytes")
    exponent = re.split("[eE]", token)[1:]
    if exponent and (len(exponent[0]) > 6 or abs(int(exponent[0])) > 10_000):
        raise ValueError("Zarr decimal exponent exceeds10000")
    return Decimal(token)


def _constant(token: str) -> None:
    raise ValueError(f"Zarr nonstandard JSON constant {token} is unsupported")


def _encode(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(
                json.dumps(key, ensure_ascii=False) + ":" + _encode(value[key])
                for key in sorted(value)
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _document(source: bytes) -> dict[str, Any]:
    text = source.decode("utf-8", errors="strict")
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 16:
                raise ValueError("Zarr JSON nesting exceeds16")
        elif char in "]}":
            depth -= 1
    document = json.loads(
        text,
        object_pairs_hook=_pairs,
        parse_int=_integer,
        parse_float=_decimal,
        parse_constant=_constant,
    )
    pending = [document]
    nodes = 0
    while pending:
        value = pending.pop()
        nodes += 1
        if nodes > 4096:
            raise ValueError("Zarr JSON exceeds4096 nodes including object keys")
        if isinstance(value, str):
            if len(value.encode("utf-8", errors="strict")) > 8192:
                raise ValueError("Zarr string exceeds8192 UTF-8 bytes")
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if not isinstance(document, dict):
        raise ValueError("Zarr array metadata must be a JSON object")
    return document


def _codec(value: Any) -> Codec:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("id"), str)
        or not 1 <= len(value["id"]) <= 128
    ):
        raise ValueError("Zarr codec must be an object with a bounded nonempty id string")
    encoded = _encode(value)
    if len(encoded.encode("utf-8")) > 32_768:
        raise ValueError("Zarr codec configuration exceeds32768 bytes")
    return Codec(id=value["id"], configuration_json=encoded)


def _fill(value: Any, kind: str, width: int) -> Fill:
    if value is None:
        return Fill(kind="undefined", exact_declaration=None)
    if kind == "b":
        if type(value) is not bool:
            raise ValueError("Zarr bool fill requires a JSON boolean")
        return Fill(kind="boolean", exact_declaration="true" if value else "false")
    if kind in {"i", "u"}:
        if type(value) is not int:
            raise ValueError("Zarr integer fill requires a JSON integer")
        low = 0 if kind == "u" else -(1 << (8 * width - 1))
        high = (1 << (8 * width - (0 if kind == "u" else 1))) - 1
        if not low <= value <= high:
            raise ValueError("Zarr integer fill is outside dtype range")
        return Fill(kind="integer", exact_declaration=str(value))
    if isinstance(value, str) and value in {"NaN", "Infinity", "-Infinity"}:
        return Fill(kind="nonfinite_float", exact_declaration=value)
    if type(value) is not int and not isinstance(value, Decimal):
        raise ValueError("Zarr float fill requires a JSON number or specified special string")
    decimal = Decimal(value)
    maximum = (
        65504 if width == 2 else ((1 << 24) - 1) << 104 if width == 4 else ((1 << 53) - 1) << 971
    )
    if decimal.copy_abs() > Decimal(maximum):
        raise ValueError("Zarr finite fill exceeds dtype magnitude range")
    return Fill(kind="decimal", exact_declaration=str(decimal))


def execute(request: Input, context: OperationContext) -> Output:
    name = PurePath(request.path).name
    if name != ".zarray" and not name.lower().endswith(".json"):
        raise ValueError("Zarr metadata path must name .zarray or a .json export")
    source = context.read_bytes(request.path, suffixes=("", ".json"), max_bytes=256_000)
    metadata = _document(source)
    required = {
        "zarr_format",
        "shape",
        "chunks",
        "dtype",
        "compressor",
        "fill_value",
        "order",
        "filters",
    }
    if not required <= metadata.keys() or metadata.keys() - required - {"dimension_separator"}:
        raise ValueError("Zarr metadata has missing/unsupported keys")
    if type(metadata["zarr_format"]) is not int or metadata["zarr_format"] != 2:
        raise ValueError("Zarr inspector supports version2 only")
    shape, chunks = metadata["shape"], metadata["chunks"]
    if (
        not isinstance(shape, list)
        or not isinstance(chunks, list)
        or not 1 <= len(shape) <= 8
        or len(shape) != len(chunks)
    ):
        raise ValueError("Zarr shape/chunks must have matching ranks1..8")
    if any(type(value) is not int or not 0 <= value <= 1_000_000_000 for value in shape) or any(
        type(value) is not int or not 1 <= value <= 1_000_000_000 for value in chunks
    ):
        raise ValueError("Zarr dimensions must be bounded integers; chunks must be positive")
    dtype = metadata["dtype"]
    match = re.fullmatch(r"([<>|])([biuf])([1248])", dtype) if isinstance(dtype, str) else None
    if match is None:
        raise ValueError("Zarr dtype requires a supported explicit-endian primitive type")
    endian, kind, width_text = match.groups()
    width = int(width_text)
    if (
        kind == "b"
        and width != 1
        or kind == "f"
        and width not in {2, 4, 8}
        or width > 1
        and endian == "|"
    ):
        raise ValueError("Zarr dtype width/byte-order combination is unsupported")
    order, separator = metadata["order"], metadata.get("dimension_separator", ".")
    if order not in ("C", "F") or separator not in (".", "/"):
        raise ValueError("Zarr order or dimension separator is invalid")
    grid = [(length + chunk - 1) // chunk for length, chunk in zip(shape, chunks, strict=True)]
    elements, chunk_elements, grid_count = math.prod(shape), math.prod(chunks), math.prod(grid)
    if max(elements * width, chunk_elements * width, grid_count) > (1 << 63) - 1:
        raise ValueError("Zarr nominal array/chunk bytes or grid count exceeds signed64 range")
    fill = _fill(metadata["fill_value"], kind, width)
    compressor = None if metadata["compressor"] is None else _codec(metadata["compressor"])
    filters = metadata["filters"]
    if filters is None:
        filters = []
    if not isinstance(filters, list) or len(filters) > 16:
        raise ValueError("Zarr filters require null or a list of at most16 codec declarations")
    codecs = [_codec(value) for value in filters]
    page: list[Chunk] = []
    for index in range(request.offset, min(grid_count, request.offset + request.limit)):
        remainder = index
        coordinate = [0] * len(grid)
        for axis in range(len(grid) - 1, -1, -1):
            remainder, coordinate[axis] = divmod(remainder, grid[axis])
        starts = [point * chunk for point, chunk in zip(coordinate, chunks, strict=True)]
        stops = [
            min(start + chunk, length)
            for start, chunk, length in zip(starts, chunks, shape, strict=True)
        ]
        logical_shape = [stop - start for start, stop in zip(starts, stops, strict=True)]
        count = math.prod(logical_shape)
        page.append(
            Chunk(
                grid_index=str(index),
                coordinate=coordinate,
                key=separator.join(map(str, coordinate)),
                logical_start=starts,
                logical_stop=stops,
                logical_shape=logical_shape,
                logical_elements=str(count),
                nominal_elements=str(chunk_elements),
                overhang_elements=str(chunk_elements - count),
            )
        )
    return Output(
        shape=shape,
        chunk_shape=chunks,
        dtype=dtype,
        dtype_kind="boolean"
        if kind == "b"
        else "signed_integer"
        if kind == "i"
        else "unsigned_integer"
        if kind == "u"
        else "float",
        item_bytes=width,
        byte_order="not_applicable" if width == 1 else "little" if endian == "<" else "big",
        chunk_memory_order=order,
        dimension_separator=separator,
        logical_elements=str(elements),
        logical_array_bytes=str(elements * width),
        nominal_chunk_elements=str(chunk_elements),
        nominal_chunk_bytes=str(chunk_elements * width),
        chunk_grid_shape=grid,
        chunk_grid_count=str(grid_count),
        array_is_empty=elements == 0,
        fill=fill,
        compressor=compressor,
        filters=codecs,
        chunk_page=page,
        offset=str(request.offset),
        has_more=request.offset + len(page) < grid_count,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.inspect_zarr_metadata",
    kind="plugin",
    description="Inspect strict Zarr v2 primitive array metadata, exact fill declarations and bounded chunk geometry without loading codecs or chunk data.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
