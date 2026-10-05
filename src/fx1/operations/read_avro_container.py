"""Parse a bounded Avro object container with a flat primitive record schema.

Magic Obj\x01, blocked metadata maps (including negative-count sized blocks),
canonical signed-64 zigzag varints, block extents and every sync marker are
checked. Null and raw DEFLATE codecs are supported; inflated blocks are capped
before parsing. A header-only file is accepted as an empty container; data
blocks must have positive counts. Primitive fields are null/boolean/int/long/
float/double/bytes/string. Names, order and all fields are supplied by the writer
schema. This subset rejects unions, nested/named/logical types, aliases, defaults,
field order metadata and custom schema attributes instead of resolving them.

Integers are exact decimal strings; finite floats use float.hex; bytes are hex;
text is strict UTF-8. Every returned cell includes its original encoded bytes
within its expanded block. Row lineage has the compressed block's source span,
block index, expanded-block span and an absolute row span only for the null
codec. Sync markers detect framing,
not authentication. No reader-schema evolution or custom codec execution.

Limits: 8 MB source, 64 metadata entries/128 KB values, 32768 schema bytes,
64 fields, 50000 records, 1M cells, 1000 data blocks, 8 MB expanded/block and
32 MB expanded total, 65536 bytes/string or bytes field; 200 rows, 10000 cells
and 512 KB encoded page. Every block/row is validated before page output.
Reference: https://avro.apache.org/docs/1.12.0/specification/
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from typing import Any, Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_PRIMITIVES = {"null", "boolean", "int", "long", "float", "double", "bytes", "string"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=50_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Column(OutputModel):
    name: str
    primitive_type: str


class Metadata(OutputModel):
    key: str
    value_bytes: int
    value_sha256: str


class Cell(OutputModel):
    value: str | bool | None
    encoded_hex: str


class Row(OutputModel):
    row_index: int
    block_index: int
    row_in_block: int
    block_source_byte_start: int
    block_source_byte_length: int
    expanded_byte_start: int
    expanded_byte_length: int
    source_byte_start: int | None
    cells: list[Cell]


class Output(OutputModel):
    record_fullname: str
    columns: list[Column]
    codec: Literal["null", "deflate"]
    metadata: list[Metadata]
    schema_sha256: str
    sync_marker_hex: str
    header_bytes: int
    data_block_count: int
    record_count: int
    has_records: bool
    expanded_record_bytes: int
    rows: list[Row]
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    sync_marker_authenticates_source: Literal[False] = False
    schema_resolution_performed: Literal[False] = False
    source_bytes: int
    source_sha256: str


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data, self.position = data, 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.position + size > len(self.data):
            raise ValueError("Avro truncated or negative-length field")
        start = self.position
        self.position += size
        return self.data[start : self.position]

    def long(self) -> int:
        value = 0
        for index in range(10):
            byte = self.take(1)[0]
            if index == 9 and byte > 1:
                raise ValueError("Avro zigzag integer exceeds signed 64-bit encoding")
            value |= (byte & 127) << (index * 7)
            if not byte & 128:
                if index and byte == 0:
                    raise ValueError("Avro overlong noncanonical varint is unsupported")
                return (value >> 1) ^ -(value & 1)
        raise ValueError("Avro unterminated zigzag integer")

    def blob(self, maximum: int) -> bytes:
        size = self.long()
        if not 0 <= size <= maximum:
            raise ValueError("Avro byte/string length exceeds its bound")
        return self.take(size)


def _metadata(cursor: _Cursor) -> dict[str, bytes]:
    metadata: dict[str, bytes] = {}
    total = blocks = 0
    while True:
        count = cursor.long()
        if count == 0:
            return metadata
        blocks += 1
        if blocks > 64 or abs(count) + len(metadata) > 64:
            raise ValueError("Avro metadata exceeds 64 entries/blocks")
        declared_end: int | None = None
        if count < 0:
            count = -count
            size = cursor.long()
            if not 0 <= size <= 160_000:
                raise ValueError("Avro metadata block byte size is out of bounds")
            declared_end = cursor.position + size
        for _ in range(count):
            key = cursor.blob(256).decode("utf-8", errors="strict")
            if not key or key in metadata:
                raise ValueError("Avro metadata names must be nonempty and unique")
            value = cursor.blob(131_072)
            total += len(value)
            if total > 131_072:
                raise ValueError("Avro metadata values exceed 128 KB")
            metadata[key] = value
        if declared_end is not None and cursor.position != declared_end:
            raise ValueError("Avro metadata block size differs from its encoded entries")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Avro schema JSON has duplicate keys")
        result[key] = value
    return result


def _schema(data: bytes) -> tuple[str, list[Column]]:
    if len(data) > 32_768:
        raise ValueError("Avro writer schema exceeds 32768 bytes")
    text = data.decode("utf-8", errors="strict")
    # Bound nesting before the JSON decoder sees an adversarial schema.
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
            if depth > 8:
                raise ValueError("Avro schema JSON nesting exceeds eight levels")
        elif char in "]}":
            depth -= 1
    schema = json.loads(
        text,
        object_pairs_hook=_json_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Avro schema contains {value}")
        ),
    )
    if (
        not isinstance(schema, dict)
        or set(schema) - {"type", "name", "namespace", "doc", "fields"}
        or schema.get("type") != "record"
    ):
        raise ValueError("Avro writer schema must be a supported flat record")
    name, namespace = schema.get("name"), schema.get("namespace", "")
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 256
        or any(not _NAME.fullmatch(part) for part in name.split("."))
        or name.split(".")[-1] in _PRIMITIVES
    ):
        raise ValueError("Avro record name is invalid")
    if (
        not isinstance(namespace, str)
        or len(namespace) > 256
        or namespace
        and any(not _NAME.fullmatch(part) for part in namespace.split("."))
    ):
        raise ValueError("Avro record namespace is invalid")
    fields = schema.get("fields")
    if not isinstance(fields, list) or not 1 <= len(fields) <= 64:
        raise ValueError("Avro flat records require 1..64 fields")
    columns: list[Column] = []
    names: set[str] = set()
    for field in fields:
        if not isinstance(field, dict) or set(field) - {"name", "type", "doc"}:
            raise ValueError("Avro field contains unsupported defaults/aliases/schema attributes")
        field_name, field_type = field.get("name"), field.get("type")
        if (
            not isinstance(field_name, str)
            or len(field_name) > 128
            or not _NAME.fullmatch(field_name)
            or field_name in names
        ):
            raise ValueError("Avro field names must be unique valid identifiers")
        if isinstance(field_type, dict) and set(field_type) == {"type"}:
            field_type = field_type["type"]
        if not isinstance(field_type, str) or field_type not in _PRIMITIVES:
            raise ValueError(
                "Avro field requires a primitive type; unions/logical/nested types are unsupported"
            )
        names.add(field_name)
        columns.append(Column(name=field_name, primitive_type=field_type))
    for item in [schema, *fields]:
        if "doc" in item:
            if not isinstance(item["doc"], str) or len(item["doc"]) > 4096:
                raise ValueError("Avro schema doc must be a string of at most 4096 characters")
            item["doc"].encode("utf-8", errors="strict")
    return name if "." in name or not namespace else f"{namespace}.{name}", columns


def _primitive(cursor: _Cursor, kind: str) -> str | bool | None:
    if kind == "null":
        return None
    if kind == "boolean":
        boolean_byte = cursor.take(1)[0]
        if boolean_byte not in (0, 1):
            raise ValueError("Avro boolean byte must be zero or one")
        return bool(boolean_byte)
    if kind in {"int", "long"}:
        number = cursor.long()
        if kind == "int" and not -(1 << 31) <= number < (1 << 31):
            raise ValueError("Avro int value exceeds signed 32-bit range")
        return str(number)
    if kind in {"float", "double"}:
        floating: float = struct.unpack(
            "<f" if kind == "float" else "<d", cursor.take(4 if kind == "float" else 8)
        )[0]
        if not math.isfinite(floating):
            raise ValueError("Avro nonfinite floating values are unsupported")
        return floating.hex()
    value = cursor.blob(65_536)
    return value.hex() if kind == "bytes" else value.decode("utf-8", errors="strict")


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".avro",), max_bytes=8_000_000)
    cursor = _Cursor(source)
    if cursor.take(4) != b"Obj\x01":
        raise ValueError("Avro object-container magic is invalid")
    metadata = _metadata(cursor)
    if "avro.schema" not in metadata or any(
        key.startswith("avro.") and key not in {"avro.schema", "avro.codec"} for key in metadata
    ):
        raise ValueError("Avro schema is required and other reserved metadata is unsupported")
    name, columns = _schema(metadata["avro.schema"])
    codec = metadata.get("avro.codec", b"null").decode("ascii")
    if codec not in {"null", "deflate"}:
        raise ValueError("Avro supports null and deflate codecs only")
    if len(columns) * request.limit > 10_000:
        raise ValueError("Avro requested page exceeds 10000 cells")
    sync = cursor.take(16)
    header_size = cursor.position
    blocks = records = expanded_total = page_bytes = 0
    rows: list[Row] = []
    while cursor.position < len(source):
        count, size = cursor.long(), cursor.long()
        if (
            count <= 0
            or size < 0
            or size > 8_000_000
            or blocks >= 1000
            or records + count > 50_000
            or (records + count) * len(columns) > 1_000_000
        ):
            raise ValueError("Avro data block violates count/size/work limits")
        block_start = cursor.position
        packed = cursor.take(size)
        if cursor.take(16) != sync:
            raise ValueError("Avro data block sync marker differs from header")
        if codec == "null":
            expanded = packed
        else:
            decoder = zlib.decompressobj(-15)
            try:
                expanded = decoder.decompress(packed, 8_000_001)
            except zlib.error as error:
                raise ValueError("Avro deflate block is malformed") from error
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError(
                    "Avro deflate block is truncated, overlong or exceeds its expansion bound"
                )
        expanded_total += len(expanded)
        if len(expanded) > 8_000_000 or expanded_total > 32_000_000:
            raise ValueError("Avro expanded data exceeds 8 MB/block or 32 MB total")
        block = _Cursor(expanded)
        for row_in_block in range(count):
            row_start = block.position
            selected = request.offset <= records < request.offset + request.limit
            cells: list[Cell] = []
            for column in columns:
                field_start = block.position
                value = _primitive(block, column.primitive_type)
                if selected:
                    cells.append(
                        Cell(value=value, encoded_hex=expanded[field_start : block.position].hex())
                    )
            if selected:
                row = Row(
                    row_index=records,
                    block_index=blocks,
                    row_in_block=row_in_block,
                    block_source_byte_start=block_start,
                    block_source_byte_length=size,
                    expanded_byte_start=row_start,
                    expanded_byte_length=block.position - row_start,
                    source_byte_start=block_start + row_start if codec == "null" else None,
                    cells=cells,
                )
                page_bytes += len(json.dumps(row.model_dump(), ensure_ascii=False).encode("utf-8"))
                if page_bytes > 512_000:
                    raise ValueError("Avro encoded page exceeds 512 KB")
                rows.append(row)
            records += 1
        if block.position != len(expanded):
            raise ValueError("Avro expanded block has bytes beyond its declared records")
        blocks += 1
    return Output(
        record_fullname=name,
        columns=columns,
        codec="null" if codec == "null" else "deflate",
        metadata=[
            Metadata(
                key=key, value_bytes=len(value), value_sha256=hashlib.sha256(value).hexdigest()
            )
            for key, value in metadata.items()
        ],
        schema_sha256=hashlib.sha256(metadata["avro.schema"]).hexdigest(),
        sync_marker_hex=sync.hex(),
        header_bytes=header_size,
        data_block_count=blocks,
        record_count=records,
        has_records=bool(records),
        expanded_record_bytes=expanded_total,
        rows=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < records,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_avro_container",
    kind="plugin",
    description="Validate bounded Avro primitive-record object containers, schema/framing/deflate limits and exact paged scalar values with block lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
