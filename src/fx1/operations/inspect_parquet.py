"""Inspect Parquet metadata with optional bounded footer-only reads of large files.

Layout: https://parquet.apache.org/docs/file-format/
Only the header, serialized metadata, and trailer are read in footer mode. The
metadata-only envelope is parsed in memory; column offsets are never dereferenced.
Compact-Thrift preflight bounds nesting (32), containers (100000), value visits
(200000 including targeted second passes), binary values (1 MB), and the flattened
Parquet schema (4096 nodes, 32 levels, 256 root fields). It removes ARROW:schema
from top-level key/value metadata before PyArrow can restore extension types.
The returned schema derives from Parquet types; Arrow-only distinctions such as
original timezone, extension identity and some list distinctions are not restored.
Hashes always refer to original file/footer bytes, not the sanitized parser input.

Wire layout: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
Fields: https://github.com/apache/parquet-format/blob/master/src/main/thrift/parquet.thrift
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    read_mode: Literal["full", "footer"] = "full"
    max_footer_bytes: int = Field(default=2_000_000, strict=True, ge=1, le=8_000_000)
    max_file_bytes: int = Field(default=1_000_000_000_000, strict=True, ge=12, le=1_000_000_000_000)
    row_group_offset: int = Field(default=0, strict=True, ge=0)
    row_group_limit: int = Field(default=20, strict=True, ge=1, le=100)


class Column(OutputModel):
    name: str
    logical_type: str
    nullable: bool


class RowGroup(OutputModel):
    index: int
    rows: int
    compressed_bytes: int
    uncompressed_bytes: int
    compression_codecs: list[str]


class Output(OutputModel):
    columns: list[Column]
    leaf_column_count: int
    total_rows: int
    row_group_count: int
    row_groups: list[RowGroup]
    row_group_offset: int
    has_more_row_groups: bool
    format_version: str
    source_sha256: str | None
    source_bytes: int
    read_mode: Literal["full", "footer"]
    bytes_read: int
    footer_offset: int
    footer_bytes: int
    footer_sha256: str
    footer_hash_scope: Literal["serialized_file_metadata"] = "serialized_file_metadata"
    snapshot_guaranteed: Literal[False] = False
    schema_origin: Literal["parquet_types_without_serialized_arrow_origin"] = (
        "parquet_types_without_serialized_arrow_origin"
    )
    serialized_arrow_schema_removed: bool
    serialized_arrow_schema_restored: Literal[False] = False


@dataclass(frozen=True)
class _FieldSpan:
    id: int
    kind: int
    start: int
    end: int


class _Compact:
    """Walk bounded Compact-Thrift without instantiating user-defined objects."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0
        self.visits = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.position + size > len(self.data):
            raise ValueError("truncated Parquet Compact-Thrift metadata")
        result = self.data[self.position : self.position + size]
        self.position += size
        return result

    def byte(self) -> int:
        return self.take(1)[0]

    def unsigned(self, bits: int) -> int:
        result = 0
        for shift in range(0, bits, 7):
            byte = self.byte()
            result |= (byte & 127) << shift
            if not byte & 128:
                if result >= 1 << bits:
                    raise ValueError("oversized Compact-Thrift integer")
                return result
        raise ValueError("unterminated or oversized Compact-Thrift integer")

    def integer(self, bits: int) -> int:
        value = self.unsigned(bits)
        return (value >> 1) ^ -(value & 1)

    def binary(self) -> bytes:
        length = self.unsigned(32)
        if length > 1_000_000:
            raise ValueError("Parquet metadata binary/string value exceeds 1 MB")
        return self.take(length)

    def collection(self, limit: int = 100_000) -> tuple[int, int]:
        header = self.byte()
        count = header >> 4
        kind = header & 15
        if count == 15:
            count = self.unsigned(32)
        if count > limit or kind not in range(1, 13):
            raise ValueError("unsupported or over-budget Compact-Thrift collection")
        return count, kind

    def structure(self, depth: int, *, capture: bool = False) -> list[_FieldSpan]:
        if depth > 32:
            raise ValueError("Parquet metadata nesting exceeds 32 levels")
        previous = 0
        ids: set[int] = set()
        fields: list[_FieldSpan] = []
        while True:
            header = self.byte()
            if header == 0:
                return fields
            kind, delta = header & 15, header >> 4
            if kind not in range(1, 13):
                raise ValueError("unknown Compact-Thrift field type")
            identifier = previous + delta if delta else self.integer(16)
            if not 1 <= identifier <= 32767 or identifier in ids:
                raise ValueError("duplicate or invalid Compact-Thrift field identifier")
            ids.add(identifier)
            start = self.position
            self.value(kind, depth + 1, field_boolean=True)
            if capture:
                fields.append(_FieldSpan(identifier, kind, start, self.position))
            previous = identifier

    def value(self, kind: int, depth: int, *, field_boolean: bool = False) -> None:
        self.visits += 1
        if self.visits > 200_000 or depth > 32:
            raise ValueError("Parquet metadata exceeds 200000 value visits or 32 nesting levels")
        if kind in (1, 2):
            if not field_boolean and self.byte() not in (1, 2):
                raise ValueError("invalid Compact-Thrift boolean value")
        elif kind == 3:
            self.take(1)
        elif kind in (4, 5, 6):
            self.integer({4: 16, 5: 32, 6: 64}[kind])
        elif kind == 7:
            self.take(8)
        elif kind == 8:
            self.binary()
        elif kind in (9, 10):
            count, element_kind = self.collection()
            for _ in range(count):
                self.value(element_kind, depth + 1)
        elif kind == 11:
            count = self.unsigned(32)
            if count > 100_000:
                raise ValueError("Parquet metadata map exceeds 100000 entries")
            if count:
                types = self.byte()
                key_kind, value_kind = types >> 4, types & 15
                if key_kind not in range(1, 13) or value_kind not in range(1, 13):
                    raise ValueError("unsupported Compact-Thrift map types")
                for _ in range(count):
                    self.value(key_kind, depth + 1)
                    self.value(value_kind, depth + 1)
        elif kind == 12:
            self.structure(depth)
        else:
            raise ValueError("unknown Compact-Thrift value type")

    def field_binary(self, field: _FieldSpan) -> bytes:
        if field.kind != 8:
            raise ValueError("Parquet metadata expected a binary/string field")
        position = self.position
        self.position = field.start
        result = self.binary()
        self.position = position
        return result

    def field_integer(self, field: _FieldSpan, bits: int) -> int:
        if field.kind != {16: 4, 32: 5, 64: 6}[bits]:
            raise ValueError("Parquet metadata integer has the wrong wire type")
        position = self.position
        self.position = field.start
        result = self.integer(bits)
        self.position = position
        return result


def _unsigned(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _schema_depth(parser: _Compact, field: _FieldSpan) -> None:
    """The Parquet schema is flat on the wire; check its logical tree too."""
    if field.kind != 9:
        raise ValueError("Parquet schema must be a list")
    parser.position = field.start
    count, kind = parser.collection(4096)
    if kind != 12 or not count:
        raise ValueError("Parquet schema must contain bounded SchemaElement structs")
    remaining_children: list[int] = []
    for index in range(count):
        fields = {item.id: item for item in parser.structure(2, capture=True)}
        if 4 not in fields:
            raise ValueError("Parquet schema element lacks its required name")
        name = parser.field_binary(fields[4]).decode("utf-8")
        if not 1 <= len(name) <= 256:
            raise ValueError("Parquet schema names require 1 through 256 characters")
        children = parser.field_integer(fields[5], 32) if 5 in fields else 0
        if not 0 <= children <= 4096:
            raise ValueError("invalid Parquet schema child count")
        if index == 0:
            if 1 in fields or 5 not in fields or children > 256:
                raise ValueError("Parquet root must be a group with at most 256 fields")
        else:
            while remaining_children and remaining_children[-1] == 0:
                remaining_children.pop()
            if not remaining_children:
                raise ValueError("Parquet flattened schema has more than one root")
            remaining_children[-1] -= 1
        if len(remaining_children) + 1 > 32:
            raise ValueError("Parquet logical schema exceeds 32 levels")
        if children:
            remaining_children.append(children)
    if any(remaining_children) or parser.position != field.end:
        raise ValueError("Parquet flattened schema has missing children or malformed extent")


def _sanitize_footer(footer: bytes) -> tuple[bytes, bool]:
    parser = _Compact(footer)
    fields = {field.id: field for field in parser.structure(0, capture=True)}
    if parser.position != len(footer):
        raise ValueError("trailing bytes after Parquet FileMetaData are unsupported")
    if not {1, 2, 3, 4}.issubset(fields):
        raise ValueError("Parquet FileMetaData lacks required fields")
    if 8 in fields or 9 in fields:
        raise ValueError("encrypted/signed Parquet metadata is unsupported")
    if parser.field_integer(fields[1], 32) not in (1, 2) or parser.field_integer(fields[3], 64) < 0:
        raise ValueError("invalid Parquet metadata version or row count")
    _schema_depth(parser, fields[2])
    if fields[4].kind != 9:
        raise ValueError("Parquet row groups must be a list")
    parser.position = fields[4].start
    _, group_kind = parser.collection(10_000)
    if group_kind != 12:
        raise ValueError("Parquet row groups must contain structs")
    if 5 not in fields:
        return footer, False
    metadata = fields[5]
    if metadata.kind != 9:
        raise ValueError("Parquet key/value metadata must be a list")
    parser.position = metadata.start
    count, kind = parser.collection()
    if kind != 12:
        raise ValueError("Parquet key/value metadata must contain structs")
    kept: list[bytes] = []
    keys: set[bytes] = set()
    removed = False
    for _ in range(count):
        start = parser.position
        entry = {field.id: field for field in parser.structure(2, capture=True)}
        if 1 not in entry:
            raise ValueError("Parquet key/value entry lacks its key")
        key = parser.field_binary(entry[1])
        if key in keys:
            raise ValueError("duplicate Parquet file metadata keys are unsupported")
        keys.add(key)
        if 2 in entry and entry[2].kind != 8:
            raise ValueError("Parquet metadata values must use binary/string wire type")
        if key == b"ARROW:schema":
            removed = True
        else:
            kept.append(footer[start : parser.position])
    if parser.position != metadata.end:
        raise ValueError("invalid Parquet key/value list extent")
    if not removed:
        return footer, False
    header = bytes([(len(kept) << 4) | 12]) if len(kept) < 15 else b"\xfc" + _unsigned(len(kept))
    # Retain field 5 itself, even for an empty list. All surrounding field-ID
    # deltas and all retained KeyValue struct bytes therefore remain unchanged.
    sanitized = footer[: metadata.start] + header + b"".join(kept) + footer[metadata.end :]
    return sanitized, True


def _footer_size(trailer: bytes, source_size: int, maximum: int) -> int:
    if len(trailer) != 8 or trailer[4:] != b"PAR1":
        raise ValueError("Parquet must end with a plaintext PAR1 footer trailer")
    size = int.from_bytes(trailer[:4], "little")
    if size == 0 or size > maximum or size > source_size - 12:
        raise ValueError("invalid or over-budget Parquet footer length")
    return size


def execute(request: Input, context: OperationContext) -> Output:
    """Inspect declared metadata, without validating or reading column data pages."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_hash: str | None = None
    if request.read_mode == "full":
        content = context.read_bytes(
            request.path, suffixes=(".parquet",), max_bytes=min(request.max_file_bytes, 8_000_000)
        )
        source_size = len(content)
        if source_size < 12 or content[:4] != b"PAR1":
            raise ValueError("Parquet must start with PAR1 and contain a footer")
        footer_size = _footer_size(content[-8:], source_size, request.max_footer_bytes)
        footer_offset = source_size - 8 - footer_size
        footer = content[footer_offset:-8]
        source_hash = hashlib.sha256(content).hexdigest()
        bytes_read = source_size
    else:
        with context.open_ranges(
            request.path,
            suffixes=(".parquet",),
            max_file_bytes=request.max_file_bytes,
            max_read_bytes=request.max_footer_bytes + 12,
        ) as ranges:
            source_size = ranges.source_bytes
            if source_size < 12 or ranges.read_at(0, 4) != b"PAR1":
                raise ValueError("Parquet must start with PAR1 and contain a footer")
            trailer = ranges.read_at(source_size - 8, 8)
            footer_size = _footer_size(trailer, source_size, request.max_footer_bytes)
            footer_offset = source_size - 8 - footer_size
            footer = ranges.read_at(footer_offset, footer_size)
            ranges.check_unchanged_metadata()
            bytes_read = ranges.bytes_read
    sanitized_footer, arrow_origin_removed = _sanitize_footer(footer)
    # Parse only the sanitized metadata envelope in either mode. Full mode's
    # original-file hash above still covers the exact source bytes that were read.
    content = b"PAR1" + sanitized_footer + len(sanitized_footer).to_bytes(4, "little") + b"PAR1"
    try:
        options: dict[str, bool] = {}
        if "arrow_extensions_enabled" in inspect.signature(pq.ParquetFile).parameters:
            # Older supported PyArrow releases predate this optional mapping of
            # Parquet JSON/UUID logical types to canonical Arrow extensions.
            options["arrow_extensions_enabled"] = False
        parquet = pq.ParquetFile(
            pa.BufferReader(content),
            thrift_string_size_limit=1_000_000,
            thrift_container_size_limit=100_000,
            **options,
        )
        metadata = parquet.metadata
        schema = parquet.schema_arrow
        if len(schema) > 256 or metadata.num_columns > 1024:
            raise ValueError("Parquet schema exceeds 256 top-level or 1024 leaf columns")
        if metadata.num_row_groups > 10_000:
            raise ValueError("Parquet metadata exceeds 10000 row groups")
        columns = [
            Column(name=field.name, logical_type=str(field.type), nullable=field.nullable)
            for field in schema
        ]
        row_groups: list[RowGroup] = []
        stop = min(metadata.num_row_groups, request.row_group_offset + request.row_group_limit)
        for index in range(request.row_group_offset, stop):
            group = metadata.row_group(index)
            compressed = 0
            uncompressed = 0
            codecs: set[str] = set()
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                compressed += column.total_compressed_size
                uncompressed += column.total_uncompressed_size
                codecs.add(column.compression)
            row_groups.append(
                RowGroup(
                    index=index,
                    rows=group.num_rows,
                    compressed_bytes=compressed,
                    uncompressed_bytes=uncompressed,
                    compression_codecs=sorted(codecs),
                )
            )
        return Output(
            columns=columns,
            leaf_column_count=metadata.num_columns,
            total_rows=metadata.num_rows,
            row_group_count=metadata.num_row_groups,
            row_groups=row_groups,
            row_group_offset=request.row_group_offset,
            has_more_row_groups=stop < metadata.num_row_groups,
            format_version=str(metadata.format_version),
            source_sha256=source_hash,
            source_bytes=source_size,
            read_mode=request.read_mode,
            bytes_read=bytes_read,
            footer_offset=footer_offset,
            footer_bytes=footer_size,
            footer_sha256=hashlib.sha256(footer).hexdigest(),
            serialized_arrow_schema_removed=arrow_origin_removed,
        )
    except (pa.ArrowException, OSError) as exc:
        raise ValueError(f"invalid or unsupported Parquet metadata: {exc}") from exc


OPERATION = Operation(
    id="plugins.inspect_parquet",
    kind="plugin",
    description=(
        "Inspect a workspace Parquet schema, row counts, compression and paginated row-group "
        "sizes. Full mode hashes up to 8 MB; footer mode reads bounded metadata from files "
        "up to 1 TB, returns an explicit footer hash, and does not hash data pages. "
        "Bounded Thrift preflight removes embedded Arrow-origin schemas before decoding."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
    version="1.1.0",
)
