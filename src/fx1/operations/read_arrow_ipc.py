"""Read a bounded flat Arrow IPC file or stream as explicitly encoded JSON.

Raw IPC/FlatBuffer preflight runs before PyArrow schema deserialization. Only
little-endian, uncompressed V4/V5 primitive columns are supported. Dictionaries,
extensions (including Python/pickle extensions), nested/view types and tensors
are rejected. Every batch and scalar is checked, including rows outside the
requested page. Both formats require an explicit EOS marker and no trailing bytes.

Limits: 8 MB source, 1 MB per metadata block, 64 columns, 256 batches, 10000
total rows, 250000 total cells and 16 MB summed buffer lengths. Strings/binary
cells are limited to 4096 bytes. Pages contain at most 200 rows, 4096 cells and
1 MB encoded row JSON. Uncompressed buffers are read without endian conversion;
array bounds are checked before decoding; metadata/Python memory is additional.

Integers and exact decimals become strings; binary becomes base64; temporal
values retain exact integer ticks as strings, with units/timezones in schema.
Timezone-free timestamps remain wall-clock values with an unspecified zone.
No datetime, pandas, extension, or object deserializer is invoked.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import Field, JsonValue, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(min_length=1, max_length=256)]
Encoding = Literal[
    "null",
    "boolean",
    "finite_number",
    "integer_string",
    "decimal_string",
    "utf8",
    "base64",
    "temporal_ticks_string",
]


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)
    columns: list[Name] | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def distinct_columns(self) -> Input:
        if self.columns is not None and len(set(self.columns)) != len(self.columns):
            raise ValueError("selected columns must be unique")
        return self


class Column(OutputModel):
    name: str
    arrow_type: str
    nullable: bool
    json_encoding: Encoding
    unit: str | None
    timezone: str | None


class Output(OutputModel):
    format: Literal["file", "stream"]
    columns: list[Column]
    selected_columns: list[str]
    rows: list[dict[str, JsonValue]]
    total_rows: int
    batch_count: int
    offset: int
    returned_rows: int
    has_more: bool
    observations_present: bool
    all_columns_and_rows_validated: Literal[True] = True
    compression_policy: Literal["reject_before_decoding"] = "reject_before_decoding"
    temporal_policy: Literal["exact_ticks_preserve_unspecified_timezone"] = (
        "exact_ticks_preserve_unspecified_timezone"
    )
    declared_buffer_bytes: int
    source_bytes: int
    source_sha256: str


class _Flatbuffer:
    """Bounds-check the small Arrow metadata subset before native parsing."""

    def __init__(self, data: bytes) -> None:
        if not 4 <= len(data) <= 1_000_000:
            raise ValueError("IPC metadata block must contain 4 through 1000000 bytes")
        if b"ARROW:extension:name" in data or b"ARROW:extension:metadata" in data:
            raise ValueError("Arrow extension metadata is unsupported before deserialization")
        self.data = data
        self.string_bytes = 0

    def number(self, position: int, fmt: str) -> int:
        width = struct.calcsize(fmt)
        if position < 0 or position + width > len(self.data):
            raise ValueError("IPC metadata scalar is out of bounds")
        return int(struct.unpack_from(fmt, self.data, position)[0])

    def table(self, position: int) -> tuple[int, int, int]:
        vtable = position - self.number(position, "<i")
        size = self.number(vtable, "<H")
        object_size = self.number(vtable + 2, "<H")
        if size < 4 or size % 2 or vtable + size > len(self.data):
            raise ValueError("invalid IPC metadata vtable")
        if object_size < 4 or position + object_size > len(self.data):
            raise ValueError("invalid IPC metadata table extent")
        return vtable, size, object_size

    def field(self, table: int, slot: int, width: int = 4) -> int | None:
        vtable, size, object_size = self.table(table)
        if 4 + slot * 2 >= size:
            return None
        offset = self.number(vtable + 4 + slot * 2, "<H")
        if offset == 0:
            return None
        if offset < 4 or offset + width > object_size:
            raise ValueError("IPC metadata field is out of bounds")
        return table + offset

    def scalar(self, table: int, slot: int, fmt: str, default: int = 0) -> int:
        position = self.field(table, slot, struct.calcsize(fmt))
        return default if position is None else self.number(position, fmt)

    def indirect(self, position: int) -> int:
        distance = self.number(position, "<I")
        target = position + distance
        if distance < 4 or target + 4 > len(self.data):
            raise ValueError("IPC metadata offset is out of bounds")
        return target

    def pointer(self, table: int, slot: int, *, required: bool = False) -> int | None:
        position = self.field(table, slot)
        if position is None:
            if required:
                raise ValueError("required IPC metadata field is absent")
            return None
        return self.indirect(position)

    def vector(self, table: int, slot: int, width: int, limit: int) -> list[int]:
        position = self.pointer(table, slot)
        if position is None:
            return []
        count = self.number(position, "<I")
        if count > limit or position + 4 + count * width > len(self.data):
            raise ValueError("IPC metadata vector exceeds its bounded extent")
        return [position + 4 + index * width for index in range(count)]

    def string(self, table: int, slot: int, limit: int) -> str:
        position = self.pointer(table, slot)
        if position is None:
            return ""
        size = self.number(position, "<I")
        start = position + 4
        if size > limit or start + size >= len(self.data) or self.data[start + size] != 0:
            raise ValueError("IPC metadata string exceeds its bounded extent")
        self.string_bytes += size
        if self.string_bytes > 1_000_000:
            raise ValueError("IPC expanded metadata strings exceed 1 MB")
        return self.data[start : start + size].decode("utf-8")

    def metadata(self, table: int, slot: int) -> tuple[tuple[str, str], ...]:
        result: list[tuple[str, str]] = []
        for position in self.vector(table, slot, 4, 64):
            item = self.indirect(position)
            result.append((self.string(item, 0, 256), self.string(item, 1, 4096)))
        if len({key for key, _ in result}) != len(result):
            raise ValueError("duplicate IPC custom metadata keys are unsupported")
        return tuple(result)

    def root(self) -> int:
        position = self.number(0, "<I")
        if position < 4:
            raise ValueError("invalid IPC metadata root")
        self.table(position)
        return position


@dataclass(frozen=True)
class _Field:
    name: str
    kind: int
    nullable: bool
    parameters: tuple[int | str, ...]
    metadata: tuple[tuple[str, str], ...]

    @property
    def buffer_count(self) -> int:
        return 0 if self.kind == 1 else (3 if self.kind in (4, 5, 19, 20) else 2)


def _schema(fb: _Flatbuffer, table: int) -> tuple[list[_Field], tuple[tuple[str, str], ...]]:
    if fb.scalar(table, 0, "<h") != 0:
        raise ValueError("only little-endian Arrow buffers are supported")
    if fb.vector(table, 3, 8, 0):
        raise ValueError("Arrow feature flags are unsupported")
    fields: list[_Field] = []
    for position in fb.vector(table, 1, 4, 64):
        field = fb.indirect(position)
        name = fb.string(field, 0, 256)
        nullable = fb.scalar(field, 1, "<B")
        kind = fb.scalar(field, 2, "<B")
        type_table = fb.pointer(field, 3, required=True)
        assert type_table is not None
        fb.table(type_table)
        if not name or nullable not in (0, 1):
            raise ValueError("Arrow fields need nonempty unique names and boolean nullability")
        if fb.pointer(field, 4) is not None or fb.vector(field, 5, 4, 0):
            raise ValueError("dictionary and nested Arrow fields are unsupported")
        parameters: tuple[int | str, ...] = ()
        if kind == 2:
            width = fb.scalar(type_table, 0, "<i")
            signed = fb.scalar(type_table, 1, "<B")
            if width not in (8, 16, 32, 64) or signed not in (0, 1):
                raise ValueError("unsupported Arrow integer parameters")
            parameters = (width, signed)
        elif kind == 3:
            precision = fb.scalar(type_table, 0, "<h")
            if precision not in (1, 2):
                raise ValueError("only float32 and float64 are supported")
            parameters = (precision,)
        elif kind == 7:
            precision = fb.scalar(type_table, 0, "<i")
            scale = fb.scalar(type_table, 1, "<i")
            width = fb.scalar(type_table, 2, "<i", 128)
            if (
                width not in (128, 256)
                or not 1 <= precision <= (38 if width == 128 else 76)
                or abs(scale) > 76
            ):
                raise ValueError("unsupported Arrow decimal precision, scale or width")
            parameters = (precision, scale, width)
        elif kind in (8, 9, 10, 18):
            unit = fb.scalar(type_table, 0, "<h", 0 if kind == 10 else 1)
            valid_units = (0, 1) if kind == 8 else (0, 1, 2, 3)
            if unit not in valid_units:
                raise ValueError("unsupported Arrow temporal unit")
            parameters = (unit,)
            if kind == 9:
                width = fb.scalar(type_table, 1, "<i", 32)
                if width != (32 if unit <= 1 else 64):
                    raise ValueError("Arrow time width does not match its unit")
                parameters += (width,)
            elif kind == 10:
                parameters += (fb.string(type_table, 1, 128),)
        elif kind == 15:
            width = fb.scalar(type_table, 0, "<i")
            if not 0 <= width <= 4096:
                raise ValueError("fixed binary width exceeds 4096 bytes")
            parameters = (width,)
        elif kind not in (1, 4, 5, 6, 19, 20):
            raise ValueError("unsupported Arrow type; only flat primitive columns are accepted")
        fields.append(_Field(name, kind, bool(nullable), parameters, fb.metadata(field, 6)))
    if len({field.name for field in fields}) != len(fields):
        raise ValueError("Arrow column names must be unique")
    return fields, fb.metadata(table, 2)


def _batch(fb: _Flatbuffer, table: int, fields: list[_Field], body_bytes: int) -> tuple[int, int]:
    if fb.pointer(table, 3) is not None:
        raise ValueError("compressed Arrow IPC buffers are rejected before decoding")
    if fb.vector(table, 4, 8, 0):
        raise ValueError("variadic Arrow buffers are unsupported")
    rows = fb.scalar(table, 0, "<q")
    if not 0 <= rows <= 10_000:
        raise ValueError("Arrow batch exceeds the 10000-row bound")
    nodes = fb.vector(table, 1, 16, 64)
    if len(nodes) != len(fields):
        raise ValueError("Arrow field-node count does not match the flat schema")
    for field, position in zip(fields, nodes, strict=True):
        length = fb.number(position, "<q")
        nulls = fb.number(position + 8, "<q")
        if (
            length != rows
            or not 0 <= nulls <= rows
            or (nulls and not field.nullable)
            or (field.kind == 1 and nulls != rows)
        ):
            raise ValueError("Arrow field-node length/null count violates the schema")
    buffers = fb.vector(table, 2, 16, 192)
    if len(buffers) != sum(field.buffer_count for field in fields):
        raise ValueError("Arrow buffer count does not match the flat schema")
    buffer_bytes = 0
    for position in buffers:
        offset = fb.number(position, "<q")
        size = fb.number(position + 8, "<q")
        if offset < 0 or size < 0 or offset + size > body_bytes:
            raise ValueError("Arrow buffer lies outside its bounded message body")
        buffer_bytes += size
    return rows, buffer_bytes


@dataclass(frozen=True)
class _Layout:
    format: Literal["file", "stream"]
    fields: list[_Field]
    batch_rows: list[int]
    buffer_bytes: int


def _preflight(content: bytes) -> _Layout:
    is_file = content.startswith(b"ARROW1")
    end = len(content)
    position = 0
    footer_schema: tuple[list[_Field], tuple[tuple[str, str], ...]] | None = None
    footer_blocks: list[tuple[int, int, int]] = []
    footer_version: int | None = None
    if is_file:
        if len(content) < 18 or content[-6:] != b"ARROW1" or content[6:8] != b"\0\0":
            raise ValueError("invalid Arrow IPC file magic or padding")
        footer_size = struct.unpack_from("<i", content, len(content) - 10)[0]
        end = len(content) - 10 - footer_size
        if not 4 <= footer_size <= 1_000_000 or end < 8:
            raise ValueError("invalid or oversized Arrow file footer")
        footer = _Flatbuffer(content[end : len(content) - 10])
        root = footer.root()
        footer_version = footer.scalar(root, 0, "<h")
        schema_position = footer.pointer(root, 1, required=True)
        assert schema_position is not None
        footer_schema = _schema(footer, schema_position)
        footer.vector(root, 2, 24, 0)
        footer.metadata(root, 4)
        for block in footer.vector(root, 3, 24, 256):
            footer_blocks.append(
                (
                    footer.number(block, "<q"),
                    footer.number(block + 8, "<i"),
                    footer.number(block + 16, "<q"),
                )
            )
        position = 8
    schema: tuple[list[_Field], tuple[tuple[str, str], ...]] | None = None
    blocks: list[tuple[int, int, int]] = []
    batch_rows: list[int] = []
    total_rows = buffer_bytes = 0
    eos = False
    stream_version: int | None = None
    while position < end:
        start = position
        if position + 4 > end:
            raise ValueError("truncated Arrow IPC message prefix")
        metadata_size = struct.unpack_from("<I", content, position)[0]
        position += 4
        if metadata_size == 0xFFFFFFFF:
            if position + 4 > end:
                raise ValueError("truncated Arrow IPC continuation prefix")
            metadata_size = struct.unpack_from("<I", content, position)[0]
            position += 4
        if metadata_size == 0:
            eos = True
            if position != end:
                raise ValueError("trailing bytes after Arrow IPC end-of-stream marker")
            break
        if not 4 <= metadata_size <= 1_000_000 or position + metadata_size > end:
            raise ValueError("Arrow message metadata is truncated or oversized")
        fb = _Flatbuffer(content[position : position + metadata_size])
        root = fb.root()
        version = fb.scalar(root, 0, "<h")
        if (
            version not in (3, 4)
            or (footer_version is not None and version != footer_version)
            or (stream_version is not None and version != stream_version)
        ):
            raise ValueError("only consistent Arrow metadata V4/V5 is supported")
        stream_version = version
        kind = fb.scalar(root, 1, "<B")
        header = fb.pointer(root, 2, required=True)
        assert header is not None
        body_size = fb.scalar(root, 3, "<q")
        fb.metadata(root, 4)
        position += metadata_size
        metadata_extent = position - start
        if body_size < 0 or position + body_size > end:
            raise ValueError("Arrow message body is truncated or oversized")
        if kind == 1:
            if schema is not None or batch_rows or body_size:
                raise ValueError("Arrow IPC requires one initial schema with no body")
            schema = _schema(fb, header)
        elif kind == 3:
            if schema is None or len(batch_rows) >= 256:
                raise ValueError("Arrow batch needs an initial schema and at most 256 batches")
            rows, declared_bytes = _batch(fb, header, schema[0], body_size)
            total_rows += rows
            buffer_bytes += declared_bytes
            if (
                total_rows > 10_000
                or total_rows * len(schema[0]) > 250_000
                or buffer_bytes > 16_000_000
            ):
                raise ValueError("Arrow data exceeds row, cell or 16 MB summed-buffer bounds")
            batch_rows.append(rows)
            blocks.append((start, metadata_extent, body_size))
        else:
            raise ValueError("dictionary, tensor and unknown Arrow message kinds are unsupported")
        position += body_size
    if schema is None or not eos:
        raise ValueError("Arrow IPC requires a schema and explicit end-of-stream marker")
    if is_file and (schema != footer_schema or blocks != footer_blocks):
        raise ValueError("Arrow file footer diverges from the bounded embedded stream")
    return _Layout("file" if is_file else "stream", schema[0], batch_rows, buffer_bytes)


def _column(field: _Field, arrow_type: Any) -> Column:
    encodings: dict[int, Encoding] = {
        1: "null",
        2: "integer_string",
        3: "finite_number",
        4: "base64",
        5: "utf8",
        6: "boolean",
        7: "decimal_string",
        8: "temporal_ticks_string",
        9: "temporal_ticks_string",
        10: "temporal_ticks_string",
        15: "base64",
        18: "temporal_ticks_string",
        19: "base64",
        20: "utf8",
    }
    unit = None
    timezone = None
    if field.kind in (8, 9, 10, 18):
        unit_index = int(field.parameters[0])
        unit = ("day", "ms")[unit_index] if field.kind == 8 else ("s", "ms", "us", "ns")[unit_index]
        if field.kind == 10:
            timezone = str(field.parameters[1]) or None
    return Column(
        name=field.name,
        arrow_type=str(arrow_type),
        nullable=field.nullable,
        json_encoding=encodings[field.kind],
        unit=unit,
        timezone=timezone,
    )


def _value(scalar: Any, field: _Field) -> JsonValue:
    if not scalar.is_valid:
        if not field.nullable:
            raise ValueError("null scalar violates Arrow field nullability")
        return None
    if field.kind in (4, 5, 15, 19, 20):
        if scalar.as_buffer().size > 4096:
            raise ValueError("Arrow string/binary cell exceeds 4096 bytes")
        value = scalar.as_py()
        if field.kind in (5, 20):
            return str(value)
        return base64.b64encode(value).decode("ascii")
    if field.kind in (8, 9, 10, 18):
        ticks = int(scalar.value)
        unit = int(field.parameters[0])
        if field.kind == 9 and not 0 <= ticks < 86_400 * (1, 1000, 1_000_000, 1_000_000_000)[unit]:
            raise ValueError("Arrow time value is outside one day")
        if field.kind == 8 and unit == 1 and ticks % 86_400_000:
            raise ValueError("Arrow date64 values must be whole days")
        return str(ticks)
    value = scalar.as_py()
    if field.kind == 3:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite Arrow floating-point cells cannot become finite JSON")
        return number
    if field.kind == 7:
        if not value.is_finite() or len(value.as_tuple().digits) > int(field.parameters[0]):
            raise ValueError("Arrow decimal cell violates declared precision")
        return str(value)
    if field.kind == 2:
        return str(value)
    if field.kind == 6:
        return bool(value)
    return None


def execute(request: Input, context: OperationContext) -> Output:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    if sys.byteorder != "little":
        raise ValueError("bounded zero-copy Arrow reading requires a little-endian host")
    content = context.read_bytes(
        request.path, suffixes=(".arrow", ".arrows", ".ipc", ".feather"), max_bytes=8_000_000
    )
    try:
        layout = _preflight(content)
        names = [field.name for field in layout.fields]
        selected = request.columns if request.columns is not None else names
        if any(name not in names for name in selected):
            raise ValueError("selected Arrow column does not exist")
        if request.limit * len(selected) > 4096:
            raise ValueError("Arrow page limit times selected columns exceeds 4096 cells")
        options = ipc.IpcReadOptions(ensure_native_endian=False, use_threads=False)
        source = pa.BufferReader(content)
        reader = (
            ipc.open_file(source, options=options)
            if layout.format == "file"
            else ipc.open_stream(source, options=options)
        )
        if len(reader.schema) != len(layout.fields) or reader.schema.names != names:
            raise ValueError("decoded Arrow schema differs from preflight")
        columns = [
            _column(field, reader.schema[index].type) for index, field in enumerate(layout.fields)
        ]
        rows: list[dict[str, JsonValue]] = []
        seen_rows = encoded_bytes = 0
        selected_set = set(selected)
        if layout.format == "file" and reader.num_record_batches != len(layout.batch_rows):
            raise ValueError("decoded Arrow batch count differs from preflight")
        for batch_index, expected_rows in enumerate(layout.batch_rows):
            batch = (
                reader.get_batch(batch_index)
                if layout.format == "file"
                else reader.read_next_batch()
            )
            if batch.num_rows != expected_rows or batch.num_columns != len(layout.fields):
                raise ValueError("decoded Arrow batch shape differs from preflight")
            batch.validate(full=True)
            for local_row in range(batch.num_rows):
                in_page = request.offset <= seen_rows < request.offset + request.limit
                row: dict[str, JsonValue] = {}
                for column_index, field in enumerate(layout.fields):
                    value = _value(batch.column(column_index)[local_row], field)
                    if in_page and field.name in selected_set:
                        row[field.name] = value
                if in_page:
                    row = {name: row[name] for name in selected}
                    encoded_bytes += len(
                        json.dumps(
                            row, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                        ).encode("utf-8")
                    )
                    if encoded_bytes > 1_000_000:
                        raise ValueError("Arrow page exceeds 1 MB encoded rows; reduce page size")
                    rows.append(row)
                seen_rows += 1
        return Output(
            format=layout.format,
            columns=columns,
            selected_columns=selected,
            rows=rows,
            total_rows=seen_rows,
            batch_count=len(layout.batch_rows),
            offset=request.offset,
            returned_rows=len(rows),
            has_more=request.offset + len(rows) < seen_rows,
            observations_present=seen_rows > 0,
            declared_buffer_bytes=layout.buffer_bytes,
            source_bytes=len(content),
            source_sha256=hashlib.sha256(content).hexdigest(),
        )
    except (pa.ArrowException, UnicodeError, OverflowError, StopIteration, struct.error) as exc:
        raise ValueError(f"invalid or unsupported bounded Arrow IPC: {exc}") from exc


OPERATION = Operation(
    id="plugins.read_arrow_ipc",
    kind="plugin",
    description=(
        "Read bounded pages from an uncompressed flat Arrow IPC file or stream, checking "
        "all rows and raw metadata first. Rejects compression, extensions, dictionaries and "
        "nested types before decoding; returns explicit lossless scalar encodings and a source hash."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
