"""Read a strict, single-file dBASE III PLUS (version 0x03) subset.

Only C/N/D/L fields are supported. The 32-byte header, field descriptors, 0x0D
header terminator, exact fixed-size record region and final single 0x1A marker
are checked. Header reserved/LAN bytes and descriptor LAN/work-area/SET FIELDS
bytes must be zero; the descriptor's memory-address bytes are merely observed.
Memo, vendor fields, language-driver extensions, sidecars and missing EOF markers
are unsupported. Header update date is an observation, never a freshness claim.

The caller selects an explicit single-byte text codec; no encoding inference.
Character values retain every padding space and byte via raw_fields_hex. Numeric
values are exact trimmed ASCII decimals: optional minus, mandatory leading digits,
optional fractional digits up to the declared scale; no exponent/plus/overflow
stars. Numeric fields must be right-justified (no trailing spaces unless all blank).
Dates are valid YYYYMMDD or eight spaces; logical Y/T/N/F (either case) maps to
booleans, ? or space to null. Blank numeric/date cells also map to null, with raw
bytes retaining the distinctions. Deleted records receive the same validation.

Limits: 8 MB file, 100000 records, 64 fields, 4096 bytes/record, 2M scanned cells;
200 page records, 10000 page cells and 512 KB encoded page. Pagination is over
records selected by include_deleted; physical record index and byte offsets are
retained. Source hash includes header, deleted records and terminator.
Reference: https://blogs.embarcadero.com/dbase-dbf-file-structure/ (III PLUS section).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import struct
from typing import Literal, cast

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    text_codec: Literal["ascii", "cp437", "cp850", "cp1252", "latin-1"]
    include_deleted: bool = Field(default=False, strict=True)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Column(OutputModel):
    name: str
    kind: Literal["C", "N", "D", "L"]
    width: int
    decimal_count: int
    record_byte_start: int
    observed_memory_address_hex: str


class Row(OutputModel):
    selected_row_index: int
    physical_record_index: int
    source_byte_start: int
    deleted: bool
    values: list[str | bool | None]
    raw_fields_hex: list[str]


class Output(OutputModel):
    version: Literal["dBASE_III_PLUS_0x03_strict_subset"] = "dBASE_III_PLUS_0x03_strict_subset"
    text_codec: str
    header_update_date: str | None
    header_bytes: int
    record_bytes: int
    columns: list[Column]
    physical_record_count: int
    active_record_count: int
    deleted_record_count: int
    selected_record_count: int
    include_deleted: bool
    has_selected_records: bool
    rows: list[Row]
    offset: int
    has_more: bool
    numeric_encoding: Literal["exact_ascii_decimal_string"] = "exact_ascii_decimal_string"
    character_padding_preserved: Literal[True] = True
    full_source_validated: Literal[True] = True
    sidecars_read: Literal[False] = False
    source_bytes: int
    source_sha256: str


def _columns(content: bytes, header_bytes: int) -> list[Column]:
    count = (header_bytes - 33) // 32
    if not 1 <= count <= 64 or header_bytes != 33 + 32 * count:
        raise ValueError("DBF requires 1..64 exact 32-byte field descriptors")
    if content[header_bytes - 1] != 0x0D:
        raise ValueError("DBF header terminator must be 0x0D")
    result: list[Column] = []
    names: set[str] = set()
    offset = 1
    for index in range(count):
        descriptor = content[32 + 32 * index : 64 + 32 * index]
        name_bytes, separator, padding = descriptor[:11].partition(b"\x00")
        if (
            not separator
            or any(padding)
            or not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]{0,9}", name_bytes)
        ):
            raise ValueError("DBF field names must be zero-filled ASCII identifiers of 1..10 bytes")
        name = name_bytes.decode("ascii")
        if name.upper() in names:
            raise ValueError("DBF field names must be unique ignoring ASCII case")
        names.add(name.upper())
        kind = descriptor[11:12].decode("ascii")
        if kind not in {"C", "N", "D", "L"}:
            raise ValueError("DBF supports C/N/D/L only; memo and vendor fields are unsupported")
        if any(descriptor[18:]):
            raise ValueError(
                "DBF descriptor LAN/work-area/SET FIELDS bytes must be zero in this subset"
            )
        width, decimals = descriptor[16], descriptor[17]
        if kind == "C" and (not 1 <= width <= 254 or decimals):
            raise ValueError("DBF character width must be 1..254 with decimal count zero")
        if kind == "N" and (
            not 1 <= width <= 20 or decimals > 18 or (decimals and decimals > width - 2)
        ):
            raise ValueError("DBF numeric width/scale is outside the supported 1..20-byte subset")
        if kind in {"D", "L"} and (width != (8 if kind == "D" else 1) or decimals):
            raise ValueError("DBF date/logical widths must be 8/1 with decimal count zero")
        result.append(
            Column(
                name=name,
                kind=cast(Literal["C", "N", "D", "L"], kind),
                width=width,
                decimal_count=decimals,
                record_byte_start=offset,
                observed_memory_address_hex=descriptor[12:16].hex(),
            )
        )
        offset += width
    return result


def _value(raw: bytes, column: Column, codec: str) -> str | bool | None:
    if column.kind == "C":
        return raw.decode(codec, errors="strict")
    text = raw.decode("ascii", errors="strict")
    if column.kind == "L":
        if text in "YyTt":
            return True
        if text in "NnFf":
            return False
        if text in {"?", " "}:
            return None
        raise ValueError("DBF logical cells accept Y/T/N/F/?/space only")
    if text == " " * len(text):
        return None
    if column.kind == "D":
        if not re.fullmatch(r"[0-9]{8}", text):
            raise ValueError("DBF date must be eight ASCII digits or eight spaces")
        dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        return text
    trimmed = text.lstrip(" ")
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", trimmed):
        raise ValueError("DBF numeric cell must contain a right-justified plain ASCII decimal")
    fractional = len(trimmed.partition(".")[2])
    if fractional > column.decimal_count:
        raise ValueError("DBF numeric fractional digits exceed the declared decimal count")
    return trimmed


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".dbf",), max_bytes=8_000_000)
    if len(content) < 66 or content[0] != 0x03:
        raise ValueError("DBF requires a complete dBASE III PLUS 0x03 file without memo fields")
    if any(content[12:32]):
        raise ValueError(
            "DBF reserved/LAN header bytes must be zero; vendor extensions are unsupported"
        )
    record_count, header_bytes, record_bytes = struct.unpack_from("<IHH", content, 4)
    if header_bytes >= len(content) or header_bytes < 65:
        raise ValueError("DBF header length is out of bounds")
    columns = _columns(content, header_bytes)
    if record_bytes != 1 + sum(column.width for column in columns) or record_bytes > 4096:
        raise ValueError(
            "DBF record length must equal deletion flag plus field widths and be at most 4096"
        )
    if record_count > 100_000 or record_count * len(columns) > 2_000_000:
        raise ValueError("DBF exceeds 100000 records or 2M scanned cells")
    if request.limit * len(columns) > 10_000:
        raise ValueError("DBF requested page exceeds 10000 cells")
    if header_bytes + record_count * record_bytes + 1 != len(content) or content[-1] != 0x1A:
        raise ValueError(
            "DBF record region must be complete and followed by exactly one 0x1A EOF byte"
        )
    update_date: str | None = None
    if any(content[1:4]):
        update_date = dt.date(1900 + content[1], content[2], content[3]).isoformat()
    rows: list[Row] = []
    deleted_count = selected_count = page_bytes = 0
    for record_index in range(record_count):
        offset = header_bytes + record_index * record_bytes
        deletion = content[offset]
        if deletion not in (0x20, 0x2A):
            raise ValueError(f"DBF invalid deletion flag in record {record_index}")
        deleted = deletion == 0x2A
        deleted_count += int(deleted)
        raw_fields = [
            content[
                offset + column.record_byte_start : offset + column.record_byte_start + column.width
            ]
            for column in columns
        ]
        values = [
            _value(raw, column, request.text_codec)
            for raw, column in zip(raw_fields, columns, strict=True)
        ]
        if deleted and not request.include_deleted:
            continue
        if request.offset <= selected_count < request.offset + request.limit:
            row = Row(
                selected_row_index=selected_count,
                physical_record_index=record_index,
                source_byte_start=offset,
                deleted=deleted,
                values=values,
                raw_fields_hex=[raw.hex() for raw in raw_fields],
            )
            page_bytes += len(json.dumps(row.model_dump(), ensure_ascii=False).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("DBF encoded page exceeds 512 KB")
            rows.append(row)
        selected_count += 1
    return Output(
        text_codec=request.text_codec,
        header_update_date=update_date,
        header_bytes=header_bytes,
        record_bytes=record_bytes,
        columns=columns,
        physical_record_count=record_count,
        active_record_count=record_count - deleted_count,
        deleted_record_count=deleted_count,
        selected_record_count=selected_count,
        include_deleted=request.include_deleted,
        has_selected_records=bool(selected_count),
        rows=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < selected_count,
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_dbase",
    kind="plugin",
    description="Validate a strict dBASE III C/N/D/L file and return exact paged values, original field bytes and deletion/source lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
