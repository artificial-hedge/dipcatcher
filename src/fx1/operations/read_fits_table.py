"""Read one bounded scalar FITS table following an empty primary HDU.

Supported files have SIMPLE=T, BITPIX=8, NAXIS=0, EXTEND=T in the primary,
then exactly one TABLE/BINTABLE extension, no heap and no further HDUs. Headers
use printable ASCII 80-byte cards, fixed-format mandatory values, unique value keywords
and 2880-byte padding. COMMENT/HISTORY/blank cards are preserved. CONTINUE,
HIERARCH, arrays, images, compression, random groups and complex values fail.

ASCII columns support Aw/Iw/Fw.d/Ew.d/Dw.d, disjoint TBCOL spans, printable
characters and blank gaps. Numeric tokens have no embedded blanks; real tokens
must contain an explicit decimal point (implicit decimal/exponent conventions
are unsupported). Real tokens are exact decimal strings, not binary floats.
TNULL matches the header string left-aligned and padded to the full field width.
Binary columns support scalar L/B/I/J/K/E/D and 1..256 character A strings.
Integers are exact decimal strings; finite floats use float.hex; IEEE NaNs and
logical zero bytes are null. Infinite floats fail. Binary A strings stop at the
first NUL; a leading NUL is null, and undefined suffix bytes remain in raw_hex.

TSCAL/TZERO are retained as exact decimal declarations. The default rejects
nonidentity scaling; preserve_raw returns stored values without applying scale
or offset. Null recognition precedes scaling. Column names may repeat: ordinal
indices identify columns. Units, display/WCS and informational metadata are not
interpreted; original header cards are returned. CHECKSUM/DATASUM declarations
are observations only, never verified checksums. Source SHA-256 hashes every
original byte. Row/cell byte positions are absolute, zero-based source offsets.

Limits: 8 MB source, 1024 cards/header, 64 fields, 8192 bytes/row, 100000 rows,
1M total cells, 256 bytes/field; pages at most 200 rows/10000 cells/512 KB JSON.
Full table, padding and file extent are validated before output. A zero-row
table is explicitly reported and supplies no data evidence.
References: NASA FITS Standard 4.0 sections 3,4,7; NASA FITS Users' Guide
sections 3.4/3.6, https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[ED][+-]?[0-9]+)?\Z")
_KEY = re.compile(r"[A-Z0-9_-]{1,8}\Z")
_INDEXED = re.compile(r"(TFORM|TBCOL|TTYPE|TUNIT|TNULL|TSCAL|TZERO|TDISP|TDIM)([0-9]+)\Z")


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)
    scaling_policy: Literal["require_identity", "preserve_raw"] = "require_identity"


class Column(OutputModel):
    column_index: int
    name: str | None
    unit: str | None
    format: str
    scalar_type: str
    row_byte_start: int
    byte_width: int
    null_declaration: str | None
    scale_declaration: str
    zero_declaration: str
    nonidentity_scaling_declared: bool


class Cell(OutputModel):
    source_byte_start: int
    raw_hex: str
    value: str | bool | None
    value_encoding: Literal["text", "integer_decimal", "real_decimal", "float_hex", "boolean"]
    null_reason: Literal["tnull", "ieee_nan", "logical_nul", "string_nul"] | None = None
    string_nul_terminated: bool = False


class Row(OutputModel):
    row_index: int
    source_byte_start: int
    cells: list[Cell]


class Output(OutputModel):
    table_kind: Literal["TABLE", "BINTABLE"]
    columns: list[Column]
    row_bytes: int
    row_count: int
    has_rows: bool
    table_data_byte_start: int
    table_data_bytes: int
    table_padding_bytes: int
    primary_header_cards: list[str]
    table_header_cards: list[str]
    primary_header_sha256: str
    table_header_sha256: str
    rows: list[Row]
    offset: int
    has_more: bool
    scaling_policy: Literal["require_identity", "preserve_raw"]
    physical_scaling_applied: Literal[False] = False
    values_represent: Literal["stored_unscaled_values_with_declared_nulls"] = (
        "stored_unscaled_values_with_declared_nulls"
    )
    fits_checksum_verified: Literal[False] = False
    informational_metadata_interpreted: Literal[False] = False
    full_source_validated_for_supported_subset: Literal[True] = True
    source_bytes: int
    source_sha256: str


@dataclass(frozen=True)
class _Value:
    kind: str
    text: str


def _value(body: str) -> _Value:
    body = body.lstrip(" ")
    if body.startswith("'"):
        chars: list[str] = []
        position = 1
        while position < len(body):
            char = body[position]
            position += 1
            if char == "'":
                if position < len(body) and body[position] == "'":
                    chars.append("'")
                    position += 1
                    continue
                rest = body[position:].lstrip(" ")
                if rest and not rest.startswith("/"):
                    raise ValueError("FITS string has trailing non-comment characters")
                return _Value("string", "".join(chars).rstrip(" "))
            chars.append(char)
        raise ValueError("FITS header string is unterminated")
    token = body.split("/", 1)[0].strip(" ")
    if token in {"T", "F"}:
        return _Value("boolean", token)
    if _INTEGER.fullmatch(token):
        return _Value("integer", token)
    if _REAL.fullmatch(token):
        return _Value("real", token)
    raise ValueError("FITS header requires a supported nonempty scalar value")


def _header(source: bytes, start: int) -> tuple[dict[str, _Value], list[str], int]:
    values: dict[str, _Value] = {}
    cards: list[str] = []
    for index in range(1024):
        position = start + 80 * index
        raw = source[position : position + 80]
        if len(raw) != 80 or any(byte < 32 or byte > 126 for byte in raw):
            raise ValueError("FITS header is truncated or contains non-printable ASCII")
        card = raw.decode("ascii")
        key = card[:8].rstrip(" ")
        cards.append(card)
        if key == "END":
            if card[8:] != " " * 72:
                raise ValueError("FITS END card must have a blank remainder")
            end = start + ((index + 1 + 35) // 36) * 2880
            if end > len(source) or source[position + 80 : end].strip(b" "):
                raise ValueError("FITS header padding must be complete ASCII spaces")
            return values, cards, end
        if key in {"", "COMMENT", "HISTORY"}:
            continue
        if not _KEY.fullmatch(key) or key in {"CONTINUE", "HIERARCH"}:
            raise ValueError("FITS keyword or long-string convention is unsupported")
        if key in values or card[8:10] != "= ":
            raise ValueError("FITS value keywords must be unique and use '= ' cards")
        values[key] = _value(card[10:])
    raise ValueError("FITS header exceeds 1024 cards or has no END")


def _integer(values: dict[str, _Value], key: str) -> int:
    value = values.get(key)
    if value is None or value.kind != "integer":
        raise ValueError(f"FITS {key} requires an integer declaration")
    return int(value.text)


def _fixed_cards(cards: list[str], values: dict[str, _Value], mandatory: set[str]) -> None:
    for card in cards:
        key = card[:8].rstrip(" ")
        if key not in mandatory:
            continue
        value = values[key]
        if value.kind == "string":
            # Fixed strings open in column 11 and occupy at least eight
            # character positions before their final quote. Doubled quotes
            # are escapes; the generic parser has already checked closure.
            position = 11
            if card[10] != "'":
                raise ValueError("FITS mandatory strings must start in column 11")
            while position < 80:
                if card[position] == "'":
                    if position + 1 < 80 and card[position + 1] == "'":
                        position += 2
                        continue
                    break
                position += 1
            if position < 19:
                raise ValueError(
                    "FITS mandatory strings must be padded to at least eight characters"
                )
        else:
            if value.kind not in {"integer", "boolean"} or card[10:30] != value.text.rjust(20):
                raise ValueError("FITS mandatory integers/logicals must end in column 30")
            remaining = card[30:].lstrip(" ")
            if remaining and not remaining.startswith("/"):
                raise ValueError("FITS fixed mandatory value extends past column 30")


def _string(values: dict[str, _Value], key: str) -> str:
    value = values.get(key)
    if value is None or value.kind != "string":
        raise ValueError(f"FITS {key} requires a string declaration")
    return value.text


def _decimal(token: str) -> str:
    if not _REAL.fullmatch(token):
        raise ValueError("FITS decimal token has unsupported syntax")
    exponent = re.split("[ED]", token)[1:]
    if exponent and (len(exponent[0]) > 6 or abs(int(exponent[0])) > 10_000):
        raise ValueError("FITS decimal exponent exceeds 10000")
    # Decimal construction is exact and does not apply the ambient context.
    return str(Decimal(token.replace("D", "E")))


def _scale(values: dict[str, _Value], key: str, default: str) -> str:
    value = values.get(key)
    if value is None:
        return default
    if value.kind not in {"integer", "real"}:
        raise ValueError(f"FITS {key} requires a numeric declaration")
    return _decimal(value.text)


def _columns(
    values: dict[str, _Value], count: int, row_bytes: int, binary: bool, policy: str
) -> tuple[list[Column], list[tuple[int, int]]]:
    for key in values:
        indexed = _INDEXED.fullmatch(key)
        if indexed and (not 1 <= int(indexed[2]) <= count or str(int(indexed[2])) != indexed[2]):
            raise ValueError("FITS indexed column keywords must address existing canonical indices")
        if key == "THEAP" or key.startswith("Z") or key.startswith("TDIM"):
            raise ValueError(
                "FITS heaps, compression and dimensional array declarations are unsupported"
            )
    columns: list[Column] = []
    spans: list[tuple[int, int]] = []
    next_start = 0
    for ordinal in range(1, count + 1):
        suffix = str(ordinal)
        form = _string(values, "TFORM" + suffix)
        if binary:
            matched = re.fullmatch(r"([0-9]*)([LBIJKAED])", form)
            if matched is None:
                raise ValueError("FITS binary TFORM supports scalar L/B/I/J/K/E/D or A text only")
            repeat, kind = int(matched[1] or "1"), matched[2]
            if not 1 <= repeat <= 256 or kind != "A" and repeat != 1:
                raise ValueError("FITS binary numeric vectors/empty fields are unsupported")
            width = repeat * {"L": 1, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8}[kind]
            start = next_start
            if "TBCOL" + suffix in values:
                raise ValueError("FITS binary tables cannot use ASCII TBCOL declarations")
        else:
            matched = re.fullmatch(r"([AI])([0-9]+)|([FED])([0-9]+)\.([0-9]+)", form)
            if matched is None:
                raise ValueError("FITS ASCII TFORM requires Aw/Iw/Fw.d/Ew.d/Dw.d")
            kind = matched[1] or matched[3]
            width = int(matched[2] or matched[4])
            if matched[5] is not None and not 0 <= int(matched[5]) <= width:
                raise ValueError("FITS ASCII format decimal count exceeds its width")
            start = _integer(values, "TBCOL" + suffix) - 1
        if not 1 <= width <= 256 or start < 0 or start + width > row_bytes:
            raise ValueError("FITS field width/span is outside its bounded row")
        if any(start < right and left < start + width for left, right in spans):
            raise ValueError("FITS ASCII column spans must not overlap")
        spans.append((start, start + width))
        next_start = start + width
        null: str | None = None
        if "TNULL" + suffix in values:
            if not binary:
                null = _string(values, "TNULL" + suffix)
                if len(null) > width:
                    raise ValueError("FITS ASCII null marker exceeds field width")
            else:
                if kind not in {"B", "I", "J", "K"}:
                    raise ValueError("FITS binary TNULL applies only to integer columns")
                number = _integer(values, "TNULL" + suffix)
                bits = 8 * width
                low, high = (0, 255) if kind == "B" else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
                if not low <= number <= high:
                    raise ValueError("FITS integer TNULL is outside its stored type")
                null = str(number)
        scale = _scale(values, "TSCAL" + suffix, "1")
        zero = _scale(values, "TZERO" + suffix, "0")
        if kind in {"A", "L"} and any(prefix + suffix in values for prefix in ("TSCAL", "TZERO")):
            raise ValueError("FITS scaling cannot be declared for character/logical columns")
        changed = Decimal(scale) != 1 or Decimal(zero) != 0
        if changed and policy == "require_identity":
            raise ValueError("FITS nonidentity scaling requires explicit preserve_raw policy")
        columns.append(
            Column(
                column_index=ordinal - 1,
                name=_string(values, "TTYPE" + suffix) if "TTYPE" + suffix in values else None,
                unit=_string(values, "TUNIT" + suffix) if "TUNIT" + suffix in values else None,
                format=form,
                scalar_type=kind,
                row_byte_start=start,
                byte_width=width,
                null_declaration=null,
                scale_declaration=scale,
                zero_declaration=zero,
                nonidentity_scaling_declared=changed,
            )
        )
    if binary and next_start != row_bytes:
        raise ValueError("FITS binary fields do not exactly fill NAXIS1")
    gaps: list[tuple[int, int]] = []
    previous = 0
    for start, end in sorted(spans):
        if start > previous:
            gaps.append((previous, start))
        previous = end
    if previous < row_bytes:
        gaps.append((previous, row_bytes))
    return columns, gaps


def _cell(raw: bytes, column: Column, binary: bool, position: int) -> Cell:
    kind = column.scalar_type
    encoding: Literal["text", "integer_decimal", "real_decimal", "float_hex", "boolean"]
    encoding = (
        "text"
        if kind == "A"
        else "integer_decimal"
        if kind in {"B", "I", "J", "K"}
        else "real_decimal"
    )
    cell = Cell(source_byte_start=position, raw_hex=raw.hex(), value=None, value_encoding=encoding)
    if not binary:
        text = raw.decode("ascii")
        if column.null_declaration is not None and text == column.null_declaration.ljust(len(text)):
            cell.null_reason = "tnull"
        elif kind == "A":
            cell.value = text
        else:
            token = text.strip(" ")
            if kind == "I":
                if not _INTEGER.fullmatch(token):
                    raise ValueError("FITS ASCII integer field has invalid or blank syntax")
                cell.value = str(int(token))
            else:
                if "." not in re.split("[ED]", token)[0]:
                    raise ValueError("FITS ASCII real fields require an explicit decimal point")
                cell.value = _decimal(token)
        return cell
    if kind == "A":
        prefix, separator, _ = raw.partition(b"\0")
        if any(byte < 32 or byte > 126 for byte in prefix):
            raise ValueError("FITS binary string prefix must contain printable ASCII")
        cell.string_nul_terminated = bool(separator)
        if separator and not prefix:
            cell.null_reason = "string_nul"
        else:
            cell.value = prefix.decode("ascii")
    elif kind == "L":
        cell.value_encoding = "boolean"
        if raw == b"\0":
            cell.null_reason = "logical_nul"
        elif raw in {b"T", b"F"}:
            cell.value = raw == b"T"
        else:
            raise ValueError("FITS binary logical field must be T, F or NUL")
    elif kind in {"B", "I", "J", "K"}:
        number = str(int.from_bytes(raw, "big", signed=kind != "B"))
        if number == column.null_declaration:
            cell.null_reason = "tnull"
        else:
            cell.value = number
    else:
        cell.value_encoding = "float_hex"
        floating = struct.unpack(">f" if kind == "E" else ">d", raw)[0]
        if math.isnan(floating):
            cell.null_reason = "ieee_nan"
        elif math.isinf(floating):
            raise ValueError("FITS infinite binary floats are outside the supported subset")
        else:
            cell.value = floating.hex()
    return cell


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(
        request.path, suffixes=(".fits", ".fit", ".fts"), max_bytes=8_000_000
    )
    if not source or len(source) % 2880:
        raise ValueError("FITS source must consist of complete 2880-byte blocks")
    primary, primary_cards, table_start = _header(source, 0)
    if [card[:8].strip() for card in primary_cards[:3]] != ["SIMPLE", "BITPIX", "NAXIS"]:
        raise ValueError("FITS primary mandatory cards are missing or out of order")
    _fixed_cards(primary_cards, primary, {"SIMPLE", "BITPIX", "NAXIS", "EXTEND"})
    if (
        primary.get("SIMPLE") != _Value("boolean", "T")
        or primary.get("EXTEND") != _Value("boolean", "T")
        or _integer(primary, "BITPIX") != 8
        or _integer(primary, "NAXIS") != 0
        or "XTENSION" in primary
        or any(re.fullmatch(r"NAXIS[0-9]+|GROUPS|PCOUNT|GCOUNT", key) for key in primary)
    ):
        raise ValueError("FITS requires an empty 8-bit primary with EXTEND=T and no random groups")
    table, table_cards, data_start = _header(source, table_start)
    required = ["XTENSION", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "PCOUNT", "GCOUNT", "TFIELDS"]
    if [card[:8].strip() for card in table_cards[:8]] != required:
        raise ValueError("FITS table mandatory cards are missing or out of order")
    _fixed_cards(table_cards, table, set(required))
    kind = _string(table, "XTENSION")
    if kind not in {"TABLE", "BINTABLE"}:
        raise ValueError("FITS extension must be TABLE or BINTABLE")
    if (
        _integer(table, "BITPIX") != 8
        or _integer(table, "NAXIS") != 2
        or _integer(table, "PCOUNT") != 0
        or _integer(table, "GCOUNT") != 1
        or any(
            re.fullmatch(r"NAXIS[0-9]+", key) and key not in {"NAXIS1", "NAXIS2"} for key in table
        )
        or "GROUPS" in table
        or "SIMPLE" in table
        or "EXTEND" in table
    ):
        raise ValueError("FITS table dimensions/groups/heaps are unsupported")
    row_bytes, row_count, count = (_integer(table, key) for key in ("NAXIS1", "NAXIS2", "TFIELDS"))
    if not 1 <= row_bytes <= 8192 or not 0 <= row_count <= 100_000 or not 1 <= count <= 64:
        raise ValueError("FITS table exceeds row width, row count or column limits")
    if row_count * count > 1_000_000 or request.limit * count > 10_000:
        raise ValueError("FITS table/page exceeds 1M/10000 cells")
    columns, gaps = _columns(table, count, row_bytes, kind == "BINTABLE", request.scaling_policy)
    column_mandatory = {"TFORM" + str(index) for index in range(1, count + 1)}
    if kind == "TABLE":
        column_mandatory.update("TBCOL" + str(index) for index in range(1, count + 1))
    _fixed_cards(table_cards, table, column_mandatory)
    data_size = row_bytes * row_count
    padded_end = data_start + ((data_size + 2879) // 2880) * 2880
    if padded_end != len(source):
        raise ValueError(
            "FITS table is truncated or contains unsupported extra HDUs/trailing bytes"
        )
    pad = b"\0" if kind == "BINTABLE" else b" "
    if source[data_start + data_size :].strip(pad):
        raise ValueError("FITS table padding contains non-padding bytes")
    rows: list[Row] = []
    page_bytes = 0
    for row_index in range(row_count):
        position = data_start + row_index * row_bytes
        raw = source[position : position + row_bytes]
        if kind == "TABLE" and (
            any(byte < 32 or byte > 126 for byte in raw)
            or any(raw[left:right].strip(b" ") for left, right in gaps)
        ):
            raise ValueError("FITS ASCII rows require printable data and blank inter-column gaps")
        selected = request.offset <= row_index < request.offset + request.limit
        cells: list[Cell] = []
        for column in columns:
            start = column.row_byte_start
            cell = _cell(
                raw[start : start + column.byte_width], column, kind == "BINTABLE", position + start
            )
            if selected:
                cells.append(cell)
        if selected:
            row = Row(row_index=row_index, source_byte_start=position, cells=cells)
            page_bytes += len(json.dumps(row.model_dump(), ensure_ascii=False).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("FITS encoded page exceeds 512 KB")
            rows.append(row)
    return Output(
        table_kind="TABLE" if kind == "TABLE" else "BINTABLE",
        columns=columns,
        row_bytes=row_bytes,
        row_count=row_count,
        has_rows=bool(row_count),
        table_data_byte_start=data_start,
        table_data_bytes=data_size,
        table_padding_bytes=padded_end - data_start - data_size,
        primary_header_cards=primary_cards,
        table_header_cards=table_cards,
        primary_header_sha256=hashlib.sha256(source[:table_start]).hexdigest(),
        table_header_sha256=hashlib.sha256(source[table_start:data_start]).hexdigest(),
        rows=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < row_count,
        scaling_policy=request.scaling_policy,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_fits_table",
    kind="plugin",
    description="Validate a bounded scalar ASCII/binary FITS table, preserve raw scaling/null metadata and exact paged cells with source-byte lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
