"""Read bounded PLY 1.0 scalar/list records in ASCII or either binary byte order.

Header and ASCII rows use LF/CRLF, printable ASCII plus horizontal tab; each
ASCII record occupies exactly one line. Standard int8/uint8/int16/uint16/int32/
uint32/float32/float64 names and char/uchar/short/ushort/int/uint/float/double
aliases are supported. Names are unique bounded ASCII identifiers per scope.
String properties, unknown directives, blank records and trailing records fail.

Integers are exact decimal strings. ASCII real tokens are exact decimal strings
within the declared type's finite range; they are not rounded to binary storage
precision, so tiny nonzero decimal values are retained. Binary floats use
float.hex, including signed zero. Nonfinite values fail throughout the file.
Raw property hex covers the scalar or list count plus items; record spans cover
the binary record or ASCII line including its line ending. All offsets zero based.

Conventional face vertex_index/vertex_indices integer lists are checked against
the declared vertex count, even if vertices occur later. They must have at least
three indices; repeated indices are retained and counted, never triangulated.
Other properties are parsed by their declarations without inferred semantics.
No geometric/manifold/winding or coordinate-unit validity is established.

Limits: 8 MB source, 64 KB header/1024 lines, 32 element types, 64 properties per
element/256 total, 100000 total records, 1M property occurrences and 2M scalar
reads including list counts, 4096 items/list, 65536 bytes/ASCII row, 128-byte
numeric tokens, 64-byte names; pages 200 records/10000 properties/512 KB JSON.
The entire declared structure and topology indices are validated before output.
Reference: Greg Turk, PLY_FILES.txt, Stanford/Georgia Tech PLY tools archive:
https://sites.cc.gatech.edu/projects/large_models/files/ply.tar.gz
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, cast

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")
_INT = re.compile(r"[+-]?[0-9]+\Z")
_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_ALIASES = {
    "char": "int8",
    "uchar": "uint8",
    "short": "int16",
    "ushort": "uint16",
    "int": "int32",
    "uint": "uint32",
    "float": "float32",
    "double": "float64",
}
_TYPES = {
    "int8": (1, "b"),
    "uint8": (1, "B"),
    "int16": (2, "h"),
    "uint16": (2, "H"),
    "int32": (4, "i"),
    "uint32": (4, "I"),
    "float32": (4, "f"),
    "float64": (8, "d"),
}


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Property(OutputModel):
    name: str
    item_type: str
    count_type: str | None


class Element(OutputModel):
    name: str
    record_count: int
    properties: list[Property]


class Value(OutputModel):
    property_name: str
    values: list[str]
    is_list: bool
    source_byte_start: int
    source_byte_length: int
    raw_hex: str


class Record(OutputModel):
    record_index: int
    element_name: str
    element_record_index: int
    source_byte_start: int
    source_byte_length: int
    values: list[Value]


class Output(OutputModel):
    format: Literal["ascii", "binary_little_endian", "binary_big_endian"]
    elements: list[Element]
    header_lines: list[str]
    header_bytes: int
    record_count: int
    has_records: bool
    scalar_reads: int
    conventional_face_list_count: int
    faces_with_repeated_indices: int
    conventional_face_property_declared: bool
    topology_indices_checked: bool
    geometric_validity_verified: Literal[False] = False
    ascii_float_rounding_applied: Literal[False] = False
    records: list[Record]
    offset: int
    has_more: bool
    full_supported_structure_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def _line(source: bytes, position: int, maximum: int, require_ending: bool) -> tuple[bytes, int]:
    end = source.find(b"\n", position, min(len(source), position + maximum + 2))
    if end < 0:
        if require_ending or len(source) - position > maximum:
            raise ValueError("PLY line is too long or missing its required line ending")
        end, following = len(source), len(source)
    else:
        following = end + 1
    line = source[position:end].removesuffix(b"\r")
    if len(line) > maximum or any(byte != 9 and not 32 <= byte <= 126 for byte in line):
        raise ValueError("PLY text lines require bounded printable ASCII/tab content")
    return line, following


def _type(name: str) -> str:
    name = _ALIASES.get(name, name)
    if name not in _TYPES:
        raise ValueError("PLY property type is unsupported")
    return name


def _header(source: bytes) -> tuple[str, list[Element], list[str], int]:
    elements: list[Element] = []
    lines: list[str] = []
    position = property_count = 0
    encoding = ""
    for index in range(1024):
        raw, position = _line(source, position, 4096, True)
        if position > 65_536:
            raise ValueError("PLY header exceeds 64 KB")
        text = raw.decode("ascii")
        lines.append(text)
        tokens = text.split()
        if index == 0:
            if text != "ply":
                raise ValueError("PLY magic must be the exact first line")
        elif index == 1:
            if (
                len(tokens) != 3
                or tokens[0] != "format"
                or tokens[2] != "1.0"
                or tokens[1] not in {"ascii", "binary_little_endian", "binary_big_endian"}
            ):
                raise ValueError("PLY requires a supported version 1.0 format line second")
            encoding = tokens[1]
        elif tokens and tokens[0] in {"comment", "obj_info"}:
            continue
        elif tokens == ["end_header"]:
            if not elements or any(not element.properties for element in elements):
                raise ValueError("PLY needs at least one element with properties")
            if (
                sum(element.record_count for element in elements) > 100_000
                or sum(element.record_count * len(element.properties) for element in elements)
                > 1_000_000
            ):
                raise ValueError("PLY declared record/property work exceeds its bound")
            return encoding, elements, lines, position
        elif len(tokens) == 3 and tokens[0] == "element":
            name, count = tokens[1:]
            if (
                not _NAME.fullmatch(name)
                or any(element.name == name for element in elements)
                or not re.fullmatch(r"[0-9]{1,6}", count)
                or int(count) > 100_000
                or len(elements) >= 32
            ):
                raise ValueError("PLY element name/count is invalid, repeated or exceeds its bound")
            elements.append(Element(name=name, record_count=int(count), properties=[]))
        elif tokens and tokens[0] == "property" and elements:
            if len(tokens) == 3:
                item_type, name = _type(tokens[1]), tokens[2]
                count_type = None
            elif len(tokens) == 5 and tokens[1] == "list":
                count_type, item_type, name = _type(tokens[2]), _type(tokens[3]), tokens[4]
                if count_type.startswith("float"):
                    raise ValueError("PLY list counts require an integer type")
            else:
                raise ValueError("PLY property declaration is malformed")
            properties = elements[-1].properties
            property_count += 1
            if (
                not _NAME.fullmatch(name)
                or any(prop.name == name for prop in properties)
                or len(properties) >= 64
                or property_count > 256
            ):
                raise ValueError("PLY property name/count is invalid or repeated")
            properties.append(Property(name=name, item_type=item_type, count_type=count_type))
        else:
            raise ValueError("PLY header directive is unsupported or out of order")
    raise ValueError("PLY header exceeds 1024 lines or lacks end_header")


@dataclass
class _Reader:
    source: bytes
    position: int
    endian: str
    scalar_reads: int = 0
    tokens: list[tuple[bytes, int]] | None = None
    token_index: int = 0

    def number(self, kind: str) -> tuple[str, int, int]:
        self.scalar_reads += 1
        if self.scalar_reads > 2_000_000:
            raise ValueError("PLY exceeds 2M scalar reads including list counts")
        width, code = _TYPES[kind]
        if self.tokens is not None:
            if self.token_index >= len(self.tokens):
                raise ValueError("PLY ASCII record has too few scalar tokens")
            raw, start = self.tokens[self.token_index]
            self.token_index += 1
            if len(raw) > 128:
                raise ValueError("PLY ASCII numeric token exceeds 128 bytes")
            token = raw.decode("ascii")
            if kind.startswith("float"):
                if not _REAL.fullmatch(token):
                    raise ValueError("PLY ASCII real token must be finite decimal notation")
                # Bound exponent spelling before Decimal construction; conversion
                # never uses a context-dependent arithmetic/rounding operation.
                exponent = re.split("[eE]", token)[1:]
                if exponent and (len(exponent[0]) > 6 or abs(int(exponent[0])) > 10_000):
                    raise ValueError("PLY real decimal exponent exceeds 10000")
                decimal = Decimal(token)
                largest = ((1 << 24) - 1) << 104 if kind == "float32" else ((1 << 53) - 1) << 971
                if decimal.copy_abs() > Decimal(largest):
                    raise ValueError("PLY real token exceeds declared finite type range")
                value = str(decimal)
            else:
                if not _INT.fullmatch(token):
                    raise ValueError("PLY integer token has invalid syntax")
                integer = int(token)
                unsigned = kind.startswith("u")
                low = 0 if unsigned else -(1 << (8 * width - 1))
                high = (1 << (8 * width - (0 if unsigned else 1))) - 1
                if not low <= integer <= high:
                    raise ValueError("PLY integer exceeds its declared type")
                value = str(integer)
            return value, start, start + len(raw)
        start = self.position
        self.position += width
        if self.position > len(self.source):
            raise ValueError("PLY binary record is truncated")
        raw = self.source[start : self.position]
        if kind.startswith("float"):
            floating: float = struct.unpack(self.endian + code, raw)[0]
            if not math.isfinite(floating):
                raise ValueError("PLY nonfinite binary values are unsupported")
            return floating.hex(), start, self.position
        return (
            str(
                int.from_bytes(
                    raw, "little" if self.endian == "<" else "big", signed=not kind.startswith("u")
                )
            ),
            start,
            self.position,
        )


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".ply",), max_bytes=8_000_000)
    encoding, elements, lines, header_end = _header(source)
    reader = _Reader(source, header_end, "<" if encoding == "binary_little_endian" else ">")
    vertex_count = next(
        (element.record_count for element in elements if element.name == "vertex"), None
    )
    faces = [
        prop
        for element in elements
        if element.name == "face"
        for prop in element.properties
        if prop.name in {"vertex_index", "vertex_indices"}
    ]
    if (
        len(faces) > 1
        or any(prop.count_type is None or prop.item_type.startswith("float") for prop in faces)
        or faces
        and vertex_count is None
    ):
        raise ValueError(
            "PLY conventional face indices require one integer list and declared vertices"
        )
    rows: list[Record] = []
    index = page_bytes = face_count = repeated = 0
    for element in elements:
        for local_index in range(element.record_count):
            start = reader.position
            if encoding == "ascii":
                line, next_line = _line(source, start, 65_536, False)
                reader.tokens = [
                    (match.group(), start + match.start())
                    for match in re.finditer(rb"[^ \t]+", line)
                ]
                reader.token_index = 0
            selected = request.offset <= index < request.offset + request.limit
            values: list[Value] = []
            for prop in element.properties:
                count = 1
                list_start: int | None = None
                if prop.count_type is not None:
                    token, list_start, end = reader.number(prop.count_type)
                    count = int(token)
                    if not 0 <= count <= 4096:
                        raise ValueError("PLY list count is negative or exceeds 4096")
                entries: list[str] = []
                value_start = list_start
                for _ in range(count):
                    token, item_start, end = reader.number(prop.item_type)
                    if value_start is None:
                        value_start = item_start
                    entries.append(token)
                if element.name == "face" and prop.name in {"vertex_index", "vertex_indices"}:
                    if (
                        count < 3
                        or vertex_count is None
                        or any(not 0 <= int(value) < vertex_count for value in entries)
                    ):
                        raise ValueError(
                            "PLY face list is too short or references an unknown vertex"
                        )
                    face_count += 1
                    repeated += len(set(entries)) < len(entries)
                if selected:
                    assert value_start is not None
                    values.append(
                        Value(
                            property_name=prop.name,
                            values=entries,
                            is_list=prop.count_type is not None,
                            source_byte_start=value_start,
                            source_byte_length=end - value_start,
                            raw_hex=source[value_start:end].hex(),
                        )
                    )
            if encoding == "ascii":
                if reader.tokens is None or reader.token_index != len(reader.tokens):
                    raise ValueError("PLY ASCII record has extra tokens")
                reader.position = next_line
            if selected:
                row = Record(
                    record_index=index,
                    element_name=element.name,
                    element_record_index=local_index,
                    source_byte_start=start,
                    source_byte_length=reader.position - start,
                    values=values,
                )
                page_bytes += len(json.dumps(row.model_dump()).encode("utf-8"))
                if (
                    page_bytes > 512_000
                    or sum(len(item.values) for item in rows) + len(values) > 10_000
                ):
                    raise ValueError("PLY page exceeds 512 KB or 10000 properties")
                rows.append(row)
            index += 1
    if reader.position != len(source):
        raise ValueError("PLY has undeclared trailing data/records")
    return Output(
        format=cast(Literal["ascii", "binary_little_endian", "binary_big_endian"], encoding),
        elements=elements,
        header_lines=lines,
        header_bytes=header_end,
        record_count=index,
        has_records=bool(index),
        scalar_reads=reader.scalar_reads,
        conventional_face_list_count=face_count,
        faces_with_repeated_indices=repeated,
        conventional_face_property_declared=bool(faces),
        topology_indices_checked=bool(face_count),
        records=rows,
        offset=request.offset,
        has_more=request.offset + len(rows) < index,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_ply",
    kind="plugin",
    description="Parse bounded ASCII/binary PLY scalar/list records and declared face indices, preserving exact values and source spans without geometric validity claims.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
