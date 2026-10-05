"""Read a bounded single-solid ASCII or unextended little-endian binary STL.

The caller selects the encoding: a binary header beginning with 'solid' never
triggers a heuristic format switch. Binary layout is an opaque 80-byte header,
uint32 triangle count and exactly 50 bytes/triangle (12 IEEE float32 values plus
a zero uint16 attribute word). Color/attribute extensions and trailing bytes
fail. ASCII accepts lowercase keywords, space indentation, LF/CRLF, blank lines,
one optional printable name matched by endsolid, and exactly three vertices per
facet. Tabs, comments, multiple solids and extra tokens fail. Empty solids are
accepted as no observations. Negative coordinates are allowed deliberately.

Every scalar must be finite. Binary values use float.hex, including signed zero;
ASCII values preserve exact decimal values and raw spelling without rounding to
float32. Decimal magnitude is bounded by finite float32 maximum, token length by
128 and explicit exponent magnitude by 100. All source spans are zero based;
scalar spans exclude whitespace, triangle spans include their grammar/line ends.

Geometry observations use exact rational arithmetic on these scalar values:
repeated vertices, zero cross-product area, exact unit/zero declared normals and
the sign of normal dot (v1-v0) cross (v2-v0). A positive dot does not establish
parallelism. The coordinate bounds are exact rational strings. There is no
tolerance, normalization, welding, triangulation or unit inference; adjacency,
watertightness, manufacturing suitability and overall mesh validity are unchecked.

Limits: 8 MB source, 10000 triangles, 100000 ASCII lines, 4096 bytes/line,
256-byte solid name, 200 triangles/512 KB encoded page. The entire supported
structure and every triangle are validated before pagination returns anything.
Primary implementation reference (layout): Blender's intern/stl_data.hh and
importer/stl_import_binary_reader.cc at https://github.com/blender/blender/tree/main/
source/blender/io/stl . Format origin: 3D Systems StereoLithography Interface
Specification (October 1989), described at https://www.fabbers.com/tech/STL_Format .
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_REAL = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_MAXIMUM = Decimal(((1 << 24) - 1) << 104)


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    format: Literal["ascii", "binary"]
    offset: int = Field(default=0, strict=True, ge=0, le=10_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Scalar(OutputModel):
    value: str
    source_byte_start: int
    source_byte_length: int
    raw_hex: str


class Triangle(OutputModel):
    triangle_index: int
    source_byte_start: int
    source_byte_length: int
    normal: list[Scalar]
    vertices: list[list[Scalar]]
    has_repeated_vertex: bool
    is_exactly_degenerate: bool
    declared_normal_is_exactly_zero: bool
    declared_normal_is_exactly_unit: bool
    normal_winding_dot_sign: Literal[-1, 0, 1]


class Output(OutputModel):
    format: Literal["ascii", "binary"]
    value_encoding: Literal["exact_decimal", "float_hex"]
    solid_name: str | None
    binary_header_hex: str | None
    triangle_count: int
    has_triangles: bool
    repeated_vertex_triangles: int
    exactly_degenerate_triangles: int
    exactly_zero_declared_normals: int
    exactly_unit_declared_normals: int
    positive_normal_winding_dot: int
    negative_normal_winding_dot: int
    zero_normal_winding_dot: int
    coordinate_minimum: list[str] | None
    coordinate_maximum: list[str] | None
    coordinate_bound_encoding: Literal["exact_rational"] = "exact_rational"
    geometry_arithmetic: Literal["exact_rational_on_reported_scalar_values"] = (
        "exact_rational_on_reported_scalar_values"
    )
    ascii_float32_quantization_applied: Literal[False] = False
    adjacency_checked: Literal[False] = False
    mesh_validity_verified: Literal[False] = False
    coordinate_units: Literal["unspecified"] = "unspecified"
    format_inferred: Literal[False] = False
    triangles: list[Triangle]
    offset: int
    has_more: bool
    full_supported_structure_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


@dataclass
class _Facet:
    start: int
    end: int
    scalars: list[Scalar]
    values: list[Fraction]


class _Text:
    def __init__(self, source: bytes) -> None:
        self.source = source
        self.position = 0
        self.lines = 0

    def line(self) -> tuple[int, bytes, int] | None:
        while self.position < len(self.source):
            self.lines += 1
            if self.lines > 100_000:
                raise ValueError("STL ASCII exceeds 100000 lines")
            start = self.position
            end = self.source.find(b"\n", start, min(len(self.source), start + 4098))
            if end < 0:
                if len(self.source) - start > 4096:
                    raise ValueError("STL ASCII line exceeds 4096 bytes")
                end = self.position = len(self.source)
            else:
                self.position = end + 1
            raw = self.source[start:end].removesuffix(b"\r")
            if len(raw) > 4096 or any(not 32 <= byte <= 126 for byte in raw):
                raise ValueError("STL ASCII requires printable characters and spaces only")
            if raw.strip(b" "):
                return start, raw, self.position
        return None

    def required(self) -> tuple[int, bytes, int]:
        line = self.line()
        if line is None:
            raise ValueError("STL ASCII grammar is truncated")
        return line


def _name(raw: bytes, keyword: bytes) -> str:
    text = raw.strip(b" ")
    if text == keyword:
        return ""
    if not text.startswith(keyword + b" "):
        raise ValueError("STL ASCII solid delimiter is invalid")
    name = text[len(keyword) + 1 :].strip(b" ")
    if len(name) > 256:
        raise ValueError("STL ASCII solid name exceeds 256 bytes")
    return name.decode("ascii")


def _ascii_values(
    line: tuple[int, bytes, int], prefix: list[bytes]
) -> tuple[list[Scalar], list[Fraction]]:
    start, raw, _ = line
    tokens = list(re.finditer(rb"[^ ]+", raw))
    if (
        len(tokens) != len(prefix) + 3
        or [token.group() for token in tokens[: len(prefix)]] != prefix
    ):
        raise ValueError("STL ASCII normal/vertex line needs exactly three scalars")
    scalars: list[Scalar] = []
    values: list[Fraction] = []
    for match in tokens[len(prefix) :]:
        token = match.group()
        if len(token) > 128 or not _REAL.fullmatch(token):
            raise ValueError("STL ASCII scalar requires bounded finite decimal syntax")
        exponent = re.split(rb"[eE]", token)[1:]
        if exponent and (len(exponent[0]) > 4 or abs(int(exponent[0])) > 100):
            raise ValueError("STL ASCII decimal exponent magnitude exceeds 100")
        decimal = Decimal(token.decode("ascii"))
        if decimal.copy_abs() > _MAXIMUM:
            raise ValueError("STL ASCII scalar exceeds finite float32 magnitude")
        scalars.append(
            Scalar(
                value=str(decimal),
                source_byte_start=start + match.start(),
                source_byte_length=len(token),
                raw_hex=token.hex(),
            )
        )
        values.append(Fraction(decimal))
    return scalars, values


def _ascii_facets(reader: _Text, name: str) -> Iterator[_Facet]:
    count = 0
    while True:
        first = reader.required()
        if first[1].strip(b" ").startswith(b"endsolid"):
            if _name(first[1], b"endsolid") != name:
                raise ValueError("STL ASCII endsolid name differs from solid name")
            if reader.line() is not None:
                raise ValueError("STL ASCII trailing content or multiple solids are unsupported")
            return
        count += 1
        if count > 10_000:
            raise ValueError("STL exceeds 10000 triangles")
        scalars, values = _ascii_values(first, [b"facet", b"normal"])
        if reader.required()[1].split() != [b"outer", b"loop"]:
            raise ValueError("STL ASCII facet is missing outer loop")
        for _ in range(3):
            vertex_scalars, vertex_values = _ascii_values(reader.required(), [b"vertex"])
            scalars.extend(vertex_scalars)
            values.extend(vertex_values)
        if reader.required()[1].strip(b" ") != b"endloop":
            raise ValueError("STL ASCII facet is missing endloop")
        last = reader.required()
        if last[1].strip(b" ") != b"endfacet":
            raise ValueError("STL ASCII facet is missing endfacet")
        yield _Facet(first[0], last[2], scalars, values)


def _binary_facets(source: bytes) -> Iterator[_Facet]:
    if len(source) < 84:
        raise ValueError("STL binary header is truncated")
    count = int.from_bytes(source[80:84], "little")
    if count > 10_000 or len(source) != 84 + 50 * count:
        raise ValueError("STL binary count exceeds 10000 or source extent differs from count")
    for index in range(count):
        start = 84 + 50 * index
        if source[start + 48 : start + 50] != b"\0\0":
            raise ValueError("STL binary color/attribute extensions are unsupported")
        scalars: list[Scalar] = []
        values: list[Fraction] = []
        for component in range(12):
            position = start + 4 * component
            raw = source[position : position + 4]
            value: float = struct.unpack("<f", raw)[0]
            if not math.isfinite(value):
                raise ValueError("STL binary scalar must be finite")
            scalars.append(
                Scalar(
                    value=value.hex(),
                    source_byte_start=position,
                    source_byte_length=4,
                    raw_hex=raw.hex(),
                )
            )
            values.append(Fraction(value))
        yield _Facet(start, start + 50, scalars, values)


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".stl",), max_bytes=8_000_000)
    name: str | None = None
    header: str | None = None
    if request.format == "ascii":
        reader = _Text(source)
        name = _name(reader.required()[1], b"solid")
        facets = _ascii_facets(reader, name)
    else:
        facets = _binary_facets(source)
        header = source[:80].hex()
    page: list[Triangle] = []
    minimum: list[Fraction] | None = None
    maximum: list[Fraction] | None = None
    total = repeated_count = degenerate_count = zero_count = unit_count = page_bytes = 0
    signs = {-1: 0, 0: 0, 1: 0}
    for facet in facets:
        normal = facet.values[:3]
        vertices = [facet.values[index : index + 3] for index in (3, 6, 9)]
        left = [vertices[1][axis] - vertices[0][axis] for axis in range(3)]
        right = [vertices[2][axis] - vertices[0][axis] for axis in range(3)]
        cross = [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
        repeated = len({tuple(vertex) for vertex in vertices}) < 3
        degenerate = all(value == 0 for value in cross)
        zero = all(value == 0 for value in normal)
        unit = sum(value * value for value in normal) == 1
        dot = sum(normal[axis] * cross[axis] for axis in range(3))
        sign: Literal[-1, 0, 1] = -1 if dot < 0 else 1 if dot > 0 else 0
        signs[sign] += 1
        repeated_count += repeated
        degenerate_count += degenerate
        zero_count += zero
        unit_count += unit
        for vertex in vertices:
            if minimum is None or maximum is None:
                minimum, maximum = vertex.copy(), vertex.copy()
            else:
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], vertex[axis])
                    maximum[axis] = max(maximum[axis], vertex[axis])
        if request.offset <= total < request.offset + request.limit:
            triangle = Triangle(
                triangle_index=total,
                source_byte_start=facet.start,
                source_byte_length=facet.end - facet.start,
                normal=facet.scalars[:3],
                vertices=[facet.scalars[index : index + 3] for index in (3, 6, 9)],
                has_repeated_vertex=repeated,
                is_exactly_degenerate=degenerate,
                declared_normal_is_exactly_zero=zero,
                declared_normal_is_exactly_unit=unit,
                normal_winding_dot_sign=sign,
            )
            page_bytes += len(json.dumps(triangle.model_dump()).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("STL encoded triangle page exceeds 512 KB")
            page.append(triangle)
        total += 1
    return Output(
        format=request.format,
        value_encoding="exact_decimal" if request.format == "ascii" else "float_hex",
        solid_name=name,
        binary_header_hex=header,
        triangle_count=total,
        has_triangles=total > 0,
        repeated_vertex_triangles=repeated_count,
        exactly_degenerate_triangles=degenerate_count,
        exactly_zero_declared_normals=zero_count,
        exactly_unit_declared_normals=unit_count,
        positive_normal_winding_dot=signs[1],
        negative_normal_winding_dot=signs[-1],
        zero_normal_winding_dot=signs[0],
        coordinate_minimum=None if minimum is None else [str(value) for value in minimum],
        coordinate_maximum=None if maximum is None else [str(value) for value in maximum],
        triangles=page,
        offset=request.offset,
        has_more=request.offset + len(page) < total,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_stl",
    kind="plugin",
    description="Read bounded explicit-format STL triangles with exact scalars, raw spans and limited exact geometry observations.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
