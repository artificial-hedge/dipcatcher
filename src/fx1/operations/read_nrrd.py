"""Read a bounded inline scalar NRRD0001..0005 raster, raw or ASCII/text/txt.

Required type/dimension/sizes/encoding are parsed independently. Dimensions are
1..8, positive sizes follow fastest-to-slowest axis order. Raw multibyte data
requires little/big endian; ASCII and one-byte samples do not use byte order.
Integers are exact decimal strings; finite raw IEEE values use float.hex with
original bytes. ASCII reals remain exact decimal tokens within the declared
finite range, without rounding or underflow to declared binary precision.
Nonfinite data values fail, including values outside the requested page.

Optional content, number (opaque deprecated field), sample units, labels/units,
spacings and axis mins/maxs are supported. Per-axis fields must follow dimension;
axis numeric metadata permits nan as unknown, but not infinity; spacing cannot
be zero. Metadata describes declarations only: spacing/bounds consistency and
units are not inferred. Quoted labels/units support escaped double quotes.
Custom key:=value metadata decodes only backslash-n and doubled backslashes,
keeps the final duplicate value and reports overwrite count (NRRD version>=2).
Custom keys cannot contain a colon-space delimiter. Standard field delimiters
before the first key/value delimiter take precedence, so content can include :=.
All original header lines remain available. Other fields, orientation/measurement
frames, detached references, skip fields, block types and compression fail.

Raw payload must have the exact declared byte count. ASCII requires exactly the
declared scalar count with ASCII whitespace separators and permits final white
space. Header LF/CRLF is supported; source offsets and sample indices are zero
based. Limits:8 MB source,64 KB/512 header lines,4096 bytes/header line,1M samples,
128-byte numeric tokens with exponent magnitude<=10000,200 samples/512 KB page.
No dense array, codec execution, external file or world-coordinate transform.
Reference: https://teem.sourceforge.net/nrrd/format.html
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from decimal import Decimal
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_REAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_TYPE_ALIASES = {
    "signed char": "int8",
    "int8_t": "int8",
    "uchar": "uint8",
    "unsigned char": "uint8",
    "uint8_t": "uint8",
    "short": "int16",
    "short int": "int16",
    "signed short": "int16",
    "signed short int": "int16",
    "int16_t": "int16",
    "ushort": "uint16",
    "unsigned short": "uint16",
    "unsigned short int": "uint16",
    "uint16_t": "uint16",
    "int": "int32",
    "signed int": "int32",
    "int32_t": "int32",
    "uint": "uint32",
    "unsigned int": "uint32",
    "uint32_t": "uint32",
    "longlong": "int64",
    "long long": "int64",
    "long long int": "int64",
    "signed long long": "int64",
    "signed long long int": "int64",
    "int64_t": "int64",
    "ulonglong": "uint64",
    "unsigned long long": "uint64",
    "unsigned long long int": "uint64",
    "uint64_t": "uint64",
}
_TYPES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float": 4,
    "double": 8,
}
_FIELD_ALIASES = {"axismins": "axis mins", "axismaxs": "axis maxs", "sampleunits": "sample units"}
_PER_AXIS = {"sizes", "labels", "units", "spacings", "axis mins", "axis maxs"}
_FIELDS = {
    "type",
    "dimension",
    "encoding",
    "endian",
    "content",
    "number",
    "sample units",
} | _PER_AXIS


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=1_000_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class CustomMetadata(OutputModel):
    key: str
    value: str


class Axis(OutputModel):
    axis_index: int
    size: int
    element_stride: int
    label: str | None
    unit: str | None
    spacing: str | None
    minimum: str | None
    maximum: str | None


class Sample(OutputModel):
    sample_index: int
    coordinate: list[int]
    source_byte_start: int
    source_byte_length: int
    raw_hex: str
    value: str


class Output(OutputModel):
    version: int
    scalar_type: str
    item_bytes: int
    encoding: Literal["raw", "ascii"]
    endian_declaration: str | None
    decoded_byte_order: Literal["little", "big", "not_applicable"]
    axes: list[Axis]
    sample_count: int
    raw_equivalent_bytes: int
    data_byte_start: int
    data_bytes: int
    header_lines: list[str]
    content_declaration: str | None
    sample_units_declaration: str | None
    deprecated_number_declaration: str | None
    custom_metadata: list[CustomMetadata]
    overwritten_custom_key_count: int
    samples: list[Sample]
    offset: int
    has_more: bool
    axis_order: Literal["first_axis_fastest"] = "first_axis_fastest"
    ascii_binary_quantization_applied: Literal[False] = False
    world_coordinate_transform_applied: Literal[False] = False
    metadata_semantics_verified: Literal[False] = False
    external_references_followed: Literal[False] = False
    full_supported_source_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


def _unescape(text: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        index += 1
        if char == "\\":
            if index == len(text) or text[index] not in {"n", "\\"}:
                raise ValueError("NRRD custom metadata uses an unsupported escape")
            char = "\n" if text[index] == "n" else "\\"
            index += 1
        result.append(char)
    return "".join(result)


def _header(source: bytes) -> tuple[int, dict[str, str], dict[str, str], int, list[str], int]:
    fields: dict[str, str] = {}
    custom: dict[str, str] = {}
    lines: list[str] = []
    position = overwritten = version = 0
    for index in range(512):
        end = source.find(b"\n", position, min(len(source), position + 4098))
        if end < 0:
            raise ValueError("NRRD header is truncated or line exceeds4096 bytes")
        raw = source[position:end].removesuffix(b"\r")
        position = end + 1
        if (
            len(raw) > 4096
            or position > 65_536
            or any(byte != 9 and not 32 <= byte <= 126 for byte in raw)
        ):
            raise ValueError("NRRD header exceeds bounds or printable ASCII/tab syntax")
        text = raw.decode("ascii")
        lines.append(text)
        if index == 0:
            if not re.fullmatch(r"NRRD000[1-5]", text):
                raise ValueError("NRRD magic must be NRRD0001 through NRRD0005")
            version = int(text[-1])
        elif not text:
            return version, fields, custom, overwritten, lines, position
        elif text.startswith("#"):
            continue
        elif ":=" in text and (": " not in text or text.index(":=") < text.index(": ")):
            key, value = text.split(":=", 1)
            key, value = _unescape(key), _unescape(value)
            if version < 2 or not key or len(key) > 256:
                raise ValueError("NRRD custom keys need version>=2 and1..256 characters")
            overwritten += key in custom
            custom[key] = value
        else:
            if ": " not in text:
                raise ValueError("NRRD field must use a colon followed by a space")
            key, value = text.split(": ", 1)
            key = _FIELD_ALIASES.get(key.lower(), key.lower())
            if key not in _FIELDS or key in fields:
                raise ValueError("NRRD field is unsupported or repeated")
            if key in _PER_AXIS and "dimension" not in fields:
                raise ValueError("NRRD per-axis fields must follow dimension")
            if key == "sample units" and version < 4:
                raise ValueError("NRRD sample units needs version>=4")
            fields[key] = value.rstrip(" \t")
    raise ValueError("NRRD header exceeds512 lines or lacks a blank-line terminator")


def _positive(token: str, maximum: int) -> int:
    if not re.fullmatch(r"[0-9]{1,7}", token) or not 1 <= int(token) <= maximum:
        raise ValueError("NRRD dimensions/sizes require bounded positive decimal integers")
    return int(token)


def _real(token: str, width: int) -> str:
    if len(token) > 128 or not _REAL.fullmatch(token):
        raise ValueError("NRRD real values require bounded finite decimal syntax")
    exponent = re.split("[eE]", token)[1:]
    if exponent and (len(exponent[0]) > 6 or abs(int(exponent[0])) > 10_000):
        raise ValueError("NRRD decimal exponent exceeds10000")
    value = Decimal(token)
    largest = ((1 << 24) - 1) << 104 if width == 4 else ((1 << 53) - 1) << 971
    if value.copy_abs() > Decimal(largest):
        raise ValueError("NRRD decimal magnitude exceeds its declared type")
    return str(value)


def _quoted(text: str, dimension: int) -> list[str]:
    values: list[str] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position] in " \t":
            position += 1
        if position == len(text):
            break
        if text[position] != '"':
            raise ValueError("NRRD axis labels/units require quoted strings")
        position += 1
        value: list[str] = []
        while position < len(text) and text[position] != '"':
            if text[position : position + 2] == '\\"':
                value.append('"')
                position += 2
            else:
                value.append(text[position])
                position += 1
        if position == len(text):
            raise ValueError("NRRD quoted axis string is unterminated")
        position += 1
        if position < len(text) and text[position] not in " \t":
            raise ValueError("NRRD quoted axis strings need whitespace separators")
        values.append("".join(value))
    if len(values) != dimension:
        raise ValueError("NRRD axis label/unit count differs from dimension")
    return values


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".nrrd",), max_bytes=8_000_000)
    version, fields, custom, overwritten, lines, start = _header(source)
    if not {"type", "dimension", "sizes", "encoding"} <= fields.keys():
        raise ValueError("NRRD required type/dimension/sizes/encoding field is missing")
    dimension = _positive(fields["dimension"], 8)
    sizes = [_positive(value, 1_000_000) for value in fields["sizes"].split()]
    if len(sizes) != dimension or math.prod(sizes) > 1_000_000:
        raise ValueError("NRRD sizes mismatch dimension or exceed1M samples")
    count = math.prod(sizes)
    type_token = " ".join(fields["type"].lower().split())
    kind = _TYPE_ALIASES.get(type_token, type_token)
    if kind not in _TYPES:
        raise ValueError("NRRD scalar type is unsupported")
    width = _TYPES[kind]
    encoding = fields["encoding"].lower()
    if encoding in {"text", "txt"}:
        encoding = "ascii"
    if encoding not in {"raw", "ascii"}:
        raise ValueError("NRRD supports uncompressed raw or ASCII encoding only")
    endian = fields.get("endian")
    if endian is not None:
        endian = endian.lower()
        if endian not in {"little", "big"}:
            raise ValueError("NRRD endian must be little or big")
    if encoding == "raw" and width > 1 and endian is None:
        raise ValueError("NRRD raw multibyte data requires explicit endian")
    labels = _quoted(fields["labels"], dimension) if "labels" in fields else [None] * dimension
    units = _quoted(fields["units"], dimension) if "units" in fields else [None] * dimension
    numeric_axes: dict[str, list[str | None]] = {}
    for key in ("spacings", "axis mins", "axis maxs"):
        values: list[str | None] = [None] * dimension
        if key in fields:
            tokens = fields[key].split()
            if len(tokens) != dimension:
                raise ValueError("NRRD axis numeric metadata must match dimension")
            values = ["nan" if token.lower() == "nan" else _real(token, 8) for token in tokens]
            if key == "spacings" and any(
                value != "nan" and Decimal(value or "0") == 0 for value in values
            ):
                raise ValueError("NRRD axis spacing cannot be zero")
        numeric_axes[key] = values
    axes: list[Axis] = []
    stride = 1
    for axis, size in enumerate(sizes):
        axes.append(
            Axis(
                axis_index=axis,
                size=size,
                element_stride=stride,
                label=labels[axis],
                unit=units[axis],
                spacing=numeric_axes["spacings"][axis],
                minimum=numeric_axes["axis mins"][axis],
                maximum=numeric_axes["axis maxs"][axis],
            )
        )
        stride *= size
    if encoding == "raw":
        if len(source) - start != count * width:
            raise ValueError("NRRD raw payload does not exactly match declared samples")
        spans = ((start + index * width, start + (index + 1) * width) for index in range(count))
    else:
        spans = (
            (match.start(), match.end()) for match in re.compile(rb"\S+").finditer(source, start)
        )
    samples: list[Sample] = []
    observed = page_bytes = 0
    for left, right in spans:
        if observed >= count:
            raise ValueError("NRRD ASCII payload has extra scalar tokens")
        raw = source[left:right]
        if encoding == "raw":
            if kind in {"float", "double"}:
                floating: float = struct.unpack(
                    ("<" if endian == "little" else ">") + ("f" if width == 4 else "d"), raw
                )[0]
                if not math.isfinite(floating):
                    raise ValueError("NRRD nonfinite raw scalars are unsupported")
                value = floating.hex()
            else:
                value = str(
                    int.from_bytes(
                        raw,
                        "little" if endian == "little" else "big",
                        signed=kind.startswith("int"),
                    )
                )
        else:
            if len(raw) > 128:
                raise ValueError("NRRD ASCII scalar token exceeds128 bytes")
            token = raw.decode("ascii")
            if kind in {"float", "double"}:
                value = _real(token, width)
            else:
                if not _INTEGER.fullmatch(token):
                    raise ValueError("NRRD integer scalar has invalid syntax")
                integer = int(token)
                signed = kind.startswith("int")
                low, high = (
                    (-(1 << (8 * width - 1)), (1 << (8 * width - 1)) - 1)
                    if signed
                    else (0, (1 << (8 * width)) - 1)
                )
                if not low <= integer <= high:
                    raise ValueError("NRRD integer scalar exceeds its declared type")
                value = str(integer)
        if request.offset <= observed < request.offset + request.limit:
            remainder = observed
            coordinate: list[int] = []
            for size in sizes:
                remainder, location = divmod(remainder, size)
                coordinate.append(location)
            sample = Sample(
                sample_index=observed,
                coordinate=coordinate,
                source_byte_start=left,
                source_byte_length=right - left,
                raw_hex=raw.hex(),
                value=value,
            )
            page_bytes += len(json.dumps(sample.model_dump()).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("NRRD encoded sample page exceeds512 KB")
            samples.append(sample)
        observed += 1
    if observed != count:
        raise ValueError("NRRD ASCII payload has too few scalar tokens")
    return Output(
        version=version,
        scalar_type=kind,
        item_bytes=width,
        encoding="raw" if encoding == "raw" else "ascii",
        endian_declaration=endian,
        decoded_byte_order="not_applicable"
        if encoding == "ascii" or width == 1
        else "little"
        if endian == "little"
        else "big",
        axes=axes,
        sample_count=count,
        raw_equivalent_bytes=count * width,
        data_byte_start=start,
        data_bytes=len(source) - start,
        header_lines=lines,
        content_declaration=fields.get("content"),
        sample_units_declaration=fields.get("sample units"),
        deprecated_number_declaration=fields.get("number"),
        custom_metadata=[CustomMetadata(key=key, value=value) for key, value in custom.items()],
        overwritten_custom_key_count=overwritten,
        samples=samples,
        offset=request.offset,
        has_more=request.offset + len(samples) < count,
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_nrrd",
    kind="plugin",
    description="Parse bounded inline raw/ASCII NRRD scalar rasters with explicit axis order, retained metadata declarations and exact paged source spans.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
