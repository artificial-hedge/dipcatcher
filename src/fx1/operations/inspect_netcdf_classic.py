"""Inspect a bounded strict subset of big-endian CDF1/CDF2 NetCDF metadata.

Specification: docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html.
Parse dimension, global-attribute and variable lists directly, including scalar
variables and one unlimited dimension used only first in record variables.
Names are bounded NFC UTF-8 with classic name syntax; duplicate names within a
scope and nonzero header padding fail. Numeric attribute bytes are retained as
hex alongside exact signed-integer/hex-float strings; nonfinite IEEE encodings
remain strings plus original bits. CHAR attributes are raw hex without inferred
text encoding. No scale, missing-value, unsigned-byte or other CF conventions
are applied, and variable arrays/data values are never decoded or allocated.

vsize must equal the four-byte-rounded fixed-variable or record-slab size.
Fixed variables must follow header order without overlap; aligned gaps are
allowed and reported without inspecting their bytes. Record variables must
interleave contiguously in header order. A lone BYTE/CHAR/SHORT record variable
uses its unpadded slab as stride, while its vsize retains the format's padding.
Data extents must fit exactly: trailing payload/preallocated records fail. With
zero records, EOF may be the final fixed-data end or the planned first record
start; later record-variable offsets may describe future bytes beyond EOF.

Unsupported: CDF5, HDF5/NetCDF4, compression, streaming record counts, oversized
vsize sentinels and slabs larger than 2^31-4 bytes. Counts/dimension lengths are
at most 1e9. Limits: 2 MB header prefix, 256 dimensions, 256 variables, rank 16,
128-byte names, 128 attributes/scope, 512 attributes total, 32768 raw bytes per
attribute/128 KB combined, and 8192 numeric attribute values. Variable pages
contain at most 64 entries. Header mode accepts contained files through 1 TB;
full mode through 8 MB adds a whole-source hash. Header hashes cover only parsed
header bytes, not gaps or payload. Prefix reads may include uninterpreted data.
Descriptor checks do not guarantee an atomic source snapshot or payload validity.
"""

from __future__ import annotations

import hashlib
import struct
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_TYPES = {
    1: ("NC_BYTE", 1),
    2: ("NC_CHAR", 1),
    3: ("NC_SHORT", 2),
    4: ("NC_INT", 4),
    5: ("NC_FLOAT", 4),
    6: ("NC_DOUBLE", 8),
}
_MAX_SLAB = (1 << 31) - 4


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    read_mode: Literal["header", "full"] = "header"
    max_file_bytes: int = Field(default=1_000_000_000_000, strict=True, ge=32, le=1_000_000_000_000)
    variable_offset: int = Field(default=0, strict=True, ge=0, le=256)
    variable_limit: int = Field(default=20, strict=True, ge=1, le=64)


class Dimension(OutputModel):
    index: int
    name: str
    declared_length: int
    unlimited: bool
    resolved_length: int


class Attribute(OutputModel):
    name: str
    type: str
    element_count: int
    raw_hex: str
    numeric_values: list[str] | None
    encoding: Literal["raw_char_bytes", "signed_decimal_strings", "ieee_hex_float_strings"]


class Variable(OutputModel):
    index: int
    name: str
    type: str
    dimension_ids: list[int]
    shape: list[int]
    attributes: list[Attribute]
    record_variable: bool
    begin: int
    declared_vsize: int
    elements_per_slab: int
    data_bytes_per_slab: int
    total_elements: int
    record_stride: int | None
    last_data_end_exclusive: int | None
    allocated_end_exclusive: int | None


class Output(OutputModel):
    format: Literal["CDF1", "CDF2"]
    declared_record_count: int
    dimensions: list[Dimension]
    global_attributes: list[Attribute]
    variable_count: int
    fixed_variable_count: int
    record_variable_count: int
    record_start: int | None
    record_stride: int | None
    unpadded_single_record_variable: bool
    uninterpreted_gap_bytes: int
    variables: list[Variable]
    variable_offset: int
    has_more_variables: bool
    attribute_count: int
    raw_attribute_bytes: int
    numeric_attribute_values: int
    read_mode: Literal["header", "full"]
    bytes_read: int
    header_bytes: int
    header_sha256: str
    header_hash_scope: Literal["parsed_header_only_excluding_gaps_and_payload"] = (
        "parsed_header_only_excluding_gaps_and_payload"
    )
    declared_layout_checked: Literal[True] = True
    variable_values_decoded: Literal[False] = False
    variable_arrays_allocated: Literal[False] = False
    payload_contents_validated: Literal[False] = False
    source_snapshot_guaranteed: Literal[False] = False
    source_bytes: int
    source_sha256: str | None


@dataclass
class _Header:
    data: bytes
    position: int = 0
    attribute_count: int = 0
    attribute_bytes: int = 0
    numeric_values: int = 0

    def take(self, count: int) -> bytes:
        end = self.position + count
        if count < 0 or end > len(self.data):
            raise ValueError("NetCDF header is truncated or exceeds its 2 MB prefix budget")
        result = self.data[self.position : end]
        self.position = end
        return result

    def integer(self, width: int = 4) -> int:
        return int.from_bytes(self.take(width), "big")

    def padding(self, payload_bytes: int) -> None:
        if any(self.take((-payload_bytes) % 4)):
            raise ValueError("NetCDF header padding must contain only zero bytes")

    def name(self) -> str:
        length = self.integer()
        if not 1 <= length <= 128:
            raise ValueError("NetCDF names require 1 through 128 UTF-8 bytes")
        raw = self.take(length)
        self.padding(length)
        try:
            name = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("NetCDF names require valid UTF-8") from exc
        first = name[0]
        if (
            unicodedata.normalize("NFC", name) != name
            or name[-1].isspace()
            or any(
                ord(character) < 32 or ord(character) == 127 or character == "/"
                for character in name
            )
            or (first.isascii() and not (first.isalnum() or first == "_"))
        ):
            raise ValueError("NetCDF name violates NFC/classic syntax or has trailing whitespace")
        return name

    def list_count(self, expected_tag: int, maximum: int) -> int:
        tag, count = self.integer(), self.integer()
        if tag == 0 and count == 0:
            return 0
        if tag != expected_tag or count > maximum:
            raise ValueError("NetCDF list tag/count exceeds the supported header contract")
        return count

    def attributes(self) -> list[Attribute]:
        count = self.list_count(12, 128)
        self.attribute_count += count
        if self.attribute_count > 512:
            raise ValueError("NetCDF header exceeds 512 total attributes")
        attributes: list[Attribute] = []
        names: set[str] = set()
        for _ in range(count):
            name = self.name()
            if name in names:
                raise ValueError("NetCDF attribute names must be unique within their scope")
            names.add(name)
            kind, elements = self.integer(), self.integer()
            if kind not in _TYPES:
                raise ValueError("NetCDF attribute uses an unsupported classic type")
            type_name, width = _TYPES[kind]
            size = elements * width
            self.attribute_bytes += size
            if size > 32_768 or self.attribute_bytes > 128_000:
                raise ValueError("NetCDF attribute payload exceeds 32768 bytes or 128 KB aggregate")
            if kind != 2:
                self.numeric_values += elements
                if elements > 2048 or self.numeric_values > 8192:
                    raise ValueError(
                        "NetCDF numeric attributes exceed per-attribute/aggregate value bounds"
                    )
            raw = self.take(size)
            self.padding(size)
            values: list[str] | None
            encoding: Literal["raw_char_bytes", "signed_decimal_strings", "ieee_hex_float_strings"]
            if kind == 2:
                values = None
                encoding = "raw_char_bytes"
            elif kind in (1, 3, 4):
                values = [
                    str(int.from_bytes(raw[index : index + width], "big", signed=True))
                    for index in range(0, size, width)
                ]
                encoding = "signed_decimal_strings"
            else:
                values = [
                    value.hex() for (value,) in struct.iter_unpack(">f" if kind == 5 else ">d", raw)
                ]
                encoding = "ieee_hex_float_strings"
            attributes.append(
                Attribute(
                    name=name,
                    type=type_name,
                    element_count=elements,
                    raw_hex=raw.hex(),
                    numeric_values=values,
                    encoding=encoding,
                )
            )
        return attributes


@dataclass(frozen=True)
class _Variable:
    index: int
    name: str
    kind: int
    dimension_ids: list[int]
    attributes: list[Attribute]
    record: bool
    begin: int
    vsize: int
    elements: int
    data_bytes: int


def execute(request: Input, context: OperationContext) -> Output:
    source_hash: str | None = None
    if request.read_mode == "full":
        content = context.read_bytes(
            request.path, suffixes=(".nc", ".cdf"), max_bytes=min(request.max_file_bytes, 8_000_000)
        )
        source_bytes = bytes_read = len(content)
        source_hash = hashlib.sha256(content).hexdigest()
        prefix = content[:2_000_000]
    else:
        with context.open_ranges(
            request.path,
            suffixes=(".nc", ".cdf"),
            max_file_bytes=request.max_file_bytes,
            max_read_bytes=2_000_000,
        ) as ranges:
            source_bytes = ranges.source_bytes
            prefix = ranges.read_at(0, min(source_bytes, 2_000_000))
            ranges.check_unchanged_metadata()
            bytes_read = ranges.bytes_read
    header = _Header(prefix)
    magic = header.take(4)
    if magic not in (b"CDF\x01", b"CDF\x02"):
        raise ValueError(
            "only classic CDF1/CDF2 are supported; CDF5/HDF5/compression are unsupported"
        )
    version = magic[3]
    records = header.integer()
    if records == 0xFFFFFFFF:
        raise ValueError("NetCDF streaming/indeterminate record counts are unsupported")
    if records > 1_000_000_000:
        raise ValueError("NetCDF record count exceeds 1e9")
    dimensions: list[Dimension] = []
    dimension_names: set[str] = set()
    unlimited: int | None = None
    for index in range(header.list_count(10, 256)):
        name, length = header.name(), header.integer()
        if name in dimension_names or length > 1_000_000_000:
            raise ValueError("NetCDF dimensions require unique names and lengths at most 1e9")
        dimension_names.add(name)
        if length == 0:
            if unlimited is not None:
                raise ValueError("classic NetCDF supports only one unlimited dimension")
            unlimited = index
        dimensions.append(
            Dimension(
                index=index,
                name=name,
                declared_length=length,
                unlimited=length == 0,
                resolved_length=records if length == 0 else length,
            )
        )
    global_attributes = header.attributes()
    variables: list[_Variable] = []
    variable_names: set[str] = set()
    for index in range(header.list_count(11, 256)):
        name, rank = header.name(), header.integer()
        if name in variable_names or rank > 16:
            raise ValueError("NetCDF variables require unique names and rank at most 16")
        variable_names.add(name)
        ids = [header.integer() for _ in range(rank)]
        if any(index >= len(dimensions) for index in ids):
            raise ValueError("NetCDF variable references an unknown dimension")
        if unlimited is not None and unlimited in ids[1:]:
            raise ValueError("NetCDF unlimited dimension may occur only first in a variable")
        is_record = bool(ids) and ids[0] == unlimited
        attributes = header.attributes()
        kind, vsize = header.integer(), header.integer()
        begin = header.integer(8 if version == 2 else 4)
        if kind not in _TYPES or vsize > _MAX_SLAB:
            raise ValueError("unsupported NetCDF variable type or oversized vsize/sentinel")
        if begin % 4 or begin > min(
            request.max_file_bytes, (1 << (63 if version == 2 else 31)) - 1
        ):
            raise ValueError(
                "NetCDF variable begin must be aligned and fit the supported offset range"
            )
        elements = 1
        for dimension_id in ids[1:] if is_record else ids:
            elements *= dimensions[dimension_id].declared_length
            if elements * _TYPES[kind][1] > _MAX_SLAB:
                raise ValueError("NetCDF fixed variable or record slab exceeds 2^31-4 bytes")
        data_bytes = elements * _TYPES[kind][1]
        if (data_bytes + 3) // 4 * 4 != vsize:
            raise ValueError("NetCDF vsize disagrees with its padded type/shape size")
        variables.append(
            _Variable(
                index, name, kind, ids, attributes, is_record, begin, vsize, elements, data_bytes
            )
        )
    header_size = header.position
    fixed = [variable for variable in variables if not variable.record]
    record_variables = [variable for variable in variables if variable.record]
    if records and not record_variables:
        raise ValueError("positive NetCDF record count requires at least one record variable")
    if any(variable.begin < header_size for variable in variables):
        raise ValueError("NetCDF variable begins within the parsed header")
    fixed_end = header_size
    gaps = 0
    for variable in fixed:
        if variable.begin < fixed_end:
            raise ValueError("NetCDF fixed variables overlap or violate header order")
        gaps += variable.begin - fixed_end
        fixed_end = variable.begin + variable.vsize
        if fixed_end > source_bytes:
            raise ValueError("NetCDF fixed-variable extent exceeds the source")
    record_start: int | None = None
    stride: int | None = None
    unpadded = False
    if record_variables:
        record_start = record_variables[0].begin
        if record_start < fixed_end:
            raise ValueError("NetCDF record section overlaps fixed data")
        expected_begin = record_start
        for variable in record_variables:
            if variable.begin != expected_begin:
                raise ValueError("NetCDF record slabs must interleave contiguously in header order")
            expected_begin += variable.vsize
        unpadded = len(record_variables) == 1 and record_variables[0].kind in (1, 2, 3)
        stride = (
            record_variables[0].data_bytes
            if unpadded
            else sum(variable.vsize for variable in record_variables)
        )
        if records:
            expected_end = record_start + records * stride
            if expected_end != source_bytes:
                raise ValueError(
                    "NetCDF record extent is truncated or has unsupported trailing data"
                )
            gaps += record_start - fixed_end
        elif source_bytes not in (fixed_end, record_start):
            raise ValueError(
                "zero-record NetCDF must end at fixed data or the planned record start"
            )
        else:
            gaps += source_bytes - fixed_end
    elif source_bytes != fixed_end:
        raise ValueError("NetCDF source has unsupported trailing bytes beyond fixed data/header")
    page: list[Variable] = []
    for variable in variables[
        request.variable_offset : request.variable_offset + request.variable_limit
    ]:
        last_start: int | None = variable.begin
        if variable.record:
            last_start = variable.begin + (records - 1) * (stride or 0) if records else None
        page.append(
            Variable(
                index=variable.index,
                name=variable.name,
                type=_TYPES[variable.kind][0],
                dimension_ids=variable.dimension_ids,
                shape=[dimensions[index].resolved_length for index in variable.dimension_ids],
                attributes=variable.attributes,
                record_variable=variable.record,
                begin=variable.begin,
                declared_vsize=variable.vsize,
                elements_per_slab=variable.elements,
                data_bytes_per_slab=variable.data_bytes,
                total_elements=variable.elements * records
                if variable.record
                else variable.elements,
                record_stride=stride if variable.record else None,
                last_data_end_exclusive=None
                if last_start is None
                else last_start + variable.data_bytes,
                allocated_end_exclusive=None
                if last_start is None
                else last_start
                + (variable.data_bytes if variable.record and unpadded else variable.vsize),
            )
        )
    return Output(
        format="CDF1" if version == 1 else "CDF2",
        declared_record_count=records,
        dimensions=dimensions,
        global_attributes=global_attributes,
        variable_count=len(variables),
        fixed_variable_count=len(fixed),
        record_variable_count=len(record_variables),
        record_start=record_start,
        record_stride=stride,
        unpadded_single_record_variable=unpadded,
        uninterpreted_gap_bytes=gaps,
        variables=page,
        variable_offset=request.variable_offset,
        has_more_variables=request.variable_offset + len(page) < len(variables),
        attribute_count=header.attribute_count,
        raw_attribute_bytes=header.attribute_bytes,
        numeric_attribute_values=header.numeric_values,
        read_mode=request.read_mode,
        bytes_read=bytes_read,
        header_bytes=header_size,
        header_sha256=hashlib.sha256(prefix[:header_size]).hexdigest(),
        source_bytes=source_bytes,
        source_sha256=source_hash,
    )


OPERATION = Operation(
    id="plugins.inspect_netcdf_classic",
    kind="plugin",
    description="Parse bounded CDF1/CDF2 metadata, preserve raw attribute encodings and verify fixed/record extent arithmetic including the lone-record padding exception, without allocating variable arrays or interpreting payload values.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
