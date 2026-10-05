"""Validate SciPy-style two-dimensional CSR/CSC/COO NPZ without object loading.

Own ZIP directory/local-header parser permits STORED or raw DEFLATE, regular
basename members and NumPy's local ZIP64 size extra. It rejects directory ZIP64,
descriptors, encryption, comments, other extras, multidisk, links, overlap, gaps,
prefix/trailing bytes, undeclared members and CRC/size disagreement. Each bounded
member is fully expanded and hashed before its NPY header/payload is inspected.

NPY versions 1.0/2.0, ASCII literal headers, scalar/one-dimensional arrays and
fortran_order=False only. Header ASTs are inspected, never evaluated. Required
format.npy is scalar |S3, shape.npy is two signed integers; CSR/CSC require indices
and indptr, COO requires row and col. Optional _is_array is a scalar |b1 marker.
Index/pointer/shape dtypes are explicit little/big endian i4/i8. Data supports
|b1, |i1, |u1, endian i/u2/4/8, f4/f8 and c8/c16. Object, structured, native-endian,
Unicode, half/extended floats and nonfinite values fail. No pickle, NumPy/SciPy
constructor, filesystem extraction or dense matrix allocation occurs.

Preserves storage order, duplicate coordinates and explicit zeros without sums.
Integer components are exact decimal strings; float/complex components use exact
float.hex strings, plus original per-entry bytes. Duplicate count counts entries
beyond the first occurrence. Sorted means nondecreasing storage coordinates:
row/column for CSR/COO, column/row for CSC. Canonical means sorted and unique.

Limits: 8 MB source; 5..6 members, 64 KB directory, 16 MB/member and 32 MB total
expanded bytes; 8192-byte NPY headers, 64 AST nodes, 250000 entries, dimensions
at most 1e9 and compressed major axis at most 100000; 1000 returned coordinates.
These bounds do not imply a total-process memory ceiling. SHA-256 source hash is
of original ZIP bytes; member hashes cover expanded original NPY bytes.
References: PKWARE APPNOTE 4.3/4.5.3; NumPy numpy.lib.format specification;
SciPy scipy/sparse/_matrix_io.py save_npz layout (row/col COO subset).
"""

from __future__ import annotations

import ast
import hashlib
import math
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    expected_format: Literal["csr", "csc", "coo"]
    offset: int = Field(default=0, strict=True, ge=0, le=250_000)
    limit: int = Field(default=100, strict=True, ge=1, le=1000)


class Member(OutputModel):
    name: str
    compressed_bytes: int
    expanded_bytes: int
    compression: Literal["stored", "deflate"]
    sha256: str
    crc32_hex: str
    npy_version: str
    dtype: str
    shape: list[int]


class Entry(OutputModel):
    storage_index: int
    row: int
    column: int
    components: list[str]
    raw_hex: str


class Output(OutputModel):
    sparse_format: Literal["csr", "csc", "coo"]
    shape: list[int]
    data_dtype: str
    value_encoding: Literal["decimal_integer", "boolean_digit", "ieee754_hex_components"]
    is_array_marker: bool | None
    stored_entry_count: int
    explicit_zero_count: int
    duplicate_entry_count: int
    coordinates_unique: bool
    storage_coordinates_sorted: bool
    canonical_storage: bool
    has_stored_entries: bool
    entries: list[Entry]
    offset: int
    has_more: bool
    members: list[Member]
    expanded_bytes: int
    full_source_validated: Literal[True] = True
    dense_matrix_allocated: Literal[False] = False
    duplicate_reduction_performed: Literal[False] = False
    source_bytes: int
    source_sha256: str


@dataclass(frozen=True)
class _ZipEntry:
    name: str
    flags: int
    method: int
    clock: int
    date: int
    crc: int
    compressed: int
    expanded: int
    offset: int


@dataclass(frozen=True)
class _Array:
    dtype: str
    shape: tuple[int, ...]
    width: int
    payload: memoryview
    version: str


def _directory(content: bytes, expected_format: str) -> tuple[list[_ZipEntry], int]:
    if len(content) < 22 or content[-22:-18] != b"PK\x05\x06":
        raise ValueError("NPZ requires an exact terminal ZIP EOCD without a comment")
    disk, central_disk, disk_count, count, size, offset, comment = struct.unpack_from(
        "<4H2IH", content, len(content) - 18
    )
    if disk or central_disk or comment or count not in (5, 6) or disk_count != count:
        raise ValueError("NPZ requires 5..6 entries, one disk and no ZIP comment")
    if size > 65_536 or offset + size != len(content) - 22:
        raise ValueError("NPZ central-directory extent is invalid or exceeds 64 KB")
    expected = {"format.npy", "shape.npy", "data.npy"}
    expected.update(
        {"row.npy", "col.npy"} if expected_format == "coo" else {"indices.npy", "indptr.npy"}
    )
    entries: list[_ZipEntry] = []
    names: set[str] = set()
    position = offset
    total = 0
    for _ in range(count):
        if position + 46 > len(content) - 22:
            raise ValueError("NPZ truncated central-directory header")
        (
            signature,
            _made,
            needed,
            flags,
            method,
            clock,
            date,
            crc,
            compressed,
            expanded,
            name_length,
            extra_length,
            comment_length,
            start_disk,
            _internal,
            external,
            local_offset,
        ) = struct.unpack_from("<4s6H3I5H2I", content, position)
        if signature != b"PK\x01\x02" or not 10 <= needed <= 45:
            raise ValueError("NPZ unsupported ZIP central header/version")
        if method not in (0, 8) or flags & ~(0x800 | (0x6 if method == 8 else 0)):
            raise ValueError("NPZ supports stored/deflate without encryption or descriptors only")
        if extra_length or comment_length or start_disk or not 1 <= name_length <= 32:
            raise ValueError("NPZ central extras/comments/multidisk/long names are unsupported")
        if external & 0x10 or ((external >> 16) & 0o170000) not in (0, 0o100000):
            raise ValueError("NPZ members must be regular files, not links/directories")
        end = position + 46 + name_length
        if end > offset + size:
            raise ValueError("NPZ central member name is truncated")
        name = content[position + 46 : end].decode("ascii")
        if name not in expected | {"_is_array.npy"} or name in names:
            raise ValueError("NPZ has duplicate, unsafe or unsupported member names")
        names.add(name)
        if expanded > 16_000_000 or compressed > len(content) or local_offset >= offset:
            raise ValueError("NPZ member size/offset exceeds its bounds")
        total += expanded
        if total > 32_000_000:
            raise ValueError("NPZ expanded members exceed 32 MB")
        entries.append(
            _ZipEntry(name, flags, method, clock, date, crc, compressed, expanded, local_offset)
        )
        position = end
    if position != offset + size or not expected <= names or len(names - expected) != count - 5:
        raise ValueError("NPZ central directory does not exactly match the expected sparse layout")
    return entries, offset


def _expand(content: bytes, entries: list[_ZipEntry], central_offset: int) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    end = 0
    for entry in sorted(entries, key=lambda item: item.offset):
        if entry.offset != end or entry.offset + 30 > central_offset:
            raise ValueError("NPZ local records overlap or contain gaps/prefix bytes")
        (
            signature,
            needed,
            flags,
            method,
            clock,
            date,
            crc,
            compressed,
            expanded,
            name_length,
            extra_length,
        ) = struct.unpack_from("<4s5H3I2H", content, entry.offset)
        if signature != b"PK\x03\x04" or not 10 <= needed <= 45:
            raise ValueError("NPZ unsupported ZIP local header/version")
        if (flags, method, clock, date, crc) != (
            entry.flags,
            entry.method,
            entry.clock,
            entry.date,
            entry.crc,
        ):
            raise ValueError("NPZ local and central metadata disagree")
        start = entry.offset + 30
        payload_start = start + name_length + extra_length
        if payload_start + entry.compressed > central_offset or content[
            start : start + name_length
        ] != entry.name.encode("ascii"):
            raise ValueError("NPZ local name/data extent disagrees with central directory")
        extra = content[start + name_length : payload_start]
        if expanded == 0xFFFFFFFF or compressed == 0xFFFFFFFF:
            if (
                expanded != 0xFFFFFFFF
                or compressed != 0xFFFFFFFF
                or needed < 45
                or len(extra) != 20
                or extra[:4] != b"\x01\x00\x10\x00"
            ):
                raise ValueError("NPZ only supports the standard pair of local ZIP64 size values")
            expanded, compressed = struct.unpack_from("<QQ", extra, 4)
        elif extra:
            raise ValueError("NPZ unsupported local ZIP extra field")
        if (expanded, compressed) != (entry.expanded, entry.compressed):
            raise ValueError("NPZ local and central member sizes disagree")
        end = payload_start + compressed
        packed = content[payload_start:end]
        if entry.method == 0:
            if compressed != expanded:
                raise ValueError("NPZ stored member compressed/expanded sizes differ")
            unpacked = packed
        else:
            decoder = zlib.decompressobj(-15)
            try:
                unpacked = decoder.decompress(packed, expanded + 1)
            except zlib.error as exc:
                raise ValueError("NPZ invalid raw DEFLATE member") from exc
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError(
                    "NPZ deflate stream is truncated, overlong or exceeds declared expansion"
                )
        if len(unpacked) != expanded or zlib.crc32(unpacked) != entry.crc:
            raise ValueError("NPZ member expanded size or CRC32 differs from its declaration")
        result[entry.name] = unpacked
    if end != central_offset:
        raise ValueError("NPZ contains a gap/trailing record before its central directory")
    return result


def _dtype_width(dtype: str) -> int:
    if dtype in {"|b1", "|i1", "|u1"}:
        return 1
    if dtype == "|S3":
        return 3
    if len(dtype) >= 3 and dtype[0] in "<>":
        supported = {"i2", "i4", "i8", "u2", "u4", "u8", "f4", "f8", "c8", "c16"}
        if dtype[1:] in supported:
            return int(dtype[2:])
    raise ValueError("NPZ NPY dtype is outside the explicit portable numeric subset")


def _array(content: bytes) -> _Array:
    if (
        len(content) < 10
        or content[:6] != b"\x93NUMPY"
        or content[6:8] not in (b"\x01\x00", b"\x02\x00")
    ):
        raise ValueError("NPZ requires NPY format 1.0 or 2.0 members")
    version = content[6]
    prefix = 10 if version == 1 else 12
    if len(content) < prefix:
        raise ValueError("NPZ truncated NPY length")
    header_length = int.from_bytes(content[8:prefix], "little")
    end = prefix + header_length
    if not 1 <= header_length <= 8192 or end > len(content) or end % 16:
        raise ValueError("NPZ NPY header exceeds 8192 bytes, is truncated or lacks alignment")
    raw = content[prefix:end]
    if not raw.endswith(b"\n"):
        raise ValueError("NPZ NPY header requires a terminal newline")
    text = raw.decode("ascii")
    if any(ord(char) < 32 for char in text[:-1]):
        raise ValueError("NPZ NPY header must be one ASCII literal line")
    expression = text.strip(" \n")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("NPZ malformed NPY header literal") from exc
    if (
        sum(1 for _ in ast.walk(tree)) > 64
        or not isinstance(tree.body, ast.Dict)
        or len(tree.body.keys) != 3
        or tree.body.col_offset != 0
        or tree.body.end_col_offset != len(expression)
    ):
        raise ValueError("NPZ NPY header must contain exactly three literal dictionary keys")
    fields: dict[str, ast.expr] = {}
    for key, value in zip(tree.body.keys, tree.body.values, strict=True):
        if not isinstance(key, ast.Constant) or type(key.value) is not str or key.value in fields:
            raise ValueError("NPZ NPY header keys must be unique strings")
        fields[key.value] = value
    if set(fields) != {"descr", "fortran_order", "shape"}:
        raise ValueError("NPZ NPY header has unsupported dictionary keys")
    descriptor, order, dimensions = fields["descr"], fields["fortran_order"], fields["shape"]
    if not isinstance(descriptor, ast.Constant) or type(descriptor.value) is not str:
        raise ValueError("NPZ structured/object dtype declarations are unsupported")
    if not isinstance(order, ast.Constant) or order.value is not False:
        raise ValueError("NPZ requires fortran_order=False")
    if not isinstance(dimensions, ast.Tuple) or len(dimensions.elts) > 1:
        raise ValueError("NPZ members must be scalar or one-dimensional arrays")
    shape: list[int] = []
    for element in dimensions.elts:
        if (
            not isinstance(element, ast.Constant)
            or type(element.value) is not int
            or not 0 <= element.value <= 250_001
        ):
            raise ValueError("NPZ member dimension must be a literal integer within its work bound")
        shape.append(element.value)
    dtype = descriptor.value
    width = _dtype_width(dtype)
    count = shape[0] if shape else 1
    if len(content) - end != count * width:
        raise ValueError("NPZ NPY payload size differs from its dtype and shape")
    return _Array(dtype, tuple(shape), width, memoryview(content)[end:], f"{version}.0")


def _integers(array: _Array) -> list[int]:
    if array.dtype not in {"<i4", ">i4", "<i8", ">i8"} or len(array.shape) != 1:
        raise ValueError(
            "NPZ shape/index/pointer arrays require one-dimensional explicit-endian i4/i8"
        )
    return [
        int.from_bytes(
            array.payload[start : start + array.width],
            "little" if array.dtype[0] == "<" else "big",
            signed=True,
        )
        for start in range(0, len(array.payload), array.width)
    ]


def _coordinates(
    arrays: dict[str, _Array], sparse_format: str, shape: list[int], count: int
) -> Iterator[tuple[int, int, int]]:
    if sparse_format == "coo":
        rows, columns = _integers(arrays["row.npy"]), _integers(arrays["col.npy"])
        if len(rows) != count or len(columns) != count:
            raise ValueError("NPZ COO row/column lengths differ from data length")
        for index, (row, column) in enumerate(zip(rows, columns, strict=True)):
            if not 0 <= row < shape[0] or not 0 <= column < shape[1]:
                raise ValueError("NPZ COO coordinate lies outside the declared matrix shape")
            yield index, row, column
        return
    major, minor = shape if sparse_format == "csr" else shape[::-1]
    if major > 100_000:
        raise ValueError("NPZ CSR/CSC major dimension exceeds 100000")
    indices, pointers = _integers(arrays["indices.npy"]), _integers(arrays["indptr.npy"])
    if len(indices) != count or len(pointers) != major + 1:
        raise ValueError("NPZ CSR/CSC index/pointer lengths disagree with shape/data")
    if (
        pointers[0] != 0
        or pointers[-1] != count
        or any(left > right for left, right in zip(pointers, pointers[1:], strict=False))
    ):
        raise ValueError(
            "NPZ pointers must start at zero, increase non-strictly and end at data length"
        )
    for major_index in range(major):
        for index in range(pointers[major_index], pointers[major_index + 1]):
            minor_index = indices[index]
            if not 0 <= minor_index < minor:
                raise ValueError("NPZ compressed sparse index lies outside the minor axis")
            row, column = (
                (major_index, minor_index) if sparse_format == "csr" else (minor_index, major_index)
            )
            yield index, row, column


def _components(array: _Array, index: int) -> tuple[list[str], str, bool]:
    raw = array.payload[index * array.width : (index + 1) * array.width]
    dtype = array.dtype
    if dtype == "|b1":
        if raw[0] not in (0, 1):
            raise ValueError("NPZ boolean payload bytes must be canonical zero or one")
        return [str(raw[0])], raw.hex(), raw[0] == 0
    if dtype[1] in "iu":
        integer = int.from_bytes(
            raw, "big" if dtype[0] == ">" else "little", signed=dtype[1] == "i"
        )
        return [str(integer)], raw.hex(), integer == 0
    if dtype[1] not in "fc":
        raise ValueError("NPZ data array must use a supported numeric dtype")
    code = (
        ("f" if array.width == 4 else "d")
        if dtype[1] == "f"
        else ("ff" if array.width == 8 else "dd")
    )
    values = struct.unpack(dtype[0] + code, raw)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NPZ nonfinite real/complex payload values are unsupported")
    return [value.hex() for value in values], raw.hex(), all(value == 0 for value in values)


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".npz",), max_bytes=8_000_000)
    directory, central_offset = _directory(content, request.expected_format)
    expanded = _expand(content, directory, central_offset)
    arrays = {name: _array(member) for name, member in expanded.items()}
    format_array = arrays["format.npy"]
    if (
        format_array.dtype != "|S3"
        or format_array.shape
        or format_array.payload != request.expected_format.encode("ascii")
    ):
        raise ValueError("NPZ scalar format.npy must match expected_format exactly")
    shape = _integers(arrays["shape.npy"])
    if len(shape) != 2 or any(dimension < 0 or dimension > 1_000_000_000 for dimension in shape):
        raise ValueError("NPZ matrix shape must contain two dimensions in 0..1e9")
    marker: bool | None = None
    if "_is_array.npy" in arrays:
        observed = arrays["_is_array.npy"]
        if observed.dtype != "|b1" or observed.shape or observed.payload[0] not in (0, 1):
            raise ValueError("NPZ optional _is_array marker must be a canonical scalar boolean")
        marker = bool(observed.payload[0])
    data = arrays["data.npy"]
    if len(data.shape) != 1 or data.shape[0] > 250_000 or data.dtype == "|S3":
        raise ValueError("NPZ data must be a supported numeric vector with at most 250000 entries")
    count = data.shape[0]
    seen: set[tuple[int, int]] = set()
    previous: tuple[int, int] | None = None
    sorted_storage = True
    duplicates = zeros = 0
    entries: list[Entry] = []
    for index, row, column in _coordinates(arrays, request.expected_format, shape, count):
        coordinate = (row, column)
        duplicates += int(coordinate in seen)
        seen.add(coordinate)
        ordered = coordinate[::-1] if request.expected_format == "csc" else coordinate
        if previous is not None and ordered < previous:
            sorted_storage = False
        previous = ordered
        components, raw_hex, zero = _components(data, index)
        zeros += int(zero)
        if request.offset <= index < request.offset + request.limit:
            entries.append(
                Entry(
                    storage_index=index,
                    row=row,
                    column=column,
                    components=components,
                    raw_hex=raw_hex,
                )
            )
    members = [
        Member(
            name=entry.name,
            compressed_bytes=entry.compressed,
            expanded_bytes=entry.expanded,
            compression="stored" if entry.method == 0 else "deflate",
            sha256=hashlib.sha256(expanded[entry.name]).hexdigest(),
            crc32_hex=f"{entry.crc:08x}",
            npy_version=arrays[entry.name].version,
            dtype=arrays[entry.name].dtype,
            shape=list(arrays[entry.name].shape),
        )
        for entry in directory
    ]
    return Output(
        sparse_format=request.expected_format,
        shape=shape,
        data_dtype=data.dtype,
        value_encoding="boolean_digit"
        if data.dtype == "|b1"
        else "decimal_integer"
        if data.dtype[1] in "iu"
        else "ieee754_hex_components",
        is_array_marker=marker,
        stored_entry_count=count,
        explicit_zero_count=zeros,
        duplicate_entry_count=duplicates,
        coordinates_unique=duplicates == 0,
        storage_coordinates_sorted=sorted_storage,
        canonical_storage=sorted_storage and duplicates == 0,
        has_stored_entries=bool(count),
        entries=entries,
        offset=request.offset,
        has_more=request.offset + len(entries) < count,
        members=members,
        expanded_bytes=sum(entry.expanded for entry in directory),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_sparse_npz",
    kind="plugin",
    description="Validate bounded SciPy CSR/CSC/COO ZIP/NPY structure and sparse indices; return exact stored-coordinate pages without pickle or dense allocation.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
