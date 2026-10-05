"""Inspect classic TIFF IFD metadata without decoding any image payload.

Accept II/MM magic 42, types 1..12 plus the type 13 IFD extension, ordered unique
tags, aligned directory/out-of-line value offsets and bounded value extents.
Follow all next-IFD links, SubIFDs tag 330 (LONG/IFD) and any explicitly typed IFD
values. Detect directed cycles; shared child directories are allowed and parsed
once. Reject BigTIFF, unknown field types and EXIF/GPS/Interoperability LONG
pointer conventions, whose auxiliary semantics are outside this declared subset.

Primitive values are observations: exact integer strings, raw rational n/d
(including zero denominators with a flag), IEEE float.hex or nan/inf strings
plus original bits. ASCII values require 7-bit bytes and a final NUL; embedded
NULs delimit strings. UNDEFINED values remain opaque. Private tag meanings are
uninterpreted, but their declared primitive encodings are decoded and explicit
IFD types are traversed. Private tags never execute extensions. No file
references are followed; unknown
LONG/private pointer conventions are not inferred. All original bytes are hashed.

StripOffsets/StripByteCounts and TileOffsets/TileByteCounts, when present, must
come in equal-length pairs with BYTE/SHORT/LONG types and contained byte extents.
This checks declared bounds only: no required image-tag schema, decoded shape,
compression, pixel contents, payload overlap or complete file coverage is proved.
Trailing/unreferenced bytes are permitted. Metadata blocks cannot overlap IFDs;
out-of-line value blocks may share an identical extent but partial overlaps fail.

Limits: 8 MB source, 128 IFDs, 256 tags/IFD and 2048 total, 1024 graph edges,
65536 bytes/value and 2 MB summed value bytes, 131072 typed values, 4096 payload
references; each directory returns at most 16 payload spans. Tag pages contain
at most 100 tags/512 KB JSON, with 128 primitive preview values and 1024 raw bytes
per tag. Every reachable supported metadata structure is validated before paging;
metadata integrity does not certify image validity or source authenticity.
References: Adobe TIFF 6.0 sections 2/8, https://www.itu.int/itudoc/itu-t/com16/tiff-fx/docs/tiff6.pdf
Adobe PageMaker 6.0 TIFF Technical Note 1 (1995), SubIFDs/type 13.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_TYPES = {
    1: ("BYTE", 1),
    2: ("ASCII", 1),
    3: ("SHORT", 2),
    4: ("LONG", 4),
    5: ("RATIONAL", 8),
    6: ("SBYTE", 1),
    7: ("UNDEFINED", 1),
    8: ("SSHORT", 2),
    9: ("SLONG", 4),
    10: ("SRATIONAL", 8),
    11: ("FLOAT", 4),
    12: ("DOUBLE", 8),
    13: ("IFD", 4),
}


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=2048)
    limit: int = Field(default=50, strict=True, ge=1, le=100)


class Tag(OutputModel):
    tag_index: int
    directory_byte_start: int
    entry_byte_start: int
    tag_id: int
    private_tag_number: bool
    type_name: str
    value_count: int
    value_byte_start: int
    value_bytes: int
    inline: bool
    value_sha256: str
    raw_preview_hex: str
    raw_preview_truncated: bool
    preview_values: list[str] | None
    preview_values_truncated: bool
    nonfinite_float_count: int
    zero_denominator_count: int
    meaning: Literal[
        "directory_pointer", "payload_offset_or_count", "opaque_undefined", "primitive_observation"
    ]


class Edge(OutputModel):
    parent_byte_start: int
    child_byte_start: int
    kind: Literal["next_ifd", "sub_ifd", "typed_ifd"]
    tag_id: int | None = None


class PayloadSpan(OutputModel):
    kind: Literal["strip", "tile"]
    index: int
    source_byte_start: int
    byte_length: int


class Directory(OutputModel):
    source_byte_start: int
    byte_length: int
    entry_count: int
    next_ifd_byte_start: int | None
    payload_reference_count: int
    summed_declared_payload_bytes: int
    payload_span_preview: list[PayloadSpan]
    payload_span_preview_truncated: bool


class Output(OutputModel):
    byte_order: Literal["little", "big"]
    first_ifd_byte_start: int
    directories: list[Directory]
    edges: list[Edge]
    shared_directory_reference_count: int
    tag_count: int
    typed_value_count: int
    summed_value_bytes: int
    payload_reference_count: int
    tags: list[Tag]
    offset: int
    has_more: bool
    supported_metadata_graph_validated: Literal[True] = True
    graph_has_cycle: Literal[False] = False
    image_schema_validated: Literal[False] = False
    image_payload_decoded: Literal[False] = False
    image_payload_overlap_checked: Literal[False] = False
    full_file_coverage_verified: Literal[False] = False
    private_pointer_semantics_inferred: Literal[False] = False
    external_references_followed: Literal[False] = False
    source_bytes: int
    source_sha256: str


@dataclass
class _Entry:
    tag: int
    kind: int
    count: int
    start: int
    size: int
    entry: int
    inline: bool


class _Parser:
    def __init__(self, source: bytes, order: Literal["little", "big"]) -> None:
        self.source, self.order = source, order
        self.entries: list[tuple[int, _Entry]] = []
        self.directories: list[Directory] = []
        self.edges: list[Edge] = []
        self.adjacency: dict[int, list[int]] = {}
        self.directory_spans: list[tuple[int, int]] = [(0, 8)]
        self.value_spans: set[tuple[int, int]] = set()
        self.value_bytes = self.value_count = self.payload_count = 0

    def integer(self, start: int, width: int) -> int:
        if start < 0 or start + width > len(self.source):
            raise ValueError("TIFF integer/structure is truncated")
        return int.from_bytes(self.source[start : start + width], self.order)

    def numbers(self, entry: _Entry) -> list[int]:
        if entry.kind not in {1, 3, 4, 13}:
            raise ValueError("TIFF pointer/count values require an unsigned integer type")
        width = _TYPES[entry.kind][1]
        return [self.integer(entry.start + index * width, width) for index in range(entry.count)]

    def edge(
        self,
        parent: int,
        child: int,
        kind: Literal["next_ifd", "sub_ifd", "typed_ifd"],
        tag: int | None,
    ) -> None:
        if len(self.edges) >= 1024 or child < 8 or child & 1 or child + 2 > len(self.source):
            raise ValueError("TIFF graph edge is null/unaligned/out of bounds or exceeds 1024")
        self.edges.append(
            Edge(parent_byte_start=parent, child_byte_start=child, kind=kind, tag_id=tag)
        )
        self.adjacency[parent].append(child)

    def directory(self, start: int) -> None:
        if len(self.directories) >= 128 or start < 8 or start & 1:
            raise ValueError("TIFF directory count/address violates its bound/alignment")
        count = self.integer(start, 2)
        end = start + 2 + 12 * count + 4
        if not 1 <= count <= 256 or len(self.entries) + count > 2048 or end > len(self.source):
            raise ValueError("TIFF directory tag count/extent exceeds its bound")
        self.directory_spans.append((start, end))
        self.adjacency[start] = []
        entries: dict[int, _Entry] = {}
        last_tag = -1
        for index in range(count):
            position = start + 2 + 12 * index
            tag, kind, number = (
                self.integer(position, 2),
                self.integer(position + 2, 2),
                self.integer(position + 4, 4),
            )
            if tag <= last_tag or kind not in _TYPES or number == 0:
                raise ValueError(
                    "TIFF tags must be ordered/unique with supported types and positive counts"
                )
            last_tag = tag
            if tag in {34665, 34853, 40965}:
                raise ValueError(
                    "TIFF EXIF/GPS/Interoperability pointer conventions are unsupported"
                )
            size = number * _TYPES[kind][1]
            self.value_count += number
            self.value_bytes += size
            if size > 65_536 or self.value_bytes > 2_000_000 or self.value_count > 131_072:
                raise ValueError("TIFF tag value bytes/count exceed metadata work limits")
            inline = size <= 4
            location = position + 8 if inline else self.integer(position + 8, 4)
            if location < 8 or location + size > len(self.source) or not inline and location & 1:
                raise ValueError("TIFF value extent is outside the source or unaligned")
            if not inline:
                self.value_spans.add((location, location + size))
            entry = _Entry(tag, kind, number, location, size, position, inline)
            entries[tag] = entry
            self.entries.append((start, entry))
            if tag == 330 or kind == 13:
                if kind not in {4, 13}:
                    raise ValueError("TIFF SubIFDs requires LONG or IFD values")
                for child in self.numbers(entry):
                    self.edge(start, child, "sub_ifd" if tag == 330 else "typed_ifd", tag)
        next_ifd = self.integer(end - 4, 4)
        if next_ifd:
            self.edge(start, next_ifd, "next_ifd", None)
        payloads: list[PayloadSpan] = []
        directory_payload_count = payload_bytes = 0
        for offset_tag, size_tag, category in ((273, 279, "strip"), (324, 325, "tile")):
            if (offset_tag in entries) != (size_tag in entries):
                raise ValueError("TIFF strip/tile offsets and byte counts must be paired")
            if offset_tag not in entries:
                continue
            offsets, sizes = entries[offset_tag], entries[size_tag]
            if (
                offsets.kind not in {1, 3, 4}
                or sizes.kind not in {1, 3, 4}
                or offsets.count != sizes.count
            ):
                raise ValueError("TIFF strip/tile arrays require equal unsigned integer counts")
            self.payload_count += offsets.count
            directory_payload_count += offsets.count
            if self.payload_count > 4096:
                raise ValueError("TIFF exceeds 4096 payload references")
            for index, (position, size) in enumerate(
                zip(self.numbers(offsets), self.numbers(sizes), strict=True)
            ):
                if (
                    position > len(self.source)
                    or position + size > len(self.source)
                    or size
                    and position < 8
                ):
                    raise ValueError("TIFF declared strip/tile extent is outside the source")
                payload_bytes += size
                if len(payloads) < 16:
                    payloads.append(
                        PayloadSpan(
                            kind="strip" if category == "strip" else "tile",
                            index=index,
                            source_byte_start=position,
                            byte_length=size,
                        )
                    )
        self.directories.append(
            Directory(
                source_byte_start=start,
                byte_length=end - start,
                entry_count=count,
                next_ifd_byte_start=next_ifd or None,
                payload_reference_count=directory_payload_count,
                summed_declared_payload_bytes=payload_bytes,
                payload_span_preview=payloads,
                payload_span_preview_truncated=directory_payload_count > len(payloads),
            )
        )

    def validate_spans(self) -> None:
        structures = sorted(self.directory_spans)
        if any(right[0] < left[1] for left, right in zip(structures, structures[1:], strict=False)):
            raise ValueError("TIFF directory structures overlap")
        values = sorted(self.value_spans)
        if any(right[0] < left[1] for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("TIFF out-of-line value extents partially overlap")
        structure_index = 0
        for start, end in values:
            while structure_index < len(structures) and structures[structure_index][1] <= start:
                structure_index += 1
            if structure_index < len(structures) and structures[structure_index][0] < end:
                raise ValueError("TIFF out-of-line values overlap header/directory structures")

    def preview(self, directory: int, entry: _Entry, index: int) -> Tag:
        raw = self.source[entry.start : entry.start + entry.size]
        values: list[str] = []
        nonfinite = zero_denominator = 0
        if entry.kind == 2:
            if raw[-1] != 0 or any(byte > 127 for byte in raw):
                raise ValueError("TIFF ASCII field must contain 7-bit bytes and end in NUL")
            strings = raw[:-1].split(b"\0")
            values = [value.decode("ascii") for value in strings[:128]]
            truncated = len(strings) > 128
        elif entry.kind == 7:
            truncated = False
        else:
            width = _TYPES[entry.kind][1]
            for item in range(entry.count):
                scalar = raw[item * width : (item + 1) * width]
                if entry.kind in {5, 10}:
                    numerator = int.from_bytes(scalar[:4], self.order, signed=entry.kind == 10)
                    denominator = int.from_bytes(scalar[4:], self.order, signed=entry.kind == 10)
                    zero_denominator += denominator == 0
                    value = f"{numerator}/{denominator}"
                elif entry.kind in {11, 12}:
                    floating: float = struct.unpack(
                        ("<" if self.order == "little" else ">")
                        + ("f" if entry.kind == 11 else "d"),
                        scalar,
                    )[0]
                    nonfinite += not math.isfinite(floating)
                    value = floating.hex()
                else:
                    value = str(int.from_bytes(scalar, self.order, signed=entry.kind in {6, 8, 9}))
                if item < 128:
                    values.append(value)
            truncated = entry.count > 128
        return Tag(
            tag_index=index,
            directory_byte_start=directory,
            entry_byte_start=entry.entry,
            tag_id=entry.tag,
            private_tag_number=entry.tag >= 32768,
            type_name=_TYPES[entry.kind][0],
            value_count=entry.count,
            value_byte_start=entry.start,
            value_bytes=entry.size,
            inline=entry.inline,
            value_sha256=hashlib.sha256(raw).hexdigest(),
            raw_preview_hex=raw[:1024].hex(),
            raw_preview_truncated=len(raw) > 1024,
            preview_values=None if entry.kind == 7 else values,
            preview_values_truncated=truncated,
            nonfinite_float_count=nonfinite,
            zero_denominator_count=zero_denominator,
            meaning="directory_pointer"
            if entry.tag == 330 or entry.kind == 13
            else "payload_offset_or_count"
            if entry.tag in {273, 279, 324, 325}
            else "opaque_undefined"
            if entry.kind == 7
            else "primitive_observation",
        )


def execute(request: Input, context: OperationContext) -> Output:
    source = context.read_bytes(request.path, suffixes=(".tif", ".tiff"), max_bytes=8_000_000)
    if len(source) < 8 or source[:2] not in {b"II", b"MM"}:
        raise ValueError("TIFF needs a classic II/MM header")
    order: Literal["little", "big"] = "little" if source[:2] == b"II" else "big"
    parser = _Parser(source, order)
    if parser.integer(2, 2) != 42:
        raise ValueError("TIFF magic must be 42; BigTIFF is unsupported")
    first = parser.integer(4, 4)
    if not first:
        raise ValueError("TIFF requires at least one IFD")
    # Iterative three-color DFS detects back edges without recursion. Completed
    # nodes are reused for shared child references; parsing occurs once per IFD.
    state: dict[int, int] = {}
    stack = [(first, False)]
    while stack:
        position, finish = stack.pop()
        if finish:
            state[position] = 2
        elif state.get(position) == 1:
            raise ValueError("TIFF directory graph has a directed cycle")
        elif state.get(position) != 2:
            parser.directory(position)
            state[position] = 1
            stack.append((position, True))
            stack.extend((child, False) for child in reversed(parser.adjacency[position]))
    parser.validate_spans()
    tags: list[Tag] = []
    page_bytes = 0
    for index, (directory, entry) in enumerate(parser.entries):
        tag = parser.preview(directory, entry, index)
        if request.offset <= index < request.offset + request.limit:
            page_bytes += len(json.dumps(tag.model_dump()).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("TIFF encoded tag page exceeds 512 KB")
            tags.append(tag)
    return Output(
        byte_order=order,
        first_ifd_byte_start=first,
        directories=parser.directories,
        edges=parser.edges,
        shared_directory_reference_count=len(parser.edges) - len(parser.directories) + 1,
        tag_count=len(parser.entries),
        typed_value_count=parser.value_count,
        summed_value_bytes=parser.value_bytes,
        payload_reference_count=parser.payload_count,
        tags=tags,
        offset=request.offset,
        has_more=request.offset + len(tags) < len(parser.entries),
        source_bytes=len(source),
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.inspect_tiff_directory",
    kind="plugin",
    description="Inspect bounded classic TIFF directory graphs, typed tag values and declared payload extents without image decompression or image validity claims.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
