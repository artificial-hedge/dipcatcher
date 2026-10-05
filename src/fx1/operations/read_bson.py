"""Read one bounded BSON 1.1 document with an independent little-endian parser.

Supports ordered documents/arrays, double, UTF-8 string, ObjectId, boolean, UTC
datetime, null, regex, int32/int64, MongoDB logical timestamp and min/max key.
Binary supports subtype 0, UUID subtype 4 and MD5 subtype 5; UUID/MD5 must have
16 payload bytes and remain uninterpreted hex. Other binary subtypes, decimal128,
undefined, DBPointer, JavaScript, symbol and code-with-scope are rejected.
No object loading, regex compilation, decryption, decompression or code execution.

Document names must be unique exact UTF-8 text, without normalization. Array
keys must be precisely "0", "1", ... in source order. Strings preserve embedded
NUL; names/regex cstrings cannot contain it. Regex options must be a sorted,
unique subset of imsux, but regex syntax is not evaluated. Integers and UTC
milliseconds are exact decimal strings. Finite doubles use float.hex plus raw
little-endian bits. ObjectIds/binary are hex. Logical timestamps report separate
unsigned seconds/increment fields and are not treated as UTC date objects.

Preorder nodes retain parent, sibling index, field name and complete element
byte span (type + name + value); value_byte_start/length isolates the payload.
Root span is the entire document. Containers have null values; node kind
distinguishes these from null/min/max key. All lengths, nested terminators and
unreturned nodes are validated before pagination; document sequences/trailing
bytes are rejected. Source SHA-256 covers original bytes.

Limits: 2 MB source, 20000 nodes, depth 32 (root depth 0), 10000 entries/container,
256 UTF-8 name bytes, 65536 string/binary/regex-pattern bytes, 100 page nodes and
512 KB encoded page. Reference: https://bsonspec.org/spec.html (version 1.1).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=20_000)
    limit: int = Field(default=50, strict=True, ge=1, le=100)


class Node(OutputModel):
    id: int
    parent_id: int | None
    child_index: int
    depth: int
    name: str | None
    byte_start: int
    byte_length: int = 0
    value_byte_start: int
    value_byte_length: int = 0
    type_byte_hex: str | None
    kind: Literal[
        "document",
        "array",
        "double",
        "string",
        "binary",
        "object_id",
        "boolean",
        "utc_datetime_milliseconds",
        "null",
        "regex",
        "int32",
        "int64",
        "logical_timestamp",
        "min_key",
        "max_key",
    ] = "null"
    value: str | bool | None = None
    scalar_payload_hex: str | None = None
    child_count: int = 0
    binary_subtype: int | None = None
    regex_options: str | None = None
    timestamp_seconds: str | None = None
    timestamp_increment: str | None = None


class Output(OutputModel):
    nodes: list[Node]
    total_nodes: int
    maximum_depth: int
    document_count: int
    array_count: int
    regex_count: int
    binary_count: int
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    regex_syntax_validated: Literal[False] = False
    database_insert_compatibility_verified: Literal[False] = False
    name_policy: Literal["unique_utf8_documents_sequential_decimal_arrays"] = (
        "unique_utf8_documents_sequential_decimal_arrays"
    )
    source_bytes: int
    source_sha256: str


class _DocumentReader:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0
        self.nodes: list[Node] = []
        self.maximum_depth = self.documents = self.arrays = self.regexes = self.binary = 0

    def take(self, length: int, limit: int) -> bytes:
        if length < 0 or self.position + length > limit:
            raise ValueError("BSON value crosses its enclosing document boundary")
        start = self.position
        self.position += length
        return self.content[start : self.position]

    def signed_length(self, limit: int) -> int:
        return int.from_bytes(self.take(4, limit), "little", signed=True)

    def cstring(self, limit: int, maximum: int) -> str:
        end = self.content.find(b"\x00", self.position, min(limit, self.position + maximum + 1))
        if end == -1:
            raise ValueError("BSON cstring is unterminated or exceeds its byte bound")
        value = self.content[self.position : end].decode("utf-8", errors="strict")
        self.position = end + 1
        return value

    def container(self, node: Node, limit: int, is_array: bool) -> None:
        start = self.position
        length = self.signed_length(limit)
        end = start + length
        if length < 5 or end > limit:
            raise ValueError("BSON document length is invalid or exceeds its parent")
        if self.content[end - 1] != 0:
            raise ValueError("BSON document must end with a NUL terminator")
        node.kind = "array" if is_array else "document"
        self.arrays += int(is_array)
        self.documents += int(not is_array)
        keys: set[str] = set()
        count = 0
        while self.position < end - 1:
            if count >= 10_000:
                raise ValueError("BSON document/array exceeds 10000 entries")
            element_start = self.position
            code = self.take(1, end - 1)[0]
            name = self.cstring(end - 1, 256)
            if name in keys:
                raise ValueError("BSON document has duplicate decoded names")
            keys.add(name)
            if is_array and name != str(count):
                raise ValueError("BSON array names must be consecutive canonical decimal indices")
            child = self.new_node(node.id, count, node.depth + 1, name, element_start, code)
            self.element(child, code, end - 1)
            child.value_byte_length = self.position - child.value_byte_start
            child.byte_length = self.position - element_start
            count += 1
        if self.position != end - 1:
            raise ValueError("BSON element accounting does not reach the document terminator")
        self.take(1, end)
        node.child_count = count

    def new_node(
        self,
        parent: int | None,
        index: int,
        depth: int,
        name: str | None,
        start: int,
        code: int | None,
    ) -> Node:
        if depth > 32 or len(self.nodes) >= 20_000:
            raise ValueError("BSON exceeds depth 32 or 20000 nodes")
        self.maximum_depth = max(self.maximum_depth, depth)
        node = Node(
            id=len(self.nodes),
            parent_id=parent,
            child_index=index,
            depth=depth,
            name=name,
            byte_start=start,
            value_byte_start=self.position,
            type_byte_hex=None if code is None else f"{code:02x}",
        )
        self.nodes.append(node)
        return node

    def element(self, node: Node, code: int, limit: int) -> None:
        if code == 1:
            raw = self.take(8, limit)
            value = struct.unpack("<d", raw)[0]
            if not math.isfinite(value):
                raise ValueError("BSON nonfinite doubles are unsupported")
            node.kind, node.value, node.scalar_payload_hex = "double", value.hex(), raw.hex()
        elif code == 2:
            length = self.signed_length(limit)
            if not 1 <= length <= 65_537:
                raise ValueError(
                    "BSON UTF-8 string length must include its terminator and at most 65536 payload bytes"
                )
            raw = self.take(length, limit)
            if raw[-1] != 0:
                raise ValueError("BSON UTF-8 string lacks its declared terminal NUL")
            node.kind, node.value = "string", raw[:-1].decode("utf-8", errors="strict")
        elif code in (3, 4):
            self.container(node, limit, code == 4)
        elif code == 5:
            length = self.signed_length(limit)
            if not 0 <= length <= 65_536:
                raise ValueError("BSON binary length must be in 0..65536")
            subtype = self.take(1, limit)[0]
            if subtype not in (0, 4, 5):
                raise ValueError(
                    "BSON binary subtype is unsupported; only generic/UUID/MD5 bytes are accepted"
                )
            if subtype in (4, 5) and length != 16:
                raise ValueError("BSON UUID/MD5 binary payload must contain exactly 16 bytes")
            node.kind, node.value, node.binary_subtype = (
                "binary",
                self.take(length, limit).hex(),
                subtype,
            )
            self.binary += 1
        elif code == 7:
            node.kind, node.value = "object_id", self.take(12, limit).hex()
        elif code == 8:
            flag = self.take(1, limit)[0]
            if flag not in (0, 1):
                raise ValueError("BSON boolean byte must be zero or one")
            node.kind, node.value = "boolean", bool(flag)
        elif code in (9, 16, 18):
            raw = self.take(4 if code == 16 else 8, limit)
            node.kind = (
                "utc_datetime_milliseconds" if code == 9 else "int32" if code == 16 else "int64"
            )
            node.value, node.scalar_payload_hex = (
                str(int.from_bytes(raw, "little", signed=True)),
                raw.hex(),
            )
        elif code == 10:
            node.kind = "null"
        elif code == 11:
            pattern, options = self.cstring(limit, 65_536), self.cstring(limit, 5)
            if options != "".join(sorted(set(options))) or any(
                option not in "imsux" for option in options
            ):
                raise ValueError("BSON regex options require a sorted unique subset of imsux")
            node.kind, node.value, node.regex_options = "regex", pattern, options
            self.regexes += 1
        elif code == 17:
            raw = self.take(8, limit)
            increment, seconds = struct.unpack("<II", raw)
            node.kind, node.scalar_payload_hex = "logical_timestamp", raw.hex()
            node.timestamp_seconds, node.timestamp_increment = str(seconds), str(increment)
        elif code in (0xFF, 0x7F):
            node.kind = "min_key" if code == 0xFF else "max_key"
        else:
            raise ValueError(
                f"BSON element type 0x{code:02x} is outside the supported non-executable subset"
            )


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".bson",), max_bytes=2_000_000)
    reader = _DocumentReader(content)
    root = reader.new_node(None, 0, 0, None, 0, None)
    reader.container(root, len(content), False)
    root.byte_length = root.value_byte_length = reader.position
    if reader.position != len(content):
        raise ValueError("BSON source must be exactly one document without trailing bytes")
    page = reader.nodes[request.offset : request.offset + request.limit]
    if (
        len(json.dumps([node.model_dump() for node in page], ensure_ascii=False).encode("utf-8"))
        > 512_000
    ):
        raise ValueError("BSON encoded node page exceeds 512 KB")
    return Output(
        nodes=page,
        total_nodes=len(reader.nodes),
        maximum_depth=reader.maximum_depth,
        document_count=reader.documents,
        array_count=reader.arrays,
        regex_count=reader.regexes,
        binary_count=reader.binary,
        offset=request.offset,
        has_more=request.offset + len(page) < len(reader.nodes),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_bson",
    kind="plugin",
    description="Validate one bounded BSON document and return exact scalar/node pages with nested length checks, ordered fields and original byte lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
