"""Read one bounded MessagePack value with an independent bytecode parser.

All core families are supported with strict UTF-8 strings and unique string map
keys (no normalization). Integers remain exact signed/unsigned decimal strings;
finite binary32/64 use float.hex and original payload bits. Binary is hex.
Map keys are attached to value nodes with separate key spans, not counted as
node rows. Arrays/maps preserve source order in a preorder table. Non-minimal
core encodings are accepted; canonical serialization is not certified.

Standard extension -1 uses exactly timestamp32/fixext4, timestamp64/fixext8 or
timestamp96/ext8-length12. Seconds remain exact decimal strings; nanoseconds
must be in 0..999999999. No calendar conversion or datetime-range truncation.
Other negative extension IDs are rejected. Application IDs 0..127 require an
explicit opaque_bytes policy and remain uninterpreted hex: no object loading,
code execution or recursive decoding. Default rejects application extensions.

Full source is validated before pagination, including all unreturned nodes.
Limits: 2 MB source; 20000 value nodes, 40000 value/key headers; depth 32 with
root depth 0; 10000 entries per array/map, 65536 bytes per string/binary/extension,
256 bytes per key, 100 returned nodes and 512 KB encoded page. Trailing values,
reserved 0xC1, invalid UTF-8, nonfinite floats and truncation fail explicitly.
Reference: https://github.com/msgpack/msgpack/blob/master/spec.md
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
    application_extensions: Literal["reject", "opaque_bytes"] = "reject"
    offset: int = Field(default=0, strict=True, ge=0, le=20_000)
    limit: int = Field(default=50, strict=True, ge=1, le=100)


class Node(OutputModel):
    id: int
    parent_id: int | None
    child_index: int
    depth: int
    map_key: str | None
    map_key_byte_start: int | None
    map_key_byte_length: int | None
    byte_start: int
    byte_length: int = 0
    format_byte_hex: str
    kind: Literal[
        "integer",
        "float",
        "string",
        "binary",
        "array",
        "map",
        "boolean",
        "nil",
        "timestamp",
        "extension",
    ] = "nil"
    value: str | bool | None = None
    scalar_payload_hex: str | None = None
    child_count: int = 0
    extension_type: int | None = None
    timestamp_seconds: str | None = None
    timestamp_nanoseconds: int | None = None


class Output(OutputModel):
    nodes: list[Node]
    total_nodes: int
    parsed_headers: int
    maximum_depth: int
    map_pair_count: int
    timestamp_count: int
    opaque_extension_count: int
    application_extensions: Literal["reject", "opaque_bytes"]
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    canonical_encoding_verified: Literal[False] = False
    object_extensions_loaded: Literal[False] = False
    map_key_policy: Literal["unique_utf8_strings"] = "unique_utf8_strings"
    source_bytes: int
    source_sha256: str


class _Reader:
    def __init__(self, content: bytes, opaque_extensions: bool) -> None:
        self.content = content
        self.position = 0
        self.opaque_extensions = opaque_extensions
        self.nodes: list[Node] = []
        self.headers = self.maximum_depth = self.pairs = self.timestamps = self.extensions = 0

    def read(self, width: int) -> bytes:
        if width < 0 or self.position + width > len(self.content):
            raise ValueError("MessagePack value is truncated")
        start = self.position
        self.position += width
        return self.content[start : self.position]

    def opcode(self) -> int:
        self.headers += 1
        if self.headers > 40_000:
            raise ValueError("MessagePack exceeds 40000 value/key headers")
        return self.read(1)[0]

    def unsigned(self, width: int) -> int:
        return int.from_bytes(self.read(width), "big")

    def string_length(self, opcode: int) -> int:
        if 0xA0 <= opcode <= 0xBF:
            return opcode & 31
        if 0xD9 <= opcode <= 0xDB:
            return self.unsigned(1 << (opcode - 0xD9))
        raise ValueError("MessagePack map keys must be UTF-8 strings")

    def map_key(self) -> tuple[str, int, int]:
        start = self.position
        length = self.string_length(self.opcode())
        if length > 256:
            raise ValueError("MessagePack map key exceeds 256 bytes")
        key = self.read(length).decode("utf-8", errors="strict")
        return key, start, self.position - start

    def container(self, node: Node, count: int, is_map: bool) -> None:
        if count > 10_000:
            raise ValueError("MessagePack container exceeds 10000 entries")
        node.kind = "map" if is_map else "array"
        node.child_count = count
        keys: set[str] = set()
        for index in range(count):
            key = None
            if is_map:
                key = self.map_key()
                if key[0] in keys:
                    raise ValueError("MessagePack map contains duplicate decoded string keys")
                keys.add(key[0])
                self.pairs += 1
            self.value(node.id, index, node.depth + 1, key)

    def extension(self, node: Node, opcode: int) -> None:
        length = (
            (1 << (opcode - 0xD4))
            if 0xD4 <= opcode <= 0xD8
            else self.unsigned(1 << (opcode - 0xC7))
        )
        if length > 65_536:
            raise ValueError("MessagePack extension exceeds 65536 bytes")
        extension_type = int.from_bytes(self.read(1), "big", signed=True)
        payload = self.read(length)
        node.extension_type = extension_type
        if extension_type == -1:
            if opcode == 0xD6:
                seconds, nanoseconds = int.from_bytes(payload, "big"), 0
            elif opcode == 0xD7:
                packed = int.from_bytes(payload, "big")
                seconds, nanoseconds = packed & ((1 << 34) - 1), packed >> 34
            elif opcode == 0xC7 and length == 12:
                nanoseconds = int.from_bytes(payload[:4], "big")
                seconds = int.from_bytes(payload[4:], "big", signed=True)
            else:
                raise ValueError("MessagePack timestamp requires its standard 32/64/96-bit form")
            if nanoseconds > 999_999_999:
                raise ValueError("MessagePack timestamp nanoseconds exceed 999999999")
            node.kind = "timestamp"
            node.timestamp_seconds, node.timestamp_nanoseconds = str(seconds), nanoseconds
            node.scalar_payload_hex = payload.hex()
            self.timestamps += 1
        elif extension_type < 0:
            raise ValueError("MessagePack reserved negative extension is unsupported")
        elif not self.opaque_extensions:
            raise ValueError(
                "MessagePack application extension requires explicit opaque_bytes policy"
            )
        else:
            node.kind, node.value = "extension", payload.hex()
            self.extensions += 1

    def value(
        self,
        parent: int | None,
        child_index: int,
        depth: int,
        key: tuple[str, int, int] | None = None,
    ) -> None:
        if len(self.nodes) >= 20_000 or depth > 32:
            raise ValueError("MessagePack exceeds 20000 value nodes or depth 32")
        self.maximum_depth = max(self.maximum_depth, depth)
        start = self.position
        opcode = self.opcode()
        node = Node(
            id=len(self.nodes),
            parent_id=parent,
            child_index=child_index,
            depth=depth,
            map_key=None if key is None else key[0],
            map_key_byte_start=None if key is None else key[1],
            map_key_byte_length=None if key is None else key[2],
            byte_start=start,
            format_byte_hex=f"{opcode:02x}",
        )
        self.nodes.append(node)
        if opcode <= 0x7F or opcode >= 0xE0:
            node.kind, node.value = "integer", str(opcode if opcode <= 0x7F else opcode - 256)
            node.scalar_payload_hex = ""
        elif 0x80 <= opcode <= 0x9F:
            self.container(node, opcode & 15, opcode < 0x90)
        elif 0xA0 <= opcode <= 0xBF or 0xD9 <= opcode <= 0xDB:
            length = self.string_length(opcode)
            if length > 65_536:
                raise ValueError("MessagePack string exceeds 65536 bytes")
            node.kind, node.value = "string", self.read(length).decode("utf-8", errors="strict")
        elif opcode == 0xC0:
            node.kind = "nil"
        elif opcode in (0xC2, 0xC3):
            node.kind, node.value = "boolean", opcode == 0xC3
        elif 0xC4 <= opcode <= 0xC6:
            length = self.unsigned(1 << (opcode - 0xC4))
            if length > 65_536:
                raise ValueError("MessagePack binary value exceeds 65536 bytes")
            node.kind, node.value = "binary", self.read(length).hex()
        elif 0xC7 <= opcode <= 0xC9 or 0xD4 <= opcode <= 0xD8:
            self.extension(node, opcode)
        elif opcode in (0xCA, 0xCB):
            payload = self.read(4 if opcode == 0xCA else 8)
            number = struct.unpack(">f" if opcode == 0xCA else ">d", payload)[0]
            if not math.isfinite(number):
                raise ValueError("MessagePack nonfinite floats are unsupported")
            node.kind, node.value, node.scalar_payload_hex = "float", number.hex(), payload.hex()
        elif 0xCC <= opcode <= 0xD3:
            signed = opcode >= 0xD0
            payload = self.read(1 << (opcode - (0xD0 if signed else 0xCC)))
            node.kind, node.value = "integer", str(int.from_bytes(payload, "big", signed=signed))
            node.scalar_payload_hex = payload.hex()
        elif 0xDC <= opcode <= 0xDF:
            self.container(node, self.unsigned(2 if opcode in (0xDC, 0xDE) else 4), opcode >= 0xDE)
        else:
            raise ValueError("MessagePack reserved 0xC1 opcode is unsupported")
        node.byte_length = self.position - start


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(
        request.path, suffixes=(".msgpack", ".mpk", ".mpack"), max_bytes=2_000_000
    )
    reader = _Reader(content, request.application_extensions == "opaque_bytes")
    reader.value(None, 0, 0)
    if reader.position != len(content):
        raise ValueError("MessagePack source must contain exactly one value without trailing bytes")
    page = reader.nodes[request.offset : request.offset + request.limit]
    if (
        len(json.dumps([node.model_dump() for node in page], ensure_ascii=False).encode("utf-8"))
        > 512_000
    ):
        raise ValueError("MessagePack encoded node page exceeds 512 KB")
    return Output(
        nodes=page,
        total_nodes=len(reader.nodes),
        parsed_headers=reader.headers,
        maximum_depth=reader.maximum_depth,
        map_pair_count=reader.pairs,
        timestamp_count=reader.timestamps,
        opaque_extension_count=reader.extensions,
        application_extensions=request.application_extensions,
        offset=request.offset,
        has_more=request.offset + len(page) < len(reader.nodes),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_messagepack",
    kind="plugin",
    description="Validate one bounded MessagePack value and return exact typed nodes, timestamps, explicit opaque extensions and byte lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
