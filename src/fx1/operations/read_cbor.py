"""Read one RFC 8949 CBOR item into a bounded preorder node table.

Supports unsigned/negative integers, byte/text strings, arrays, maps, finite
binary16/32/64, booleans, null, undefined and unassigned simple values. Definite
and indefinite strings/containers are supported; text chunks must each be valid
UTF-8. All tags are rejected, including temporal, bignum and self-description
tags. Simple values are observed codes, with no application interpretation.
Map keys must be definite UTF-8 strings, unique by decoded text without Unicode
normalization. Key items/chunks count toward parser work but are not node rows.

Integer values are decimal strings, floats are exact float.hex strings plus
original encoded scalar bytes, binary values are hex, and text is unchanged.
Undefined and container kinds distinguish entries whose value field is null;
simple values retain their unsigned code as a decimal string.
Indefinite string chunks are joined; their number and the original complete
byte span are reported. Node IDs are preorder positions; child_index is array
index/map pair index. Map key spans are separate from their value-node spans.
Non-shortest integer/length arguments are accepted and counted; deterministic
serialization, float minimization and key ordering are not certified.

Full input is validated before a page returns; trailing bytes fail. Limits:
2 MB source, 20000 nodes, 50000 item/chunk/key headers, depth 32 (root depth 0),
10000 children/pairs per container, 65536 joined bytes per string, 256 key bytes,
100 page nodes and 512 KB encoded page. Source SHA-256 covers original bytes.
Reference: https://www.rfc-editor.org/rfc/rfc8949.html sections 3 and 5.
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
    map_key: str | None
    map_key_byte_start: int | None
    map_key_byte_length: int | None
    byte_start: int
    byte_length: int = 0
    kind: Literal[
        "unsigned_integer",
        "negative_integer",
        "bytes",
        "text",
        "array",
        "map",
        "float",
        "boolean",
        "null",
        "undefined",
        "simple",
    ] = "null"
    value: str | bool | None = None
    scalar_encoding_hex: str | None = None
    indefinite: bool = False
    string_chunk_count: int | None = None
    child_count: int = 0


class Output(OutputModel):
    nodes: list[Node]
    total_nodes: int
    parsed_headers: int
    maximum_depth: int
    map_pair_count: int
    string_payload_bytes: int
    nonshortest_argument_count: int
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    deterministic_encoding_verified: Literal[False] = False
    tag_policy: Literal["reject_all_tags"] = "reject_all_tags"
    map_key_policy: Literal["unique_definite_utf8_text"] = "unique_definite_utf8_text"
    source_bytes: int
    source_sha256: str


class _Decoder:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0
        self.nodes: list[Node] = []
        self.headers = self.maximum_depth = self.pairs = self.string_bytes = self.nonshortest = 0

    def take(self, length: int) -> bytes:
        if length < 0 or self.position + length > len(self.content):
            raise ValueError("CBOR item is truncated")
        start = self.position
        self.position += length
        return self.content[start : self.position]

    def head(self) -> tuple[int, int, int | None]:
        self.headers += 1
        if self.headers > 50_000:
            raise ValueError("CBOR exceeds 50000 item/key/chunk headers")
        initial = self.take(1)[0]
        major, additional = initial >> 5, initial & 31
        if additional < 24:
            return major, additional, additional
        if additional in (24, 25, 26, 27):
            width = 1 << (additional - 24)
            argument = int.from_bytes(self.take(width), "big")
            if major != 7 and argument < (24 if width == 1 else 1 << (8 * (width // 2))):
                self.nonshortest += 1
            return major, additional, argument
        if additional == 31 and major in (2, 3, 4, 5):
            return major, additional, None
        raise ValueError("CBOR reserved head, unexpected break or invalid indefinite item")

    def at_break(self) -> bool:
        if self.position >= len(self.content):
            raise ValueError("CBOR indefinite item lacks a terminating break")
        if self.content[self.position] == 0xFF:
            self.position += 1
            return True
        return False

    def string(self, major: int, argument: int | None) -> tuple[bytes, int]:
        if argument is not None:
            if argument > 65_536:
                raise ValueError("CBOR string exceeds 65536 payload bytes")
            data = self.take(argument)
            if major == 3:
                data.decode("utf-8", errors="strict")
            self.string_bytes += len(data)
            return data, 1
        pieces: list[bytes] = []
        total = 0
        while not self.at_break():
            chunk_major, _additional, length = self.head()
            if chunk_major != major or length is None:
                raise ValueError("CBOR indefinite string requires definite chunks of its own type")
            total += length
            if total > 65_536:
                raise ValueError("CBOR joined string exceeds 65536 payload bytes")
            chunk = self.take(length)
            if major == 3:
                chunk.decode("utf-8", errors="strict")
            pieces.append(chunk)
        self.string_bytes += total
        return b"".join(pieces), len(pieces)

    def key(self) -> tuple[str, int, int]:
        start = self.position
        major, _additional, length = self.head()
        if major != 3 or length is None or length > 256:
            raise ValueError("CBOR map keys require definite UTF-8 strings of at most 256 bytes")
        value = self.take(length).decode("utf-8", errors="strict")
        self.string_bytes += length
        return value, start, self.position - start

    def item(
        self,
        parent: int | None,
        child_index: int,
        depth: int,
        key: tuple[str, int, int] | None = None,
    ) -> None:
        if depth > 32 or len(self.nodes) >= 20_000:
            raise ValueError("CBOR exceeds depth 32 or 20000 nodes")
        self.maximum_depth = max(self.maximum_depth, depth)
        start = self.position
        major, additional, argument = self.head()
        if major == 6:
            raise ValueError("CBOR semantic tags are unsupported in this reader")
        node = Node(
            id=len(self.nodes),
            parent_id=parent,
            child_index=child_index,
            depth=depth,
            map_key=None if key is None else key[0],
            map_key_byte_start=None if key is None else key[1],
            map_key_byte_length=None if key is None else key[2],
            byte_start=start,
            indefinite=argument is None,
        )
        self.nodes.append(node)
        if major in (0, 1):
            assert argument is not None
            node.kind = "unsigned_integer" if major == 0 else "negative_integer"
            node.value = str(argument if major == 0 else -1 - argument)
            node.scalar_encoding_hex = self.content[start : self.position].hex()
        elif major in (2, 3):
            data, count = self.string(major, argument)
            node.kind = "bytes" if major == 2 else "text"
            node.value = data.hex() if major == 2 else data.decode("utf-8")
            node.string_chunk_count = count
        elif major in (4, 5):
            node.kind = "array" if major == 4 else "map"
            if argument is not None and argument > 10_000:
                raise ValueError("CBOR container exceeds 10000 children/pairs")
            keys: set[str] = set()
            count = 0
            while argument is None or count < argument:
                if argument is None and self.at_break():
                    break
                if count >= 10_000:
                    raise ValueError("CBOR container exceeds 10000 children/pairs")
                pair_key = None
                if major == 5:
                    pair_key = self.key()
                    if pair_key[0] in keys:
                        raise ValueError("CBOR map has duplicate decoded text keys")
                    keys.add(pair_key[0])
                    self.pairs += 1
                self.item(node.id, count, depth + 1, pair_key)
                count += 1
            node.child_count = count
        elif major == 7:
            self.simple(node, additional, argument)
            node.scalar_encoding_hex = self.content[start : self.position].hex()
        else:
            raise ValueError("CBOR unsupported major type")
        node.byte_length = self.position - start

    def simple(self, node: Node, additional: int, argument: int | None) -> None:
        if additional in (25, 26, 27):
            width = 1 << (additional - 24)
            number = struct.unpack(
                {2: ">e", 4: ">f", 8: ">d"}[width],
                self.content[self.position - width : self.position],
            )[0]
            if not math.isfinite(number):
                raise ValueError("CBOR nonfinite floating-point values are unsupported")
            node.kind, node.value = "float", number.hex()
        elif additional in (20, 21):
            node.kind, node.value = "boolean", additional == 21
        elif additional in (22, 23):
            node.kind = "null" if additional == 22 else "undefined"
        else:
            if argument is None or additional == 24 and argument < 32:
                raise ValueError("CBOR reserved/non-well-formed simple encoding")
            node.kind, node.value = "simple", str(argument)


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".cbor", ".cb"), max_bytes=2_000_000)
    decoder = _Decoder(content)
    decoder.item(None, 0, 0)
    if decoder.position != len(content):
        raise ValueError("CBOR source must contain exactly one item without trailing bytes")
    page = decoder.nodes[request.offset : request.offset + request.limit]
    if (
        len(json.dumps([node.model_dump() for node in page], ensure_ascii=False).encode("utf-8"))
        > 512_000
    ):
        raise ValueError("CBOR encoded node page exceeds 512 KB")
    return Output(
        nodes=page,
        total_nodes=len(decoder.nodes),
        parsed_headers=decoder.headers,
        maximum_depth=decoder.maximum_depth,
        map_pair_count=decoder.pairs,
        string_payload_bytes=decoder.string_bytes,
        nonshortest_argument_count=decoder.nonshortest,
        offset=request.offset,
        has_more=request.offset + len(page) < len(decoder.nodes),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_cbor",
    kind="plugin",
    description="Validate one bounded CBOR item and return typed preorder nodes with exact scalar encodings, string-map rules and original byte spans.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
