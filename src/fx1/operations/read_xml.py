"""Read one bounded UTF-8 XML 1.0 document as an element node table.

Rows use preorder IDs, parent IDs and zero-based sibling indices. Names are
expanded into namespace URI plus local name; prefixes and xmlns declarations
are not preserved. Attributes use the same representation and are sorted by
URI/local name. Element text precedes its first element child; child tail is
the text after that child and before the next sibling/end tag. Comments and
processing instructions are counted and omitted, so surrounding text joins.
CDATA boundaries and outer-document whitespace are omitted. Element whitespace,
predefined entities, numeric character references and parser newline/attribute
normalization are retained.

DTD declarations and custom/external entities are rejected by parser callbacks;
no network, external file, XInclude or schema resolution is performed. The
entire document is validated before pagination. Limits: 1 MB source, 5000
elements, 64 nesting levels, 64 attributes per element, 20000 attributes total,
500000 text characters and 500000 attribute-value characters. Individual
text/tail fields are at most 16384 characters and attribute values 4096.
Namespace declarations have separate 64-per-element and 20000-total limits,
with prefix/URI lengths at most 256/1024. Omitted comments and processing
instructions each have 5000-event limits and share a 500000-character budget;
each body is limited to 16384 characters and PI targets to 256.
Pages contain at most 100 nodes and 1 MB encoded node JSON.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, NoReturn
from xml.parsers import expat

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=5000)
    limit: int = Field(default=50, strict=True, ge=1, le=100)


class Attribute(OutputModel):
    namespace_uri: str | None
    local_name: str
    value: str


class Node(OutputModel):
    id: int
    parent_id: int | None
    sibling_index: int
    depth: int
    namespace_uri: str | None
    local_name: str
    attributes: list[Attribute]
    child_count: int
    text: str
    tail: str


class Output(OutputModel):
    nodes: list[Node]
    total_nodes: int
    maximum_depth: int
    total_attributes: int
    total_text_characters: int
    total_attribute_characters: int
    comment_count: int
    processing_instruction_count: int
    namespace_declaration_count: int
    offset: int
    has_more: bool
    document_validated: Literal[True] = True
    custom_entities_allowed: Literal[False] = False
    namespace_representation: Literal["expanded_uri_and_local_name"] = "expanded_uri_and_local_name"
    source_bytes: int
    source_sha256: str


@dataclass
class _Node:
    id: int
    parent_id: int | None
    sibling_index: int
    depth: int
    namespace_uri: str | None
    local_name: str
    attributes: list[Attribute]
    child_count: int = 0
    last_child_id: int | None = None
    text: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)
    text_length: int = 0
    tail_length: int = 0


def _name(expanded: str) -> tuple[str | None, str]:
    uri, separator, local = expanded.partition("\x1f")
    if not separator:
        uri, local = "", uri
    if len(uri) > 1024 or not 1 <= len(local) <= 256:
        raise ValueError("XML namespace/local name exceeds 1024/256 characters")
    return uri or None, local


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".xml",), max_bytes=1_000_000)
    nodes: list[_Node] = []
    stack: list[int] = []
    maximum_depth = total_attributes = total_text = total_attribute_text = 0
    comments = instructions = ignored_markup = 0
    namespace_declarations = pending_namespaces = 0
    parser = expat.ParserCreate(encoding="UTF-8", namespace_separator="\x1f")

    def forbidden(*arguments: object) -> NoReturn:
        raise ValueError("XML DTD declarations and custom/external entities are unsupported")

    def namespace(prefix: str | None, uri: str | None) -> None:
        nonlocal namespace_declarations, pending_namespaces
        namespace_declarations += 1
        pending_namespaces += 1
        if (
            len(prefix or "") > 256
            or len(uri or "") > 1024
            or namespace_declarations > 20_000
            or pending_namespaces > 64
        ):
            raise ValueError("XML namespace declarations exceed name, count or per-element bounds")

    def declaration(version: str, encoding: str | None, standalone: int) -> None:
        if version != "1.0" or encoding is not None and encoding.lower() not in ("utf-8", "utf8"):
            raise ValueError("only XML 1.0 with UTF-8 encoding is supported")

    def start(name: str, attributes: dict[str, str]) -> None:
        nonlocal maximum_depth, total_attributes, total_attribute_text, pending_namespaces
        pending_namespaces = 0
        if len(nodes) >= 5000 or len(stack) >= 64:
            raise ValueError("XML exceeds 5000 elements or 64 nesting levels")
        if len(attributes) > 64:
            raise ValueError("XML element exceeds 64 attributes")
        total_attributes += len(attributes)
        if total_attributes > 20_000:
            raise ValueError("XML exceeds 20000 total attributes")
        uri, local = _name(name)
        converted: list[Attribute] = []
        for attribute_name, value in attributes.items():
            attribute_uri, attribute_local = _name(attribute_name)
            if len(value) > 4096:
                raise ValueError("XML attribute value exceeds 4096 characters")
            total_attribute_text += len(value)
            if total_attribute_text > 500_000:
                raise ValueError("XML attribute values exceed 500000 total characters")
            converted.append(
                Attribute(namespace_uri=attribute_uri, local_name=attribute_local, value=value)
            )
        converted.sort(key=lambda item: (item.namespace_uri or "", item.local_name))
        parent_id = stack[-1] if stack else None
        sibling_index = nodes[parent_id].child_count if parent_id is not None else 0
        node = _Node(len(nodes), parent_id, sibling_index, len(stack) + 1, uri, local, converted)
        if parent_id is not None:
            nodes[parent_id].child_count += 1
            nodes[parent_id].last_child_id = node.id
        nodes.append(node)
        stack.append(node.id)
        maximum_depth = max(maximum_depth, len(stack))

    def end(name: str) -> None:
        stack.pop()

    def text(value: str) -> None:
        nonlocal total_text
        if not stack:
            return
        total_text += len(value)
        if total_text > 500_000:
            raise ValueError("XML text exceeds 500000 total characters")
        parent = nodes[stack[-1]]
        if parent.last_child_id is None:
            parent.text_length += len(value)
            if parent.text_length > 16_384:
                raise ValueError("XML element text exceeds 16384 characters")
            parent.text.append(value)
        else:
            child = nodes[parent.last_child_id]
            child.tail_length += len(value)
            if child.tail_length > 16_384:
                raise ValueError("XML element tail exceeds 16384 characters")
            child.tail.append(value)

    def comment(value: str) -> None:
        nonlocal comments, ignored_markup
        comments += 1
        ignored_markup += len(value)
        if comments > 5000 or len(value) > 16_384 or ignored_markup > 500_000:
            raise ValueError("XML omitted comments/processing instructions exceed their bounds")

    def instruction(target: str, value: str) -> None:
        nonlocal instructions, ignored_markup
        instructions += 1
        ignored_markup += len(target) + len(value)
        if (
            instructions > 5000
            or len(target) > 256
            or len(value) > 16_384
            or ignored_markup > 500_000
        ):
            raise ValueError("XML omitted comments/processing instructions exceed their bounds")

    parser.StartDoctypeDeclHandler = forbidden
    parser.EntityDeclHandler = forbidden
    parser.ExternalEntityRefHandler = forbidden
    parser.UnparsedEntityDeclHandler = forbidden
    parser.NotationDeclHandler = forbidden
    parser.SkippedEntityHandler = forbidden
    parser.XmlDeclHandler = declaration
    parser.StartElementHandler = start
    parser.StartNamespaceDeclHandler = namespace
    parser.EndElementHandler = end
    parser.CharacterDataHandler = text
    parser.CommentHandler = comment
    parser.ProcessingInstructionHandler = instruction
    parser.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    parser.UseForeignDTD(False)
    try:
        decoded = content.decode("utf-8-sig")
        # A single bounded input avoids repeatedly reparsing an unfinished token
        # on older Expat builds. Callbacks still enforce structural limits.
        parser.Parse(decoded, True)
    except (expat.ExpatError, UnicodeError) as exc:
        raise ValueError(f"invalid bounded UTF-8 XML: {exc}") from exc
    page: list[Node] = []
    encoded_bytes = 0
    for node in nodes[request.offset : request.offset + request.limit]:
        converted = Node(
            id=node.id,
            parent_id=node.parent_id,
            sibling_index=node.sibling_index,
            depth=node.depth,
            namespace_uri=node.namespace_uri,
            local_name=node.local_name,
            attributes=node.attributes,
            child_count=node.child_count,
            text="".join(node.text),
            tail="".join(node.tail),
        )
        encoded_bytes += len(
            json.dumps(converted.model_dump(), ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if encoded_bytes > 1_000_000:
            raise ValueError("XML node page exceeds 1 MB encoded JSON; reduce page size")
        page.append(converted)
    return Output(
        nodes=page,
        total_nodes=len(nodes),
        maximum_depth=maximum_depth,
        total_attributes=total_attributes,
        total_text_characters=total_text,
        total_attribute_characters=total_attribute_text,
        comment_count=comments,
        processing_instruction_count=instructions,
        namespace_declaration_count=namespace_declarations,
        offset=request.offset,
        has_more=request.offset + len(page) < len(nodes),
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_xml",
    kind="plugin",
    description="Parse bounded UTF-8 XML into paginated preorder element rows with expanded namespaces, attributes and text/tail fields; reject DTD/custom/external entities and enforce full-document structural limits.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
