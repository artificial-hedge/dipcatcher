"""Parse one bounded YAML document into finite JSON without constructors.

YAML containers and quoted/block strings are supported. Plain numeric scalars
use JSON spelling; booleans are lowercase true/false and null is lowercase
null or an empty scalar. Legacy booleans, nondecimal numbers, implicit dates,
explicit tags, anchors, aliases, merge keys and non-string map keys are rejected.
Quote a date-like or legacy boolean-like value when it is intended as text.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

import yaml
from pydantic import Field, JsonValue
from yaml.events import (
    AliasEvent,
    CollectionEndEvent,
    CollectionStartEvent,
    DocumentStartEvent,
    ScalarEvent,
)
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_INTEGER = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_AMBIGUOUS_NUMBER = re.compile(
    r"[+-]?(?:0[bBoOxX][0-9A-Fa-f_]+|[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9_]+)?|\.[0-9_]+(?:[eE][+-]?[0-9_]+)?)\Z"
)
_PREFIX = "tag:yaml.org,2002:"


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)


class Output(OutputModel):
    data: JsonValue
    # Mapping keys count toward the preflight node bound, but are excluded
    # from this count of output values and their containers.
    value_node_count: int
    maximum_depth: int
    scalar_policy: Literal["json_spellings_with_yaml_strings"] = "json_spellings_with_yaml_strings"
    aliases_allowed: Literal[False] = False
    explicit_tags_allowed: Literal[False] = False
    source_sha256: str
    source_bytes: int


def _preflight(decoded: str) -> None:
    documents = nodes = depth = 0
    for event in yaml.parse(decoded, Loader=yaml.SafeLoader):
        if isinstance(event, DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise ValueError("YAML must contain exactly one document")
            if event.tags or (event.version is not None and event.version != (1, 2)):
                raise ValueError(
                    "tag directives and non-1.2 YAML version directives are unsupported"
                )
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ValueError(
                "YAML anchors and aliases are unsupported, including recursive aliases"
            )
        if getattr(event, "tag", None) is not None:
            raise ValueError(
                "explicit YAML tags are unsupported; use ordinary JSON scalar spelling"
            )
        if isinstance(event, (CollectionStartEvent, ScalarEvent)):
            nodes += 1
            if nodes > 20_000:
                raise ValueError("YAML exceeds 20000 scalar/container nodes, including keys")
        if isinstance(event, CollectionStartEvent):
            depth += 1
            if depth > 32:
                raise ValueError("YAML exceeds 32 collection nesting levels")
        elif isinstance(event, CollectionEndEvent):
            depth -= 1
    if documents != 1:
        raise ValueError(
            "YAML must contain exactly one document; an empty stream is not a document"
        )


def _scalar(node: ScalarNode) -> JsonValue:
    value: str = node.value
    if len(value) > 4096:
        raise ValueError("YAML scalar strings cannot exceed 4096 characters")
    value.encode("utf-8")
    if node.tag == _PREFIX + "str":
        if node.style is not None:
            return value
        if not _NUMBER.fullmatch(value):
            if _AMBIGUOUS_NUMBER.fullmatch(value):
                raise ValueError(
                    "ambiguous plain numeric spelling; use a JSON number or quote the text"
                )
            return value
    elif node.tag == _PREFIX + "null":
        if value in ("null", ""):
            return None
        raise ValueError("null scalars must use lowercase null, or quote the intended text")
    elif node.tag == _PREFIX + "bool":
        if value in ("true", "false"):
            return value == "true"
        raise ValueError("legacy YAML booleans are ambiguous; use true/false or quote the text")
    elif node.tag not in (_PREFIX + "int", _PREFIX + "float"):
        raise ValueError("unsupported scalar type; quote dates/times and other intended text")
    if len(value) > 100 or not _NUMBER.fullmatch(value):
        raise ValueError(
            "numeric scalars must use finite JSON number spelling of at most 100 characters"
        )
    if _INTEGER.fullmatch(value):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("YAML number is outside the finite float range")
    if number == 0.0 and any(
        character in "123456789" for character in value.split("e")[0].split("E")[0]
    ):
        raise ValueError("YAML number underflows finite float precision")
    return number


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".yaml", ".yml"), max_bytes=500_000)
    visited = maximum_depth = 0

    def convert(node: Node, depth: int) -> JsonValue:
        nonlocal visited, maximum_depth
        visited += 1
        maximum_depth = max(maximum_depth, depth)
        if isinstance(node, ScalarNode):
            return _scalar(node)
        if isinstance(node, SequenceNode) and node.tag == _PREFIX + "seq":
            return [convert(item, depth + 1) for item in node.value]
        if isinstance(node, MappingNode) and node.tag == _PREFIX + "map":
            mapping: dict[str, JsonValue] = {}
            for key, item in node.value:
                if not isinstance(key, ScalarNode) or key.tag != _PREFIX + "str":
                    raise ValueError(
                        "YAML mapping keys must be strings; merge keys are unsupported"
                    )
                if not isinstance(_scalar(key), str):
                    raise ValueError("numeric YAML mapping keys must be quoted strings")
                if not 1 <= len(key.value) <= 256:
                    raise ValueError("YAML mapping keys must contain 1 through 256 characters")
                key.value.encode("utf-8")
                if key.value in mapping:
                    raise ValueError(f"duplicate YAML mapping key {key.value!r}")
                mapping[key.value] = convert(item, depth + 1)
            return mapping
        raise ValueError("unsupported YAML node type")

    try:
        decoded = content.decode("utf-8-sig")
        _preflight(decoded)
        root = yaml.compose(decoded, Loader=yaml.SafeLoader)
        if root is None:
            raise ValueError("YAML document is absent")
        result = convert(root, 0)
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid bounded UTF-8 YAML: {exc}") from exc
    return Output(
        data=result,
        value_node_count=visited,
        maximum_depth=maximum_depth,
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
    )


OPERATION = Operation(
    id="plugins.read_yaml",
    kind="plugin",
    description=(
        "Read one bounded UTF-8 YAML document as finite JSON with duplicate-key checks, "
        "JSON numeric spellings and explicit ambiguity rejection. Rejects aliases, anchors, "
        "tags, merge keys, non-string keys and implicit dates before unsafe construction."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
