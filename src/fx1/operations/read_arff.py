"""Read a bounded, explicitly supported UTF-8 ARFF subset.

Supported: one relation, 1..128 unique attributes, NUMERIC/REAL/INTEGER (all
binary64-range numeric values), nominal declarations, STRING, and comma-separated
dense or sparse records. Numeric cells are the exact decimal lexemes, not rounded
JSON numbers. Sparse indices must increase; omitted numeric cells are "0" and
nominal cells the first declared level. STRING columns must occur explicitly in
every sparse row because Weka's implicit string dictionary index is ambiguous.
An unquoted ? is missing; a quoted ? is a string/nominal literal.

Single/double quotes support backslash escapes for quotes, backslash and n/r/t/b/f.
Percent comments apply outside quotes. Declarations and each record occupy one
physical LF/CRLF line. Unsupported: dates, relational attributes, instance
weights, tab-separated records, multiline quotes and other escape grammars.
Bare CR, BOM, NUL, nonfinite numeric values and nonzero binary64 underflow fail.
All rows, including those outside the requested page, are parsed and checked.

Limits: 4 MB source, 100000 lines, 32768 code points/line, 4096/token (128/numeric),
50000 rows, 2M logical cells, 256 nominal levels/attribute, 256 KB encoded schema,
200 page rows/10000 cells/512 KB encoded page. These are work/output budgets, not
a total-process memory guarantee. Original source bytes are hashed with SHA-256.
Reference: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f", "\\": "\\", "'": "'", '"': '"'}


class Input(InputModel):
    path: str = Field(strict=True, min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=50_000)
    limit: int = Field(default=50, strict=True, ge=1, le=200)


class Attribute(OutputModel):
    name: str
    kind: Literal["numeric", "nominal", "string"]
    declared_type: str
    levels: list[str]
    missing_count: int = 0
    implicit_default_count: int = 0


class Row(OutputModel):
    row_index: int
    source_line: int
    source_byte_start: int
    source_record_bytes: int
    representation: Literal["dense", "sparse"]
    cells: list[str | None]
    implicit_default_columns: list[int]


class Output(OutputModel):
    relation: str
    attributes: list[Attribute]
    rows: list[Row]
    row_count: int
    dense_row_count: int
    sparse_row_count: int
    has_data_rows: bool
    physical_line_count: int
    numeric_encoding: Literal["original_decimal_lexeme_binary64_range"] = (
        "original_decimal_lexeme_binary64_range"
    )
    offset: int
    has_more: bool
    full_source_validated: Literal[True] = True
    source_bytes: int
    source_sha256: str


@dataclass(frozen=True)
class _Token:
    value: str
    quoted: bool = False
    punctuation: bool = False


def _tokens(line: str) -> list[_Token]:
    result: list[_Token] = []
    position = 0
    while position < len(line):
        char = line[position]
        if char in " \t":
            position += 1
            continue
        if char == "%":
            break
        if char in "{},":
            result.append(_Token(char, punctuation=True))
            position += 1
            continue
        quoted = char in "\"'"
        if quoted:
            quote = char
            value: list[str] = []
            position += 1
            while position < len(line) and line[position] != quote:
                char = line[position]
                position += 1
                if char == "\\":
                    if position == len(line) or line[position] not in _ESCAPES:
                        raise ValueError("ARFF has an unsupported quoted escape")
                    char = _ESCAPES[line[position]]
                    position += 1
                value.append(char)
            if position == len(line):
                raise ValueError("ARFF quotes must close on the same physical line")
            position += 1
            if position < len(line) and line[position] not in " \t{},%":
                raise ValueError("ARFF quoted token must be followed by a delimiter")
            text = "".join(value)
        else:
            start = position
            while position < len(line) and line[position] not in " \t{},%":
                char = line[position]
                if char in "\"'\\" or char.isspace() or ord(char) < 32:
                    raise ValueError("ARFF unquoted token contains unsupported whitespace/escape")
                position += 1
            text = line[start:position]
        if len(text) > 4096:
            raise ValueError("ARFF token exceeds 4096 code points")
        result.append(_Token(text, quoted=quoted))
    return result


def _values(tokens: list[_Token]) -> list[_Token]:
    if not tokens or len(tokens) % 2 != 1:
        raise ValueError("ARFF requires nonempty, comma-separated values")
    for index, token in enumerate(tokens):
        if index % 2 == 0:
            if token.punctuation:
                raise ValueError("ARFF expected a scalar value")
        elif not token.punctuation or token.value != ",":
            raise ValueError("ARFF values must be separated by one comma")
    return tokens[::2]


def _name(token: _Token) -> str:
    if token.punctuation or not 1 <= len(token.value) <= 128:
        raise ValueError("ARFF names must have 1..128 characters")
    if any(ord(char) < 32 for char in token.value):
        raise ValueError("ARFF names cannot contain control characters")
    return token.value


def _attribute(tokens: list[_Token]) -> Attribute:
    if len(tokens) < 3:
        raise ValueError("ARFF attribute requires a name and a supported type")
    name = _name(tokens[1])
    declaration = tokens[2:]
    if declaration[0].punctuation and declaration[0].value == "{":
        if not declaration[-1].punctuation or declaration[-1].value != "}":
            raise ValueError("ARFF nominal declaration must close with }")
        levels = [token.value for token in _values(declaration[1:-1])]
        if len(levels) > 256 or len(set(levels)) != len(levels):
            raise ValueError("ARFF nominal levels must be unique and number at most 256")
        if any(token.value == "?" and not token.quoted for token in declaration[1:-1]):
            raise ValueError("ARFF nominal literal ? must be quoted")
        return Attribute(name=name, kind="nominal", declared_type="nominal", levels=levels)
    if len(declaration) != 1 or declaration[0].quoted or declaration[0].punctuation:
        raise ValueError("ARFF unsupported attribute declaration")
    kind = declaration[0].value.lower()
    if kind not in {"numeric", "integer", "real", "string"}:
        raise ValueError("ARFF supports numeric, nominal and string attributes only")
    return Attribute(
        name=name, kind="string" if kind == "string" else "numeric", declared_type=kind, levels=[]
    )


def _cell(token: _Token, attribute: Attribute, levels: set[str]) -> str | None:
    if token.value == "?" and not token.quoted:
        return None
    if attribute.kind == "numeric":
        if token.quoted or len(token.value) > 128 or not _NUMBER.fullmatch(token.value):
            raise ValueError(
                "ARFF numeric value must be an unquoted decimal of at most 128 characters"
            )
        value = float(token.value)
        mantissa = token.value.lower().split("e", 1)[0]
        if not math.isfinite(value) or (
            value == 0 and any(char in "123456789" for char in mantissa)
        ):
            raise ValueError("ARFF numeric value is nonfinite or underflows binary64")
    elif attribute.kind == "nominal" and token.value not in levels:
        raise ValueError(f"ARFF value is not declared for nominal attribute {attribute.name}")
    return token.value


def _record(
    tokens: list[_Token], attributes: list[Attribute], levels: list[set[str]]
) -> tuple[list[str | None], list[int], bool]:
    if not tokens[0].punctuation or tokens[0].value != "{":
        values = _values(tokens)
        if len(values) != len(attributes):
            raise ValueError("ARFF dense row width differs from attribute count")
        return (
            [
                _cell(token, attribute, allowed)
                for token, attribute, allowed in zip(values, attributes, levels, strict=True)
            ],
            [],
            False,
        )
    if not tokens[-1].punctuation or tokens[-1].value != "}":
        raise ValueError("ARFF sparse row must end with }; weights are unsupported")
    entries = tokens[1:-1]
    supplied: dict[int, _Token] = {}
    previous = -1
    position = 0
    while position < len(entries):
        if position + 1 >= len(entries):
            raise ValueError("ARFF sparse entry requires an index and a value")
        index, value = entries[position : position + 2]
        if index.quoted or index.punctuation or not re.fullmatch(r"0|[1-9][0-9]{0,2}", index.value):
            raise ValueError("ARFF sparse index must be a canonical zero-based integer")
        column = int(index.value)
        if column <= previous or column >= len(attributes) or value.punctuation:
            raise ValueError("ARFF sparse indices must increase within the schema")
        supplied[column] = value
        previous = column
        position += 2
        if position < len(entries):
            if entries[position] != _Token(",", punctuation=True) or position + 1 == len(entries):
                raise ValueError("ARFF sparse entries require comma separators")
            position += 1
    cells: list[str | None] = []
    implicit: list[int] = []
    for column, attribute in enumerate(attributes):
        if column in supplied:
            cells.append(_cell(supplied[column], attribute, levels[column]))
        else:
            if attribute.kind == "string":
                raise ValueError("ARFF sparse string attributes must be explicitly supplied")
            cells.append("0" if attribute.kind == "numeric" else attribute.levels[0])
            implicit.append(column)
    return cells, implicit, True


def execute(request: Input, context: OperationContext) -> Output:
    content = context.read_bytes(request.path, suffixes=(".arff",), max_bytes=4_000_000)
    if content.startswith(b"\xef\xbb\xbf") or b"\x00" in content:
        raise ValueError("ARFF initial BOM and NUL are unsupported")
    if content.count(b"\n") > 100_000:
        raise ValueError("ARFF exceeds 100000 physical lines")
    lines = content.split(b"\n") if content else []
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) > 100_000:
        raise ValueError("ARFF exceeds 100000 physical lines")
    relation: str | None = None
    attributes: list[Attribute] = []
    levels: list[set[str]] = []
    attribute_names: set[str] = set()
    in_data = False
    row_count = sparse_count = byte_offset = page_bytes = schema_bytes = 0
    rows: list[Row] = []
    for line_index, raw in enumerate(lines):
        has_lf = line_index + 1 < len(lines) or content.endswith(b"\n")
        record = raw[:-1] if has_lf and raw.endswith(b"\r") else raw
        start = byte_offset
        byte_offset += len(raw) + int(has_lf)
        if b"\r" in record:
            raise ValueError("ARFF bare CR is unsupported")
        text = record.decode("utf-8")
        if len(text) > 32_768 or any(ord(char) < 32 and char != "\t" for char in text):
            raise ValueError("ARFF physical line exceeds length or control-character bounds")
        tokens = _tokens(text)
        if not tokens:
            continue
        if not in_data:
            directive = tokens[0].value.lower() if not tokens[0].quoted else ""
            if directive == "@relation" and relation is None and len(tokens) == 2:
                relation = _name(tokens[1])
            elif directive == "@attribute" and relation is not None:
                attribute = _attribute(tokens)
                if len(attributes) == 128 or attribute.name in attribute_names:
                    raise ValueError("ARFF requires at most 128 uniquely named attributes")
                schema_bytes += len(
                    json.dumps(attribute.model_dump(), ensure_ascii=False).encode("utf-8")
                )
                if schema_bytes > 256_000:
                    raise ValueError("ARFF encoded schema exceeds 256 KB")
                attributes.append(attribute)
                levels.append(set(attribute.levels))
                attribute_names.add(attribute.name)
            elif directive == "@data" and len(tokens) == 1 and attributes:
                if len(attributes) * request.limit > 10_000:
                    raise ValueError("ARFF requested page exceeds 10000 cells")
                in_data = True
            else:
                raise ValueError(
                    f"ARFF unsupported/out-of-order declaration on line {line_index + 1}"
                )
            continue
        if row_count >= 50_000 or (row_count + 1) * len(attributes) > 2_000_000:
            raise ValueError("ARFF exceeds 50000 rows or 2M logical cells")
        cells, implicit, sparse = _record(tokens, attributes, levels)
        sparse_count += int(sparse)
        for attribute, cell in zip(attributes, cells, strict=True):
            attribute.missing_count += int(cell is None)
        for column in implicit:
            attributes[column].implicit_default_count += 1
        if request.offset <= row_count < request.offset + request.limit:
            row = Row(
                row_index=row_count,
                source_line=line_index + 1,
                source_byte_start=start,
                source_record_bytes=len(record),
                representation="sparse" if sparse else "dense",
                cells=cells,
                implicit_default_columns=implicit,
            )
            page_bytes += len(json.dumps(row.model_dump(), ensure_ascii=False).encode("utf-8"))
            if page_bytes > 512_000:
                raise ValueError("ARFF encoded page exceeds 512 KB")
            rows.append(row)
        row_count += 1
    if not in_data or relation is None:
        raise ValueError("ARFF requires relation, attributes and a data declaration")
    return Output(
        relation=relation,
        attributes=attributes,
        rows=rows,
        row_count=row_count,
        dense_row_count=row_count - sparse_count,
        sparse_row_count=sparse_count,
        has_data_rows=bool(row_count),
        physical_line_count=len(lines),
        offset=request.offset,
        has_more=request.offset + len(rows) < row_count,
        source_bytes=len(content),
        source_sha256=hashlib.sha256(content).hexdigest(),
    )


OPERATION = Operation(
    id="plugins.read_arff",
    kind="plugin",
    description="Validate a bounded numeric/nominal/string ARFF subset and return exact paged cells with explicit sparse defaults and source lineage.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
