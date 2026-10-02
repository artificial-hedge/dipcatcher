"""Parse workspace JSON Lines with exact line diagnostics and bounded output.

Every nonblank line must contain one JSON object. Duplicate keys are errors at
any depth; NaN and Infinity are rejected. Field discovery covers the whole
validated file rather than only the returned page.
"""

from __future__ import annotations

import io
import json
import math
from typing import Any

from pydantic import Field, JsonValue

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=0, strict=True, ge=0, le=100_000)
    limit: int = Field(default=100, strict=True, ge=1, le=200)


class Output(OutputModel):
    columns: list[str]
    rows: list[dict[str, JsonValue]]
    physical_line_numbers: list[int]
    total_rows: int
    offset: int
    returned_rows: int
    has_more: bool
    blank_lines_skipped: int
    source_sha256: str
    source_bytes: int


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate object key {name!r}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is outside the finite float range")
    return parsed


def _validate_tree(value: object) -> None:
    """Reject surrogates and excessive nesting before returning JSON to a host."""
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 64:
            raise ValueError("JSON nesting exceeds 64 levels")
        if isinstance(node, str):
            node.encode("utf-8")
        elif isinstance(node, dict):
            for key, item in node.items():
                key.encode("utf-8")
                pending.append((item, depth + 1))
        elif isinstance(node, list):
            pending.extend((item, depth + 1) for item in node)


def execute(request: Input, context: OperationContext) -> Output:
    """Validate all records and return the selected records with source line numbers."""
    rows: list[dict[str, JsonValue]] = []
    line_numbers: list[int] = []
    columns: dict[str, None] = {}
    total_rows = 0
    blanks = 0
    page_source_characters = 0
    with context.open_binary(
        request.path, suffixes=(".jsonl", ".ndjson"), max_bytes=64_000_000
    ) as source:
        buffered = io.BufferedReader(source)
        # Only LF separates records; Unicode line separators may occur inside
        # strings, and bare CR must not silently separate JSON documents.
        with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="\n") as text:
            line_number = 0
            try:
                while line := text.readline(1_000_001):
                    line_number += 1
                    if len(line) > 1_000_000:
                        raise ValueError(
                            "JSON Lines physical lines cannot exceed 1000000 characters"
                        )
                    if not line.strip(" \t\r\n"):
                        blanks += 1
                        continue
                    if total_rows >= 100_000:
                        raise ValueError("JSON Lines input exceeds 100000 records")
                    try:
                        record = json.loads(
                            line,
                            object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant,
                            parse_float=_finite_float,
                        )
                        if not isinstance(record, dict):
                            raise ValueError("each nonblank line must be a JSON object")
                        _validate_tree(record)
                    except (ValueError, RecursionError, UnicodeError) as exc:
                        raise ValueError(
                            f"invalid JSON object on physical line {line_number}: {exc}"
                        ) from exc
                    for name in record:
                        columns.setdefault(name, None)
                    if len(columns) > 1000:
                        raise ValueError("JSON Lines input exceeds 1000 distinct top-level fields")
                    if request.offset <= total_rows < request.offset + request.limit:
                        page_source_characters += len(line)
                        if page_source_characters > 1_000_000:
                            raise ValueError(
                                "JSON Lines page exceeds 1000000 source characters; reduce limit"
                            )
                        rows.append(record)
                        line_numbers.append(line_number)
                    total_rows += 1
            except UnicodeDecodeError as exc:
                raise ValueError("JSON Lines input must be UTF-8 encoded") from exc
            source_sha256, source_bytes = source.source_sha256, source.bytes_read
    return Output(
        columns=list(columns),
        rows=rows,
        physical_line_numbers=line_numbers,
        total_rows=total_rows,
        offset=request.offset,
        returned_rows=len(rows),
        has_more=request.offset + len(rows) < total_rows,
        blank_lines_skipped=blanks,
        source_sha256=source_sha256,
        source_bytes=source_bytes,
    )


OPERATION = Operation(
    id="plugins.read_jsonl",
    kind="plugin",
    description=(
        "Read workspace JSONL/NDJSON objects with duplicate-key, finite-number and nesting "
        "checks; streaming validation up to 64 MB/100000 records, paginated rows with "
        "physical line numbers, and a hash of every source byte."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
    version="1.1.0",
)
