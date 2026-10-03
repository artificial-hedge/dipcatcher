"""Read a workspace TOML configuration into JSON with explicit temporal metadata."""

from __future__ import annotations

import hashlib
import math
import tomllib
from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel


class Input(InputModel):
    path: str = Field(min_length=1, max_length=4096)
    table_path: list[Annotated[str, Field(max_length=256)]] = Field(
        default_factory=list, max_length=64
    )


class TemporalValue(OutputModel):
    path: list[str | int]
    kind: Literal["offset_datetime", "local_datetime", "local_date", "local_time"]
    iso_value: str


class Output(OutputModel):
    table_path: list[str]
    data: dict[str, JsonValue]
    temporal_values: list[TemporalValue]
    source_sha256: str
    source_bytes: int


def execute(request: Input, context: OperationContext) -> Output:
    """Parse TOML without environment interpolation, file includes, or execution."""
    content = context.read_bytes(request.path, suffixes=(".toml",))
    try:
        parsed = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError) as exc:
        raise ValueError(f"invalid UTF-8 TOML: {exc}") from exc
    temporals: list[TemporalValue] = []
    visited = 0

    def convert(value: object, path: list[str | int]) -> JsonValue:
        nonlocal visited
        visited += 1
        if visited > 20_000 or len(path) > 64:
            raise ValueError("TOML exceeds 20000 value nodes or 64 nesting levels")
        if isinstance(value, (datetime, date, time)):
            kind: Literal["offset_datetime", "local_datetime", "local_date", "local_time"]
            if isinstance(value, datetime):
                kind = "offset_datetime" if value.utcoffset() is not None else "local_datetime"
            elif isinstance(value, date):
                kind = "local_date"
            else:
                kind = "local_time"
            encoded = value.isoformat()
            temporals.append(TemporalValue(path=path, kind=kind, iso_value=encoded))
            return encoded
        if isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("TOML nonfinite floats cannot be represented as finite JSON")
            return value
        if isinstance(value, list):
            return [convert(item, [*path, index]) for index, item in enumerate(value)]
        if isinstance(value, dict):
            result: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("TOML table keys must be strings")
                if len(key) > 256:
                    raise ValueError("TOML table keys cannot exceed 256 characters")
                result[key] = convert(item, [*path, key])
            return result
        raise ValueError(f"unsupported TOML value type {type(value).__name__}")

    # Validate the full source tree, including tables outside the selected path.
    converted = convert(parsed, [])
    selected: JsonValue = converted
    for component in request.table_path:
        if not isinstance(selected, dict) or component not in selected:
            raise ValueError(f"table_path does not resolve at component {component!r}")
        selected = selected[component]
    if not isinstance(selected, dict):
        raise ValueError("table_path must select a TOML table")
    prefix_length = len(request.table_path)
    selected_temporals = [
        temporal for temporal in temporals if temporal.path[:prefix_length] == request.table_path
    ]
    if len(selected_temporals) > 200:
        raise ValueError("selected TOML table exceeds 200 temporal values")
    return Output(
        table_path=request.table_path,
        data=selected,
        temporal_values=selected_temporals,
        source_sha256=hashlib.sha256(content).hexdigest(),
        source_bytes=len(content),
    )


OPERATION = Operation(
    id="plugins.read_toml",
    kind="plugin",
    description=(
        "Read a UTF-8 workspace TOML table as finite JSON; preserve date/time type information "
        "in path-indexed metadata. Bounds source bytes, tree size and nesting; never executes config."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
