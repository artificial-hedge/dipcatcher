"""Audit composite key uniqueness with explicit null and JSON scalar semantics."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

KeyString = Annotated[str, Field(strict=True, max_length=4096)]
Scalar = KeyString | StrictBool | StrictInt | StrictFloat | None
KeyName = Annotated[str, Field(min_length=1, max_length=128, pattern=r"\S")]


class Input(InputModel):
    rows: list[dict[KeyName, Scalar]] = Field(min_length=1, max_length=10_000)
    keys: list[KeyName] = Field(min_length=1, max_length=16)
    null_policy: Literal["reject", "equal", "distinct"] = "reject"
    max_examples: int = Field(default=50, strict=True, ge=1, le=200)
    max_indices_per_group: int = Field(default=20, strict=True, ge=2, le=200)

    @field_validator("keys")
    @classmethod
    def unique_key_names(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("keys must contain distinct field names")
        return value

    @field_validator("rows")
    @classmethod
    def bounded_row_width(cls, value: list[dict[str, Scalar]]) -> list[dict[str, Scalar]]:
        if any(len(row) > 64 for row in value):
            raise ValueError("rows may contain at most 64 fields")
        return value


class InvalidKey(OutputModel):
    row_index: int
    missing_fields: list[str]
    null_fields: list[str]


class DuplicateGroup(OutputModel):
    key_values: dict[str, Scalar]
    row_count: int
    row_indices: list[int]
    omitted_row_indices: int


class Output(OutputModel):
    row_count: int
    keys: list[str]
    null_policy: str
    passed: bool
    invalid_key_rows: int
    null_key_rows: int
    comparable_rows: int
    distinct_keys: int
    duplicate_key_groups: int
    duplicate_rows: int
    rows_in_duplicate_groups: int
    invalid_key_examples: list[InvalidKey]
    omitted_invalid_examples: int
    duplicate_examples: list[DuplicateGroup]
    omitted_duplicate_groups: int


def _token(value: Scalar) -> tuple[str, Scalar]:
    # Numeric 1 and 1.0 have the same key, but True and the string "1" do not.
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, (int, float)):
        return "number", value
    if value is None:
        return "null", None
    return "string", value


def execute(request: Input, context: OperationContext) -> Output:
    groups: dict[tuple[tuple[str, Scalar], ...], list[int]] = defaultdict(list)
    invalid_examples: list[InvalidKey] = []
    invalid_rows = null_rows = comparable_rows = 0

    for row_index, row in enumerate(request.rows):
        missing = [key for key in request.keys if key not in row]
        nulls = [key for key in request.keys if key in row and row[key] is None]
        null_rows += bool(nulls)
        if missing or (nulls and request.null_policy == "reject"):
            invalid_rows += 1
            if len(invalid_examples) < request.max_examples:
                invalid_examples.append(
                    InvalidKey(row_index=row_index, missing_fields=missing, null_fields=nulls)
                )
            continue
        comparable_rows += 1
        if nulls and request.null_policy == "distinct":
            # Each null-bearing row is unique under the explicit distinct policy.
            continue
        groups[tuple(_token(row[key]) for key in request.keys)].append(row_index)

    duplicate_examples: list[DuplicateGroup] = []
    duplicate_groups = duplicate_rows = rows_in_duplicate_groups = 0
    for row_indices in groups.values():
        if len(row_indices) < 2:
            continue
        duplicate_groups += 1
        duplicate_rows += len(row_indices) - 1
        rows_in_duplicate_groups += len(row_indices)
        if len(duplicate_examples) < request.max_examples:
            example_indices = row_indices[: request.max_indices_per_group]
            first = request.rows[row_indices[0]]
            duplicate_examples.append(
                DuplicateGroup(
                    key_values={key: first[key] for key in request.keys},
                    row_count=len(row_indices),
                    row_indices=example_indices,
                    omitted_row_indices=len(row_indices) - len(example_indices),
                )
            )

    return Output(
        row_count=len(request.rows),
        keys=request.keys,
        null_policy=request.null_policy,
        passed=invalid_rows == 0 and duplicate_rows == 0,
        invalid_key_rows=invalid_rows,
        null_key_rows=null_rows,
        comparable_rows=comparable_rows,
        distinct_keys=comparable_rows - duplicate_rows,
        duplicate_key_groups=duplicate_groups,
        duplicate_rows=duplicate_rows,
        rows_in_duplicate_groups=rows_in_duplicate_groups,
        invalid_key_examples=invalid_examples,
        omitted_invalid_examples=invalid_rows - len(invalid_examples),
        duplicate_examples=duplicate_examples,
        omitted_duplicate_groups=duplicate_groups - len(duplicate_examples),
    )


OPERATION = Operation(
    id="skills.audit_duplicate_keys",
    kind="skill",
    description=(
        "Audit supplied JSON rows for composite-key uniqueness and missing keys, "
        "with explicit reject/equal/distinct null policies, type-aware scalar "
        "comparison, exact aggregate counts, and bounded duplicate examples."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
