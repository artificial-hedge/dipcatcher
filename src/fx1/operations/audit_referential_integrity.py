"""Audit scalar composite foreign keys against a supplied parent table.

Parent keys must be present, non-null and unique. Missing child fields are
always invalid; explicit child nulls may optionally represent absent links.
Bool never equals a numeric key. The numeric policy controls whether integer
1 and float 1.0 match, without converting integers through a lossy float.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Text = Annotated[str, Field(strict=True, max_length=4096)]
Scalar = Text | StrictBool | StrictInt | StrictFloat | None
Token = tuple[str, Scalar]
FailureCode = Literal[
    "invalid_parent_key",
    "duplicate_parent_key",
    "invalid_child_key",
    "unmatched_child_reference",
    "ambiguous_child_reference",
]


class Input(InputModel):
    parent_rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    child_rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    parent_keys: list[Name] = Field(min_length=1, max_length=16)
    child_keys: list[Name] | None = Field(default=None, min_length=1, max_length=16)
    numeric_policy: Literal["distinct", "compatible"] = "distinct"
    child_null_policy: Literal["reject", "ignore"] = "reject"
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_parent_indices: int = Field(default=10, strict=True, ge=1, le=20)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len(self.parent_keys) != len(set(self.parent_keys)):
            raise ValueError("parent_keys must be distinct")
        if self.child_keys is not None:
            if len(self.child_keys) != len(self.parent_keys):
                raise ValueError("child_keys and parent_keys must have equal lengths")
            if len(self.child_keys) != len(set(self.child_keys)):
                raise ValueError("child_keys must be distinct")
        cells = 0
        for table in (self.parent_rows, self.child_rows):
            for row in table:
                if len(row) > 64:
                    raise ValueError("each row may contain at most 64 fields")
                cells += len(row)
        if cells > 250_000:
            raise ValueError("both tables combined may contain at most 250000 fields")
        return self


class Finding(OutputModel):
    code: FailureCode
    table: Literal["parent", "child"]
    row_index: int
    key_values: list[Scalar]
    missing_fields: list[str] = Field(default_factory=list)
    null_fields: list[str] = Field(default_factory=list)
    parent_match_count: int = 0
    parent_row_indices: list[int] = Field(default_factory=list)
    omitted_parent_indices: int = 0


class Output(OutputModel):
    parent_row_count: int
    child_row_count: int
    parent_keys: list[str]
    child_keys: list[str]
    numeric_policy: str
    child_null_policy: str
    assessment: Literal["passed", "failed", "no_comparable_references"]
    passed: bool | None
    invalid_parent_rows: int
    parent_rows_with_missing_fields: int
    parent_rows_with_null_fields: int
    valid_parent_rows: int
    distinct_parent_keys: int
    duplicate_parent_key_groups: int
    extra_duplicate_parent_rows: int
    invalid_child_rows: int
    child_rows_with_missing_fields: int
    child_rows_with_null_fields: int
    ignored_null_child_rows: int
    comparable_child_rows: int
    matched_child_rows: int
    ambiguous_child_rows: int
    unmatched_child_rows: int
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def _token(value: Scalar, *, compatible_numbers: bool) -> Token:
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, int):
        return ("number" if compatible_numbers else "integer"), value
    if isinstance(value, float):
        return ("number" if compatible_numbers else "float"), value
    if value is None:
        return "null", None
    return "string", value


def execute(request: Input, context: OperationContext) -> Output:
    child_keys = request.child_keys if request.child_keys is not None else request.parent_keys
    compatible = request.numeric_policy == "compatible"
    groups: dict[tuple[Token, ...], list[int]] = defaultdict(list)
    diagnostics: list[Finding] = []
    diagnostic_count = 0

    def record(finding: Finding) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    invalid_parents = parent_missing = parent_nulls = 0
    for index, row in enumerate(request.parent_rows):
        missing = [key for key in request.parent_keys if key not in row]
        nulls = [key for key in request.parent_keys if key in row and row[key] is None]
        parent_missing += bool(missing)
        parent_nulls += bool(nulls)
        if missing or nulls:
            invalid_parents += 1
            record(
                Finding(
                    code="invalid_parent_key",
                    table="parent",
                    row_index=index,
                    key_values=[row.get(key) for key in request.parent_keys],
                    missing_fields=missing,
                    null_fields=nulls,
                )
            )
            continue
        token = tuple(
            _token(row[key], compatible_numbers=compatible) for key in request.parent_keys
        )
        groups[token].append(index)

    duplicate_groups = duplicate_rows = 0
    for indices in groups.values():
        if len(indices) > 1:
            duplicate_groups += 1
            duplicate_rows += len(indices) - 1
            example_indices = indices[: request.max_parent_indices]
            record(
                Finding(
                    code="duplicate_parent_key",
                    table="parent",
                    row_index=indices[0],
                    key_values=[
                        request.parent_rows[indices[0]][key] for key in request.parent_keys
                    ],
                    parent_match_count=len(indices),
                    parent_row_indices=example_indices,
                    omitted_parent_indices=len(indices) - len(example_indices),
                )
            )

    invalid_children = child_missing = child_nulls = ignored = comparable = 0
    matched = ambiguous = unmatched = 0
    for index, row in enumerate(request.child_rows):
        missing = [key for key in child_keys if key not in row]
        nulls = [key for key in child_keys if key in row and row[key] is None]
        child_missing += bool(missing)
        child_nulls += bool(nulls)
        values = [row.get(key) for key in child_keys]
        if missing or (nulls and request.child_null_policy == "reject"):
            invalid_children += 1
            record(
                Finding(
                    code="invalid_child_key",
                    table="child",
                    row_index=index,
                    key_values=values,
                    missing_fields=missing,
                    null_fields=nulls,
                )
            )
            continue
        if nulls:
            ignored += 1
            continue
        comparable += 1
        token = tuple(_token(value, compatible_numbers=compatible) for value in values)
        parent_indices = groups.get(token, [])
        if len(parent_indices) == 1:
            matched += 1
        else:
            code: FailureCode
            if parent_indices:
                ambiguous += 1
                code = "ambiguous_child_reference"
            else:
                unmatched += 1
                code = "unmatched_child_reference"
            examples = parent_indices[: request.max_parent_indices]
            record(
                Finding(
                    code=code,
                    table="child",
                    row_index=index,
                    key_values=values,
                    parent_match_count=len(parent_indices),
                    parent_row_indices=examples,
                    omitted_parent_indices=len(parent_indices) - len(examples),
                )
            )

    passed = False if diagnostic_count else (True if comparable else None)
    return Output(
        parent_row_count=len(request.parent_rows),
        child_row_count=len(request.child_rows),
        parent_keys=request.parent_keys,
        child_keys=child_keys,
        numeric_policy=request.numeric_policy,
        child_null_policy=request.child_null_policy,
        assessment="failed"
        if passed is False
        else ("passed" if passed else "no_comparable_references"),
        passed=passed,
        invalid_parent_rows=invalid_parents,
        parent_rows_with_missing_fields=parent_missing,
        parent_rows_with_null_fields=parent_nulls,
        valid_parent_rows=len(request.parent_rows) - invalid_parents,
        distinct_parent_keys=len(groups),
        duplicate_parent_key_groups=duplicate_groups,
        extra_duplicate_parent_rows=duplicate_rows,
        invalid_child_rows=invalid_children,
        child_rows_with_missing_fields=child_missing,
        child_rows_with_null_fields=child_nulls,
        ignored_null_child_rows=ignored,
        comparable_child_rows=comparable,
        matched_child_rows=matched,
        ambiguous_child_rows=ambiguous,
        unmatched_child_rows=unmatched,
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_referential_integrity",
    kind="skill",
    description=(
        "Audit child-to-parent composite keys for missing/null fields, duplicate parents, "
        "unmatched and ambiguous references. Uses typed JSON scalars with explicit numeric "
        "and child-null policies, exact counts, and bounded diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
