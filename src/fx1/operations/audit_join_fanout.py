"""Predict equijoin cardinality from key multiplicities without materialization.

Null keys may be rejected, may never match (SQL-style), or may compare equal.
Missing key fields are always invalid. Full-table predictions are withheld if
any key is invalid; separately named valid-subset counts remain available.
Expected relationship constraints apply to keys matched on both sides.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128, pattern=r"\S")]
Cell = (
    Annotated[str, Field(strict=True, max_length=4096)]
    | StrictBool
    | StrictInt
    | StrictFloat
    | None
)
Token = tuple[str, Cell]


class Input(InputModel):
    left_rows: list[dict[Name, Cell]] = Field(max_length=10_000)
    right_rows: list[dict[Name, Cell]] = Field(max_length=10_000)
    left_keys: list[Name] = Field(min_length=1, max_length=16)
    right_keys: list[Name] | None = Field(default=None, min_length=1, max_length=16)
    numeric_policy: Literal["distinct", "compatible"] = "distinct"
    null_policy: Literal["reject", "never_match", "match"] = "never_match"
    expected_relationship: Literal["any", "one_to_one", "many_to_one", "one_to_many"] = "any"
    require_left_match: bool = Field(default=False, strict=True)
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)
    max_row_indices: int = Field(default=5, strict=True, ge=1, le=20)

    @model_validator(mode="after")
    def key_and_cell_bounds(self) -> Self:
        if len(set(self.left_keys)) != len(self.left_keys):
            raise ValueError("left_keys must be distinct")
        if self.right_keys is not None and (
            len(self.right_keys) != len(self.left_keys)
            or len(set(self.right_keys)) != len(self.right_keys)
        ):
            raise ValueError("right_keys must be distinct and match the length of left_keys")
        cells = 0
        for table in (self.left_rows, self.right_rows):
            for row in table:
                if len(row) > 64:
                    raise ValueError("rows may contain at most 64 fields")
                cells += len(row)
        if cells > 250_000:
            raise ValueError("tables may contain at most 250000 fields combined")
        return self


class Finding(OutputModel):
    code: Literal["invalid_key", "null_never_matches", "unmatched_key", "fanout", "many_to_many"]
    table: Literal["left", "right", "both"]
    left_row_count: int = 0
    right_row_count: int = 0
    predicted_inner_rows: int = 0
    left_row_indices: list[int] = Field(default_factory=list)
    right_row_indices: list[int] = Field(default_factory=list)
    omitted_left_indices: int = 0
    omitted_right_indices: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    null_fields: list[str] = Field(default_factory=list)
    violates_relationship: bool = False


class Output(OutputModel):
    left_row_count: int
    right_row_count: int
    invalid_left_rows: int
    invalid_right_rows: int
    left_null_key_rows: int
    right_null_key_rows: int
    matched_left_rows: int
    matched_right_rows: int
    unmatched_valid_left_rows: int
    unmatched_valid_right_rows: int
    matched_key_groups: int
    many_to_many_groups: int
    left_rows_with_multiple_matches: int
    right_rows_with_multiple_matches: int
    predicted_inner_rows: int | None
    predicted_left_join_rows: int | None
    valid_subset_inner_rows: int
    valid_subset_left_join_rows: int
    relationship_violating_groups: int
    expected_relationship: str
    numeric_policy: str
    null_policy: str
    passed: bool
    diagnostic_count: int
    diagnostics: list[Finding]
    omitted_diagnostics: int


def _token(value: Cell, compatible: bool) -> Token:
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, int):
        return ("number" if compatible else "integer"), value
    if isinstance(value, float):
        return ("number" if compatible else "float"), value
    return ("null" if value is None else "string"), value


def execute(request: Input, context: OperationContext) -> Output:
    right_keys = request.right_keys if request.right_keys is not None else request.left_keys
    groups: list[dict[tuple[Token, ...], list[int]]] = [defaultdict(list), defaultdict(list)]
    invalid = [0, 0]
    null_rows = [0, 0]
    nonmatching_nulls = [0, 0]
    diagnostics: list[Finding] = []
    diagnostic_count = 0

    def report(finding: Finding) -> None:
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < request.max_diagnostics:
            diagnostics.append(finding)

    for side, (rows, keys) in enumerate(
        ((request.left_rows, request.left_keys), (request.right_rows, right_keys))
    ):
        table: Literal["left", "right"] = "left" if side == 0 else "right"
        for index, row in enumerate(rows):
            missing = [key for key in keys if key not in row]
            nulls = [key for key in keys if key in row and row[key] is None]
            null_rows[side] += bool(nulls)
            code: Literal["invalid_key", "null_never_matches"] | None = None
            if missing or (nulls and request.null_policy == "reject"):
                invalid[side] += 1
                code = "invalid_key"
            elif nulls and request.null_policy == "never_match":
                nonmatching_nulls[side] += 1
                code = "null_never_matches"
            if code is not None:
                report(
                    Finding(
                        code=code,
                        table=table,
                        left_row_count=1 if side == 0 else 0,
                        right_row_count=1 if side == 1 else 0,
                        left_row_indices=[index] if side == 0 else [],
                        right_row_indices=[index] if side == 1 else [],
                        missing_fields=missing,
                        null_fields=nulls,
                    )
                )
                continue
            key = tuple(_token(row[name], request.numeric_policy == "compatible") for name in keys)
            groups[side][key].append(index)

    inner = matched_left = matched_right = matched_groups = many_many = violations = 0
    left_multiple = right_multiple = 0
    unmatched_left, unmatched_right = nonmatching_nulls
    # Insertion ordering makes diagnostics stable without sorting mixed scalar types.
    all_keys = dict.fromkeys((*groups[0], *groups[1]))
    for key in all_keys:
        left = groups[0].get(key, [])
        right = groups[1].get(key, [])
        lcount, rcount = len(left), len(right)
        violates = False
        code_group: Literal["unmatched_key", "fanout", "many_to_many"] | None = None
        if lcount and rcount:
            matched_groups += 1
            matched_left += lcount
            matched_right += rcount
            inner += lcount * rcount
            left_multiple += lcount if rcount > 1 else 0
            right_multiple += rcount if lcount > 1 else 0
            violates = (
                (request.expected_relationship == "one_to_one" and (lcount > 1 or rcount > 1))
                or (request.expected_relationship == "many_to_one" and rcount > 1)
                or (request.expected_relationship == "one_to_many" and lcount > 1)
            )
            violations += violates
            if lcount > 1 and rcount > 1:
                many_many += 1
                code_group = "many_to_many"
            elif lcount > 1 or rcount > 1:
                code_group = "fanout"
        else:
            unmatched_left += lcount
            unmatched_right += rcount
            code_group = "unmatched_key"
        if code_group is not None:
            left_examples, right_examples = (
                left[: request.max_row_indices],
                right[: request.max_row_indices],
            )
            report(
                Finding(
                    code=code_group,
                    table="both" if left and right else "left" if left else "right",
                    left_row_count=lcount,
                    right_row_count=rcount,
                    predicted_inner_rows=lcount * rcount,
                    left_row_indices=left_examples,
                    right_row_indices=right_examples,
                    omitted_left_indices=lcount - len(left_examples),
                    omitted_right_indices=rcount - len(right_examples),
                    violates_relationship=violates,
                )
            )

    complete = not any(invalid)
    left_total = inner + unmatched_left
    return Output(
        left_row_count=len(request.left_rows),
        right_row_count=len(request.right_rows),
        invalid_left_rows=invalid[0],
        invalid_right_rows=invalid[1],
        left_null_key_rows=null_rows[0],
        right_null_key_rows=null_rows[1],
        matched_left_rows=matched_left,
        matched_right_rows=matched_right,
        unmatched_valid_left_rows=unmatched_left,
        unmatched_valid_right_rows=unmatched_right,
        matched_key_groups=matched_groups,
        many_to_many_groups=many_many,
        left_rows_with_multiple_matches=left_multiple,
        right_rows_with_multiple_matches=right_multiple,
        predicted_inner_rows=inner if complete else None,
        predicted_left_join_rows=left_total if complete else None,
        valid_subset_inner_rows=inner,
        valid_subset_left_join_rows=left_total,
        relationship_violating_groups=violations,
        expected_relationship=request.expected_relationship,
        numeric_policy=request.numeric_policy,
        null_policy=request.null_policy,
        passed=complete
        and violations == 0
        and (not request.require_left_match or unmatched_left == 0),
        diagnostic_count=diagnostic_count,
        diagnostics=diagnostics,
        omitted_diagnostics=diagnostic_count - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_join_fanout",
    kind="skill",
    description="Predict composite equijoin inner/left cardinality, unmatched keys and many-to-many fanout from bounded multiplicities, with explicit null/numeric policies and no Cartesian materialization.",
    input_model=Input,
    output_model=Output,
    handler=execute,
)
