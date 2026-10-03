"""Evaluate explicit scalar comparisons without code evaluation or coercion.

Equality is type-aware; bool never equals a number. Ordering requires two
numbers. Integer/float compatibility is explicit. Missing fields are handled
before nulls; null comparison permits only equality/inequality. Skipped
comparisons are unassessed, never counted as passes.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, model_validator

from fx1.operations.base import InputModel, Operation, OperationContext, OutputModel

Name = Annotated[str, Field(strict=True, min_length=1, max_length=128)]
Text = Annotated[str, Field(strict=True, max_length=1024)]
Integer = Annotated[int, Field(strict=True, ge=-(10**100), le=10**100)]
Real = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-1e100, le=1e100)]
Scalar = Text | StrictBool | Integer | Real | None
Failure = Literal["missing_field", "null_operand", "incompatible_types", "comparison_failed"]


class FieldOperand(InputModel):
    kind: Literal["field"]
    field: Name


class LiteralOperand(InputModel):
    kind: Literal["literal"]
    value: Scalar


class Rule(InputModel):
    id: Name
    left_field: Name
    operator: Literal["eq", "ne", "lt", "le", "gt", "ge"]
    right: Annotated[FieldOperand | LiteralOperand, Field(discriminator="kind")]
    missing_policy: Literal["fail", "skip"] = "fail"
    null_policy: Literal["fail", "skip", "compare"] = "fail"


class Input(InputModel):
    rows: list[dict[Name, Scalar]] = Field(default_factory=list, max_length=10_000)
    rules: list[Rule] = Field(min_length=1, max_length=64)
    numeric_policy: Literal["compatible", "distinct"] = "compatible"
    max_diagnostics: int = Field(default=100, strict=True, ge=0, le=200)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if len({rule.id for rule in self.rules}) != len(self.rules):
            raise ValueError("rule ids must be unique")
        if len(self.rows) * len(self.rules) > 250_000:
            raise ValueError("at most 250000 row/rule comparisons are supported")
        if any(len(row) > 64 for row in self.rows):
            raise ValueError("each row may contain at most 64 fields")
        if sum(map(len, self.rows)) > 250_000:
            raise ValueError("rows may contain at most 250000 fields")
        return self


class Finding(OutputModel):
    row_index: int
    rule_id: str
    code: Failure
    left_value: Scalar
    right_value: Scalar
    missing_fields: list[str]


class RuleSummary(OutputModel):
    rule_id: str
    passed_comparisons: int
    failed_comparisons: int
    skipped_missing: int
    skipped_null: int
    failure_counts: dict[str, int]
    passed: bool | None


class Output(OutputModel):
    row_count: int
    rule_count: int
    numeric_policy: str
    assessment: Literal["passed", "failed", "no_assessed_comparisons"]
    passed: bool | None
    checked_comparisons: int
    failed_comparisons: int
    skipped_comparisons: int
    conforming_rows: int
    nonconforming_rows: int
    unassessed_rows: int
    rules: list[RuleSummary]
    diagnostics: list[Finding]
    omitted_diagnostics: int


def _kind(value: Scalar, *, compatible: bool) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number" if compatible else "integer"
    if isinstance(value, float):
        return "number" if compatible else "float"
    return "string"


def _compare(left: Scalar, right: Scalar, operator: str, *, compatible: bool) -> bool | None:
    left_kind, right_kind = _kind(left, compatible=compatible), _kind(right, compatible=compatible)
    if operator in ("eq", "ne"):
        equal = left_kind == right_kind and left == right
        return equal if operator == "eq" else not equal
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
        or left_kind != right_kind
    ):
        return None
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    raise ValueError("unsupported comparison operator")


def execute(request: Input, context: OperationContext) -> Output:
    row_checked = [False] * len(request.rows)
    row_failed = [False] * len(request.rows)
    summaries: list[RuleSummary] = []
    diagnostics: list[Finding] = []
    checked = failed = skipped = 0
    for rule in request.rules:
        passes = failures = missing_skips = null_skips = 0
        codes: Counter[str] = Counter()
        for index, row in enumerate(request.rows):
            missing = [rule.left_field] if rule.left_field not in row else []
            left = row.get(rule.left_field)
            if isinstance(rule.right, FieldOperand):
                right = row.get(rule.right.field)
                if rule.right.field not in row and rule.right.field not in missing:
                    missing.append(rule.right.field)
            else:
                right = rule.right.value
            code: Failure | None = None
            if missing:
                if rule.missing_policy == "skip":
                    missing_skips += 1
                    continue
                code = "missing_field"
            elif left is None or right is None:
                if rule.null_policy == "skip":
                    null_skips += 1
                    continue
                if rule.null_policy == "fail":
                    code = "null_operand"
            if code is None:
                comparison = _compare(
                    left, right, rule.operator, compatible=request.numeric_policy == "compatible"
                )
                if comparison is None:
                    code = "incompatible_types"
                elif not comparison:
                    code = "comparison_failed"
            row_checked[index] = True
            if code is None:
                passes += 1
            else:
                failures += 1
                codes[code] += 1
                row_failed[index] = True
                if len(diagnostics) < request.max_diagnostics:
                    diagnostics.append(
                        Finding(
                            row_index=index,
                            rule_id=rule.id,
                            code=code,
                            left_value=left,
                            right_value=right,
                            missing_fields=missing,
                        )
                    )
        checked += passes + failures
        failed += failures
        skipped += missing_skips + null_skips
        summaries.append(
            RuleSummary(
                rule_id=rule.id,
                passed_comparisons=passes,
                failed_comparisons=failures,
                skipped_missing=missing_skips,
                skipped_null=null_skips,
                failure_counts=dict(sorted(codes.items())),
                passed=False if failures else (True if passes else None),
            )
        )
    passed = False if failed else (True if checked else None)
    failing_rows = sum(row_failed)
    assessed_rows = sum(row_checked)
    return Output(
        row_count=len(request.rows),
        rule_count=len(request.rules),
        numeric_policy=request.numeric_policy,
        assessment="failed" if failed else ("passed" if checked else "no_assessed_comparisons"),
        passed=passed,
        checked_comparisons=checked,
        failed_comparisons=failed,
        skipped_comparisons=skipped,
        conforming_rows=assessed_rows - failing_rows,
        nonconforming_rows=failing_rows,
        unassessed_rows=len(request.rows) - assessed_rows,
        rules=summaries,
        diagnostics=diagnostics,
        omitted_diagnostics=failed - len(diagnostics),
    )


OPERATION = Operation(
    id="skills.audit_cross_field_contracts",
    kind="skill",
    description=(
        "Check declarative field-to-field or field-to-literal scalar contracts using six "
        "whitelisted comparisons. Applies explicit missing/null and numeric-type policies, "
        "rejects ordering of nonnumeric values, and bounds row/rule work and diagnostics."
    ),
    input_model=Input,
    output_model=Output,
    handler=execute,
)
