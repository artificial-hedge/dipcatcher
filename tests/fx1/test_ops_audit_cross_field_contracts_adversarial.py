"""Adversarial probes for declarative cross-field contracts.

Nonnumeric operands must never order-compare (incompatible_types, not a silent
False); bool never equals a number under either numeric policy; missing fields
are handled before nulls; and a comparison skipped by policy is unassessed —
it can never inflate a pass count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_cross_field_contracts as cfc,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def rule(identifier: str, left: str, operator: str, right: object, **flags: object) -> dict:
    return {
        "id": identifier,
        "left_field": left,
        "operator": operator,
        "right": right,
        **flags,
    }


def literal(value: object) -> dict[str, object]:
    return {"kind": "literal", "value": value}


def field(name: str) -> dict[str, object]:
    return {"kind": "field", "field": name}


def run(args: dict[str, object], context: OperationContext) -> cfc.Output:
    return cfc.execute(cfc.Input.model_validate(args), context)


@pytest.mark.parametrize("operator", ["lt", "le", "gt", "ge"])
def test_strings_never_order_compare(context: OperationContext, operator: str) -> None:
    result = run(
        {
            "rows": [{"a": "2"}],
            "rules": [rule("r", "a", operator, literal(10))],
        },
        context,
    )
    assert result.failed_comparisons == 1
    assert result.rules[0].failure_counts == {"incompatible_types": 1}
    assert result.assessment == "failed"


@pytest.mark.parametrize("policy", ["compatible", "distinct"])
def test_bool_never_equals_number(context: OperationContext, policy: str) -> None:
    result = run(
        {
            "rows": [{"a": True, "b": 1}],
            "rules": [rule("r", "a", "eq", field("b"))],
            "numeric_policy": policy,
        },
        context,
    )
    assert result.rules[0].failure_counts == {"comparison_failed": 1}
    assert not result.passed


def test_numeric_policy_controls_int_float_compatibility(context: OperationContext) -> None:
    args: dict[str, object] = {
        "rows": [{"a": 1, "b": 1.0}],
        "rules": [rule("r", "a", "eq", field("b"))],
    }
    distinct = run({**args, "numeric_policy": "distinct"}, context)
    assert distinct.rules[0].failure_counts == {"comparison_failed": 1}
    compatible = run({**args, "numeric_policy": "compatible"}, context)
    assert compatible.rules[0].passed is True
    assert compatible.assessment == "passed"
    distinct_order = run(
        {
            "rows": [{"a": 1, "b": 1.5}],
            "rules": [rule("r", "a", "lt", field("b"))],
            "numeric_policy": "distinct",
        },
        context,
    )
    assert distinct_order.rules[0].failure_counts == {"incompatible_types": 1}


def test_null_compare_matrix_is_exact(context: OperationContext) -> None:
    base = {"null_policy": "compare"}
    eq_null = run(
        {
            "rows": [{"a": None}],
            "rules": [rule("r", "a", "eq", literal(None), **base)],
        },
        context,
    )
    assert eq_null.rules[0].passed is True
    ne_null = run(
        {
            "rows": [{"a": None}],
            "rules": [rule("r", "a", "ne", literal(None), **base)],
        },
        context,
    )
    assert ne_null.rules[0].failure_counts == {"comparison_failed": 1}
    lt_null = run(
        {
            "rows": [{"a": None}],
            "rules": [rule("r", "a", "lt", literal(5), **base)],
        },
        context,
    )
    assert lt_null.rules[0].failure_counts == {"incompatible_types": 1}
    ne_value = run(
        {
            "rows": [{"a": None}],
            "rules": [rule("r", "a", "ne", literal(5), **base)],
        },
        context,
    )
    assert ne_value.rules[0].passed is True


def test_missing_fields_are_handled_before_nulls(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"b": None}],
            "rules": [
                rule(
                    "r",
                    "absent",
                    "eq",
                    field("b"),
                    null_policy="compare",
                )
            ],
        },
        context,
    )
    assert result.rules[0].failure_counts == {"missing_field": 1}
    assert result.diagnostics[0].missing_fields == ["absent"]


def test_skipped_comparisons_never_count_as_passes(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": 1}, {"b": 2}],
            "rules": [
                rule("r1", "absent", "eq", literal(1), missing_policy="skip"),
                rule("r2", "a", "eq", field("a")),
            ],
        },
        context,
    )
    assert result.skipped_comparisons == 2
    assert result.rules[0].passed is None
    assert result.rules[0].skipped_missing == 2
    assert result.rules[1].passed_comparisons == 1
    assert result.rules[1].failure_counts == {"missing_field": 1}
    assert result.assessment == "failed"
    assert result.unassessed_rows == 0


def test_skip_only_assessment_is_not_a_pass(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": 1}, {"b": 1}],
            "rules": [rule("r", "absent", "eq", literal(1), missing_policy="skip")],
        },
        context,
    )
    assert result.assessment == "no_assessed_comparisons"
    assert result.passed is None
    assert result.unassessed_rows == 2


def test_one_failing_rule_marks_the_row_nonconforming(
    context: OperationContext,
) -> None:
    result = run(
        {
            "rows": [{"a": 1, "b": 2}],
            "rules": [
                rule("r1", "a", "eq", literal(1)),
                rule("r2", "b", "lt", literal(2)),
            ],
        },
        context,
    )
    assert result.checked_comparisons == 2
    assert result.failed_comparisons == 1
    assert result.conforming_rows == 0
    assert result.nonconforming_rows == 1


def test_bounded_work_and_unique_rules_enforced() -> None:
    with pytest.raises(ValidationError):
        cfc.Input.model_validate(
            {
                "rows": [{"a": 1}],
                "rules": [
                    rule("r", "a", "eq", literal(1)),
                    rule("r", "a", "ne", literal(2)),
                ],
            }
        )
    with pytest.raises(ValidationError):
        cfc.Input.model_validate(
            {
                "rows": [{"a": 1}] * 5_000,
                "rules": [rule(f"r{i}", "a", "eq", literal(1)) for i in range(64)],
            }
        )
    with pytest.raises(ValidationError):
        cfc.Input.model_validate({"rows": [], "rules": [rule("r", "a", "eq", literal([1]))]})
    with pytest.raises(ValidationError):
        cfc.Input.model_validate(
            {
                "rows": [{"a": 1}],
                "rules": [rule("r", "a", "between", literal(1))],
            }
        )


def test_zero_diagnostics_still_counts_every_failure(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": i} for i in range(5)],
            "rules": [rule("r", "a", "lt", literal(3))],
            "max_diagnostics": 0,
        },
        context,
    )
    assert result.failed_comparisons == 2
    assert result.checked_comparisons == 5
    assert result.rules[0].passed_comparisons == 3
    assert result.omitted_diagnostics == 2
    assert result.diagnostics == []


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "rows": [{"a": 1, "b": "x"}, {"a": None, "b": 2}],
        "rules": [
            rule("r1", "a", "le", field("b")),
            rule("r2", "a", "eq", literal(1)),
        ],
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
