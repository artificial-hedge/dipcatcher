"""Adversarial probes for observed-schema comparison.

Type sets, optionality and nullability are compared without coercion: bool
stays distinct from numbers under either numeric policy, a null-only field
carries an empty type set, and an empty side yields an honest unassessed
verdict — never a vacuous "equal".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_schema_drift as drift,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def run(args: dict[str, object], context: OperationContext) -> drift.Output:
    return drift.execute(drift.Input.model_validate(args), context)


def changes_for(result: drift.Output, name: str) -> list[str]:
    match = [c for c in result.changes if c.name == name]
    return match[0].changes if match else []


def test_added_and_removed_fields_are_not_symmetric(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": 1, "b": "x"}],
            "candidate_rows": [{"a": 1, "c": 2.5}],
        },
        context,
    )
    assert changes_for(result, "b") == ["removed_field"]
    assert changes_for(result, "c") == ["added_field"]
    assert result.change_counts == {"added_field": 1, "removed_field": 1}
    assert result.schema_equal is False


def test_type_set_change_detected_without_coercion(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": 1}],
            "candidate_rows": [{"a": "1"}],
            "numeric_policy": "distinct",
        },
        context,
    )
    assert changes_for(result, "a") == ["non_null_types_changed"]


def test_int_float_policy_split(context: OperationContext) -> None:
    args: dict[str, object] = {
        "reference_rows": [{"a": 1}],
        "candidate_rows": [{"a": 1.0}],
    }
    distinct = run({**args, "numeric_policy": "distinct"}, context)
    assert changes_for(distinct, "a") == ["non_null_types_changed"]
    compatible = run({**args, "numeric_policy": "compatible"}, context)
    assert compatible.schema_equal is True
    assert compatible.changed_field_count == 0


def test_bool_stays_distinct_under_compatible_policy(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": True}],
            "candidate_rows": [{"a": 1}],
            "numeric_policy": "compatible",
        },
        context,
    )
    assert changes_for(result, "a") == ["non_null_types_changed"]


def test_optionality_and_nullability_are_separate_axes(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": 1, "b": 1}, {"a": 1, "b": None}],
            "candidate_rows": [{"a": 1, "b": None}, {"a": 1}],
        },
        context,
    )
    # b: optional in both (one absent vs all present flipped), nullable in both.
    # a: unchanged required non-null integer.
    assert changes_for(result, "a") == []
    assert "optionality_changed" in changes_for(result, "b")
    assert "nullability_changed" not in changes_for(result, "b")


def test_null_only_field_carries_empty_type_set(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": None}],
            "candidate_rows": [{"a": 1}, {"a": None}],
        },
        context,
    )
    # Both sides stay nullable and required; only the type set differs.
    assert changes_for(result, "a") == ["non_null_types_changed"]
    reference = [p for p in result.reference_fields if p.name == "a"][0]
    assert reference.non_null_types == []
    assert reference.nullable


def test_one_field_can_report_multiple_axes(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": 1}, {"a": 2}],
            "candidate_rows": [{"a": "x"}, {}],
        },
        context,
    )
    assert changes_for(result, "a") == [
        "non_null_types_changed",
        "optionality_changed",
    ]


def test_empty_side_cannot_establish_schema_equality(context: OperationContext) -> None:
    for reference_rows, candidate_rows in (
        ([], [{"a": 1}]),
        ([{"a": 1}], []),
        ([], []),
    ):
        result = run(
            {"reference_rows": reference_rows, "candidate_rows": candidate_rows},
            context,
        )
        assert result.assessment == "insufficient_observations"
        assert result.schema_equal is None


def test_bounded_change_list_keeps_exact_counts(context: OperationContext) -> None:
    result = run(
        {
            "reference_rows": [{"a": 1}],
            "candidate_rows": [{"a": 1, "x": 1, "y": 2, "z": 3}],
            "max_changes": 2,
        },
        context,
    )
    assert result.changed_field_count == 3
    assert len(result.changes) == 2
    assert result.omitted_changed_fields == 1
    assert result.change_counts == {"added_field": 3}


def test_hostile_row_shapes_rejected() -> None:
    with pytest.raises(ValidationError):
        drift.Input.model_validate({"reference_rows": [{str(i): i for i in range(65)}]})
    with pytest.raises(ValidationError):
        drift.Input.model_validate(
            {
                "reference_rows": [{"a": 1}],
                "candidate_rows": [{f"c{i}": i} for i in range(257)],
            }
        )


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "reference_rows": [{"a": 1, "b": None}, {"a": 2}],
        "candidate_rows": [{"a": 1.5, "b": None, "c": "x"}],
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
