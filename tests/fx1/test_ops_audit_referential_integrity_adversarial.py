"""Adversarial probes for composite foreign-key integrity.

An invalid parent row must never satisfy a child reference; duplicated parents
make the reference ambiguous rather than silently matched; and the compatible
numeric policy must compare without lossy float conversion so a large int
cannot collide with a rounded float.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_referential_integrity as ref,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def run(args: dict[str, object], context: OperationContext) -> ref.Output:
    return ref.execute(ref.Input.model_validate(args), context)


def test_invalid_parent_cannot_satisfy_child(context: OperationContext) -> None:
    """A parent missing/null key is flagged AND cannot absorb references."""
    result = run(
        {
            "parent_rows": [{"k": None}, {"j": 1}],
            "child_rows": [{"k": 1}],
            "parent_keys": ["k"],
        },
        context,
    )
    assert result.invalid_parent_rows == 2
    assert result.unmatched_child_rows == 1
    assert result.matched_child_rows == 0
    assert result.assessment == "failed"


def test_duplicate_parents_make_children_ambiguous(context: OperationContext) -> None:
    result = run(
        {
            "parent_rows": [{"k": 1}, {"k": 1}, {"k": 1}],
            "child_rows": [{"k": 1}, {"k": 1}],
            "parent_keys": ["k"],
        },
        context,
    )
    assert result.duplicate_parent_key_groups == 1
    assert result.ambiguous_child_rows == 2
    assert result.matched_child_rows == 0
    finding = [f for f in result.diagnostics if f.code == "ambiguous_child_reference"][0]
    assert finding.parent_match_count == 3


def test_compatible_numbers_never_convert_through_lossy_float(
    context: OperationContext,
) -> None:
    """2**53+1 must not collide with float(2**53) under the compatible policy."""
    result = run(
        {
            "parent_rows": [{"k": 9007199254740993}],
            "child_rows": [{"k": 9007199254740992.0}],
            "parent_keys": ["k"],
            "numeric_policy": "compatible",
        },
        context,
    )
    assert result.unmatched_child_rows == 1
    assert result.matched_child_rows == 0
    twin = run(
        {
            "parent_rows": [{"k": 9007199254740993}],
            "child_rows": [{"k": 9007199254740993}],
            "parent_keys": ["k"],
            "numeric_policy": "compatible",
        },
        context,
    )
    assert twin.matched_child_rows == 1


def test_numeric_policy_split_and_bool_isolation(context: OperationContext) -> None:
    distinct = run(
        {
            "parent_rows": [{"k": 1}],
            "child_rows": [{"k": 1.0}],
            "parent_keys": ["k"],
            "numeric_policy": "distinct",
        },
        context,
    )
    assert distinct.unmatched_child_rows == 1
    compatible = run(
        {
            "parent_rows": [{"k": 1}],
            "child_rows": [{"k": 1.0}],
            "parent_keys": ["k"],
            "numeric_policy": "compatible",
        },
        context,
    )
    assert compatible.matched_child_rows == 1
    for policy in ("distinct", "compatible"):
        bool_child = run(
            {
                "parent_rows": [{"k": 1}],
                "child_rows": [{"k": True}],
                "parent_keys": ["k"],
                "numeric_policy": policy,
            },
            context,
        )
        assert bool_child.unmatched_child_rows == 1


def test_positional_composite_mapping_cannot_be_shuffled(
    context: OperationContext,
) -> None:
    """child_keys[i] binds parent_keys[i]; swapped names must not match."""
    result = run(
        {
            "parent_rows": [{"a": 1, "b": 2}],
            "child_rows": [{"x": 2, "y": 1}],
            "parent_keys": ["a", "b"],
            "child_keys": ["x", "y"],
        },
        context,
    )
    assert result.unmatched_child_rows == 1
    matched = run(
        {
            "parent_rows": [{"a": 1, "b": 2}],
            "child_rows": [{"x": 1, "y": 2}],
            "parent_keys": ["a", "b"],
            "child_keys": ["x", "y"],
        },
        context,
    )
    assert matched.matched_child_rows == 1


def test_ignored_null_children_are_not_comparable(context: OperationContext) -> None:
    result = run(
        {
            "parent_rows": [{"k": 1}],
            "child_rows": [{"k": None}, {"k": 1}],
            "parent_keys": ["k"],
            "child_null_policy": "ignore",
        },
        context,
    )
    assert result.ignored_null_child_rows == 1
    assert result.comparable_child_rows == 1
    assert result.matched_child_rows == 1
    assert result.assessment == "passed"


def test_no_comparable_references_is_not_a_pass(context: OperationContext) -> None:
    """Clean parents with zero comparable children is explicitly unassessed."""
    result = run(
        {
            "parent_rows": [{"k": 1}],
            "child_rows": [{"k": None}],
            "parent_keys": ["k"],
            "child_null_policy": "ignore",
        },
        context,
    )
    assert result.assessment == "no_comparable_references"
    assert result.passed is None
    empty = run(
        {"parent_rows": [{"k": 1}], "child_rows": [], "parent_keys": ["k"]},
        context,
    )
    assert empty.assessment == "no_comparable_references"
    assert empty.passed is None


def test_child_key_mismatch_shapes_rejected() -> None:
    with pytest.raises(ValidationError):
        ref.Input.model_validate(
            {
                "parent_rows": [{"a": 1}],
                "parent_keys": ["a"],
                "child_keys": ["a", "b"],
            }
        )
    with pytest.raises(ValidationError):
        ref.Input.model_validate(
            {
                "parent_rows": [{"a": 1, "b": 2}],
                "parent_keys": ["a", "a"],
            }
        )
    with pytest.raises(ValidationError):
        ref.Input.model_validate(
            {
                "parent_rows": [{"a": 1, "b": 2}],
                "parent_keys": ["a", "b"],
                "child_keys": ["x", "x"],
            }
        )


def test_diagnostics_bounded_with_exact_counts(context: OperationContext) -> None:
    result = run(
        {
            "parent_rows": [{"k": None}] * 3 + [{"k": 1}] * 2,
            "child_rows": [{"k": 9}] * 4 + [{"k": None}] * 3,
            "parent_keys": ["k"],
            "max_diagnostics": 5,
        },
        context,
    )
    assert result.diagnostic_count == 11
    assert len(result.diagnostics) == 5
    assert result.omitted_diagnostics == 6


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "parent_rows": [{"k": 1}, {"k": None}, {"k": 1}],
        "child_rows": [{"k": 1}, {"k": 2}, {"k": "1"}],
        "parent_keys": ["k"],
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
