"""Adversarial probes for composite-key uniqueness.

Missing fields must never be conflated with explicit nulls; the null policy
(reject/equal/distinct) must be honored exactly; and the type-aware token
semantics (1 == 1.0, True != 1, "1" != 1) must not be blurred by coercion.
Counts stay exact when examples are capped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_duplicate_keys as duplicates,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def test_missing_field_never_counts_as_null(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [{"k": 1}, {}, {"k": None}],
                "keys": ["k"],
                "null_policy": "distinct",
            }
        ),
        context,
    )
    assert result.invalid_key_rows == 1
    assert result.null_key_rows == 1
    assert result.comparable_rows == 2
    assert result.invalid_key_examples[0].missing_fields == ["k"]
    assert result.invalid_key_examples[0].null_fields == []


def test_reject_policy_fails_on_nulls_without_any_duplicate(
    context: OperationContext,
) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {"rows": [{"k": None}, {"k": "a"}], "keys": ["k"], "null_policy": "reject"}
        ),
        context,
    )
    assert result.invalid_key_rows == 1
    assert result.duplicate_rows == 0
    assert not result.passed


def test_equal_policy_groups_nulls_as_duplicates(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [{"k": None}, {"k": None}, {"k": None}, {"k": "a"}],
                "keys": ["k"],
                "null_policy": "equal",
            }
        ),
        context,
    )
    assert result.duplicate_key_groups == 1
    assert result.duplicate_rows == 2
    assert result.rows_in_duplicate_groups == 3
    assert result.distinct_keys == 2


def test_distinct_policy_never_duplicates_null_rows(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [{"k": None}, {"k": None}, {"k": "a"}, {"k": "a"}],
                "keys": ["k"],
                "null_policy": "distinct",
            }
        ),
        context,
    )
    assert result.duplicate_rows == 1
    assert result.rows_in_duplicate_groups == 2
    assert result.distinct_keys == 3


def test_partial_nulls_compare_by_full_composite(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [
                    {"a": None, "b": 1},
                    {"a": None, "b": 1},
                    {"a": None, "b": 2},
                ],
                "keys": ["a", "b"],
                "null_policy": "equal",
            }
        ),
        context,
    )
    assert result.duplicate_key_groups == 1
    assert result.duplicate_examples[0].row_indices == [0, 1]
    assert result.distinct_keys == 2


def test_numeric_boolean_and_string_tokens_do_not_blur(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [
                    {"k": 1},
                    {"k": 1.0},
                    {"k": True},
                    {"k": "1"},
                    {"k": "1.0"},
                    {"k": 0.0},
                    {"k": -0.0},
                ],
                "keys": ["k"],
            }
        ),
        context,
    )
    assert result.distinct_keys == 5
    assert result.duplicate_key_groups == 2
    assert result.duplicate_examples[0].row_indices == [0, 1]
    assert result.duplicate_examples[1].row_indices == [5, 6]


def test_multi_row_group_reports_every_extra_row(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate({"rows": [{"k": i % 2} for i in range(6)], "keys": ["k"]}),
        context,
    )
    assert result.duplicate_key_groups == 2
    assert result.duplicate_rows == 4
    assert result.rows_in_duplicate_groups == 6
    assert result.distinct_keys == 2


@pytest.mark.parametrize("cell", [[1], {"x": 1}, float("nan"), float("inf"), -float("inf")])
def test_non_scalar_cells_rejected(cell: object) -> None:
    with pytest.raises(ValidationError):
        duplicates.Input.model_validate({"rows": [{"k": cell}], "keys": ["k"]})


def test_hostile_key_shapes_rejected() -> None:
    with pytest.raises(ValidationError):
        duplicates.Input(rows=[{"k": 1}], keys=["k", "k"])
    with pytest.raises(ValidationError):
        duplicates.Input(rows=[{str(i): i for i in range(65)}], keys=["0"])
    with pytest.raises(ValidationError):
        duplicates.Input.model_validate({"rows": [{"k": 1}] * 10_001, "keys": ["k"]})
    with pytest.raises(ValidationError):
        duplicates.Input.model_validate({"rows": [{"k": 1}], "keys": []})


def test_examples_bounded_and_omitted_counts_exact(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [{"k": "x"}] * 5 + [{"k": "y"}] * 5 + [{"k": "z"}] * 5,
                "keys": ["k"],
                "max_examples": 2,
                "max_indices_per_group": 3,
            }
        ),
        context,
    )
    assert result.duplicate_key_groups == 3
    assert len(result.duplicate_examples) == 2
    assert result.omitted_duplicate_groups == 1
    assert result.duplicate_examples[0].row_indices == [0, 1, 2]
    assert result.duplicate_examples[0].row_count == 5
    assert result.duplicate_examples[0].omitted_row_indices == 2


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {"rows": [{"k": 1}, {"k": 1.0}, {"k": None}, {}], "keys": ["k"]}
    first = duplicates.execute(duplicates.Input.model_validate(args), context)
    second = duplicates.execute(duplicates.Input.model_validate(args), context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
