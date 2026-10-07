"""Adversarial probes for pairwise missingness association.

Contingency counts must be exact row-level truth, phi/jaccard/conditional
rates must be null (never zero or one by fiat) when the denominator is empty,
and association is descriptive only — a perfect phi is not a market signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_missingness_association as assoc,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def run(args: dict[str, object], context: OperationContext) -> assoc.Output:
    return assoc.execute(assoc.Input.model_validate(args), context)


def pair(result: assoc.Output) -> assoc.Association:
    return result.associations[0]


def test_perfect_overlap_is_exact(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{}, {}, {"a": 1, "b": 2}, {}],
            "columns": ["a", "b"],
        },
        context,
    )
    p = pair(result)
    assert p.both_missing == 3
    assert p.left_missing_only == 0
    assert p.right_missing_only == 0
    assert p.neither_missing == 1
    assert p.jaccard == 1.0
    assert p.phi == 1.0
    assert p.left_missing_given_right_missing == 1.0


def test_disjoint_masks_have_exact_negative_association(
    context: OperationContext,
) -> None:
    result = run(
        {
            "rows": [{"a": 1}, {"b": 2}, {"a": 3}, {"b": 4}],
            "columns": ["a", "b"],
        },
        context,
    )
    p = pair(result)
    assert p.both_missing == 0
    assert p.left_missing_only == 2
    assert p.right_missing_only == 2
    assert p.neither_missing == 0
    assert p.jaccard == 0.0
    assert p.phi == -1.0
    assert p.left_missing_given_right_missing == 0.0
    assert p.right_missing_given_left_missing == 0.0


def test_zero_denominator_statistics_are_null_not_zero(
    context: OperationContext,
) -> None:
    """A constant (always-missing) column makes phi undefined — not 0, not 1."""
    result = run(
        {
            "rows": [{"b": 1}, {"b": 2}],
            "columns": ["a", "b"],
        },
        context,
    )
    p = pair(result)
    assert p.both_missing == 0
    assert p.left_missing_only == 2
    assert p.phi is None
    assert p.jaccard == 0.0
    # b never misses: the conditional rate conditioned on b is undefined.
    assert p.left_missing_given_right_missing is None
    assert p.right_missing_given_left_missing == 0.0


def test_partial_overlap_counts_are_exact(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{}, {"a": 1}, {"a": 1, "b": 2}, {"b": 3}],
            "columns": ["a", "b"],
        },
        context,
    )
    p = pair(result)
    # a missing at rows 0 and 3; b missing at rows 0 and 1.
    assert p.both_missing == 1
    assert p.left_missing_only == 1
    assert p.right_missing_only == 1
    assert p.neither_missing == 1
    assert p.joint_missing_rate == 0.25
    assert p.jaccard == 1 / 3
    assert p.left_missing_given_right_missing == 0.5
    assert p.right_missing_given_left_missing == 0.5
    assert p.phi is not None and -1.0 <= p.phi <= 1.0


def test_empty_table_reports_no_observations_with_null_rates(
    context: OperationContext,
) -> None:
    result = run({"rows": [], "columns": ["a", "b"]}, context)
    assert result.assessment == "no_observations"
    p = pair(result)
    assert p.both_missing == 0
    assert p.jaccard is None
    assert p.phi is None
    assert p.joint_missing_rate is None
    for column in result.columns:
        assert column.missing_rate is None


def test_whitespace_and_empty_strings_stay_separate(context: OperationContext) -> None:
    args: dict[str, object] = {"rows": [{"a": " "}, {"a": ""}], "columns": ["a", "b"]}
    lax = run(args, context)
    assert lax.columns[0].empty_string_count == 1
    assert lax.columns[0].missing_count == 0
    strict = run({**args, "empty_string_is_missing": True}, context)
    assert strict.columns[0].missing_count == 1
    assert strict.columns[0].empty_string_count == 1
    p = pair(strict)
    assert p.both_missing == 1
    assert p.left_missing_only == 0
    assert p.right_missing_only == 1


def test_multi_column_pairs_cover_combinations(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": 1}, {"b": 2}, {}],
            "columns": ["a", "b", "c"],
        },
        context,
    )
    assert result.pair_count == 3
    seen = {(p.left_column, p.right_column) for p in result.associations}
    assert seen == {("a", "b"), ("a", "c"), ("b", "c")}


def test_hostile_column_sets_rejected() -> None:
    with pytest.raises(ValidationError):
        assoc.Input.model_validate({"rows": [{"a": 1}], "columns": ["a"]})
    with pytest.raises(ValidationError):
        assoc.Input.model_validate({"rows": [{"a": 1}], "columns": ["a", "a"]})
    with pytest.raises(ValidationError):
        assoc.Input.model_validate({"rows": [{"a": 1}], "columns": [f"c{i}" for i in range(24)]})
    with pytest.raises(ValidationError):
        assoc.Input.model_validate({"rows": [{"a": {}}], "columns": ["a", "b"]})


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "rows": [{}, {"a": 1, "b": None}, {"a": ""}, {"b": 2}],
        "columns": ["a", "b"],
        "empty_string_is_missing": True,
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
