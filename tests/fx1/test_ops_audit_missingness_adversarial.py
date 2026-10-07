"""Adversarial probes for missingness measurement.

Absent fields, explicit nulls, and exact empty strings are three separate
channels — they must never blur into one count. Whitespace-only strings are
values, runs follow supplied order (never sorted), and a table with no column
scope is unassessed, not "clean".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_missingness as miss,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def run(args: dict[str, object], context: OperationContext) -> miss.Output:
    return miss.execute(miss.Input.model_validate(args), context)


def profile(result: miss.Output, name: str) -> miss.ColumnMissingness:
    return [c for c in result.columns if c.column == name][0]


def test_absent_null_and_empty_are_three_channels(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": 1}, {"a": None}, {"a": ""}, {}],
        },
        context,
    )
    column = profile(result, "a")
    assert column.absent_count == 1
    assert column.null_count == 1
    assert column.empty_string_count == 1
    assert column.effective_missing_count == 2
    assert column.observed_value_count == 2


def test_whitespace_only_strings_are_values(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"a": " "}, {"a": "\t"}, {"a": ""}],
            "empty_string_is_missing": True,
        },
        context,
    )
    column = profile(result, "a")
    assert column.empty_string_count == 1
    assert column.effective_missing_count == 1
    assert column.observed_value_count == 2


def test_empty_string_flag_moves_only_effective_counts(
    context: OperationContext,
) -> None:
    args: dict[str, object] = {"rows": [{"a": ""}, {"a": None}, {}]}
    lax = run(args, context)
    strict = run({**args, "empty_string_is_missing": True}, context)
    for result, expected in ((lax, 2), (strict, 3)):
        column = profile(result, "a")
        assert column.empty_string_count == 1
        assert column.effective_missing_count == expected


def test_runs_follow_supplied_order_never_sorted(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{}, {}, {"a": 1}, {}, {"a": 2}, {}, {}, {}],
            "columns": ["a"],
        },
        context,
    )
    column = profile(result, "a")
    assert column.leading_missing_run == 2
    assert column.longest_missing_run == 3
    assert column.trailing_missing_run == 3


def test_declared_columns_preserve_order_and_report_absent(
    context: OperationContext,
) -> None:
    result = run(
        {
            "rows": [{"z": 1, "b": 2}],
            "columns": ["z", "ghost", "b"],
        },
        context,
    )
    assert [c.column for c in result.columns] == ["z", "ghost", "b"]
    ghost = profile(result, "ghost")
    assert ghost.absent_count == 1
    assert ghost.effective_missing_count == 1


def test_union_scope_is_sorted_and_extras_ignored(context: OperationContext) -> None:
    result = run(
        {
            "rows": [{"z": 1, "b": 2, "extra": 3}, {"b": None}],
            "columns": ["b"],
        },
        context,
    )
    assert [c.column for c in result.columns] == ["b"]
    union = run({"rows": [{"z": 1, "b": 2}, {"b": None, "m": 0}]}, context)
    assert [c.column for c in union.columns] == ["b", "m", "z"]


def test_incomplete_row_indices_cover_any_missing_column(
    context: OperationContext,
) -> None:
    result = run(
        {
            "rows": [{"a": 1, "b": 2}, {"a": 1}, {"a": 1, "b": None}, {"a": 1, "b": 2}],
            "max_incomplete_examples": 1,
        },
        context,
    )
    assert result.incomplete_row_count == 2
    assert result.complete_row_count == 2
    assert result.incomplete_row_indices == [1]
    assert result.omitted_incomplete_examples == 1


def test_unassessed_verdicts_are_not_clean(context: OperationContext) -> None:
    no_rows = run({"rows": [], "columns": ["a"]}, context)
    assert no_rows.assessment == "no_observations"
    assert no_rows.no_missing_values is None
    no_columns = run({"rows": [{}, {}]}, context)
    assert no_columns.assessment == "no_columns"
    assert no_columns.no_missing_values is None
    assert no_columns.complete_row_count is None


def test_declared_columns_over_empty_rows(context: OperationContext) -> None:
    result = run({"rows": [{"a": None}, {}], "columns": ["a", "ghost"]}, context)
    ghost = profile(result, "ghost")
    assert ghost.absent_count == 2
    assert result.no_missing_values is False
    assert result.incomplete_row_count == 2


def test_hostile_shapes_rejected() -> None:
    with pytest.raises(ValidationError):
        miss.Input.model_validate({"rows": [{"a": 1}], "columns": ["a", "a"]})
    with pytest.raises(ValidationError):
        miss.Input.model_validate({"rows": [{"a": [1]}]})
    with pytest.raises(ValidationError):
        miss.Input.model_validate({"rows": [{"a": 1}] * 10_001})
    with pytest.raises(ValidationError):
        miss.Input.model_validate({"rows": [{str(i): i} for i in range(257)]})
    with pytest.raises(ValidationError):
        miss.Input.model_validate({"rows": [{str(i): i for i in range(65)}]})


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {"rows": [{"a": 1, "b": None}, {}, {"a": "", "b": 2, "z": " "}]}
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
