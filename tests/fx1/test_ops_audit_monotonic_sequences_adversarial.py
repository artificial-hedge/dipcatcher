"""Adversarial probes for event-order and numeric monotonicity.

Rows are audited in supplied order — a reversed clock is a violation, never
sorted away. Duplicate clocks are caught even when nonadjacent, timezone
equivalence cannot fake distinct instants, and singleton groups are honestly
unassessed rather than passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_monotonic_sequences as mono,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def obs(clock: str, value: float, group: str = "g") -> dict[str, object]:
    return {"group": group, "event_time": clock, "value": value}


def run(args: dict[str, object], context: OperationContext) -> mono.Output:
    return mono.execute(mono.Input.model_validate(args), context)


def test_reversed_clock_is_never_sorted_away(context: OperationContext) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:02:00Z", 3),
                obs("2026-01-01T10:01:00Z", 4),
                obs("2026-01-01T10:03:00Z", 5),
            ]
        },
        context,
    )
    assert result.event_time_reversals == 1
    assert result.assessment == "failed"
    reversal = [f for f in result.diagnostics if f.code == "event_time_reversal"][0]
    assert reversal.row_index == 1
    assert reversal.related_row_index == 0


def test_nonadjacent_duplicate_clocks_are_caught(context: OperationContext) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:00:00Z", 1),
                obs("2026-01-01T10:01:00Z", 2),
                obs("2026-01-01T10:00:00Z", 3),
            ]
        },
        context,
    )
    assert result.duplicate_event_rows == 1
    assert result.event_time_reversals == 1
    dup = [f for f in result.diagnostics if f.code == "duplicate_event_time"][0]
    assert dup.related_row_index == 0


def test_timezone_equivalent_instants_are_one_clock(context: OperationContext) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:00:00Z", 1),
                obs("2026-01-01T05:00:00-05:00", 2),
            ]
        },
        context,
    )
    assert result.duplicate_event_rows == 1
    assert result.assessment == "failed"


def test_allowed_duplicates_are_informational_not_violations(
    context: OperationContext,
) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:00:00Z", 1),
                obs("2026-01-01T10:00:00Z", 2),
                obs("2026-01-01T10:01:00Z", 3),
            ],
            "allow_duplicate_event_times": True,
        },
        context,
    )
    assert result.duplicate_event_rows == 1
    assert result.violation_count == 0
    assert result.assessment == "passed"
    dup = [f for f in result.diagnostics if f.code == "duplicate_event_time"][0]
    assert dup.is_violation is False


def test_strict_flag_separates_equal_steps(context: OperationContext) -> None:
    args: dict[str, object] = {
        "observations": [
            obs("2026-01-01T10:00:00Z", 2),
            obs("2026-01-01T10:01:00Z", 2),
        ]
    }
    lax = run(args, context)
    assert lax.assessment == "passed"
    strict = run({**args, "strict": True}, context)
    assert strict.numeric_direction_violations == 1
    assert strict.assessment == "failed"


def test_decreasing_direction_flags_rises(context: OperationContext) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:00:00Z", 5),
                obs("2026-01-01T10:01:00Z", 4),
                obs("2026-01-01T10:02:00Z", 4.5),
            ],
            "direction": "decreasing",
        },
        context,
    )
    assert result.numeric_direction_violations == 1
    assert result.diagnostics[0].row_index == 2


def test_numeric_check_runs_even_when_clock_order_is_broken(
    context: OperationContext,
) -> None:
    """Adjacent numeric comparisons are not skipped on reversed rows."""
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:02:00Z", 9),
                obs("2026-01-01T10:01:00Z", 1),
            ]
        },
        context,
    )
    assert result.event_time_reversals == 1
    assert result.numeric_direction_violations == 1


def test_singletons_and_empty_input_are_unassessed(context: OperationContext) -> None:
    singleton = run({"observations": [obs("2026-01-01T10:00:00Z", 1)]}, context)
    assert singleton.assessment == "no_comparable_pairs"
    assert singleton.passed is None
    assert singleton.unassessed_groups == 1
    empty = run({"observations": []}, context)
    assert empty.assessment == "no_comparable_pairs"
    assert empty.passed is None


def test_groups_are_isolated(context: OperationContext) -> None:
    result = run(
        {
            "observations": [
                obs("2026-01-01T10:00:00Z", 1, "a"),
                obs("2026-01-01T10:01:00Z", 0, "b"),
                obs("2026-01-01T10:01:00Z", 2, "a"),
            ]
        },
        context,
    )
    assert result.assessment == "passed"
    assert result.event_time_reversals == 0
    assert result.numeric_direction_violations == 0


@pytest.mark.parametrize(
    "clock",
    [
        "1767223200",
        "2026-01-01",
        "later",
        "x" * 65,
        1767223200,
        None,
        "2026-01-01T10:00:00",
    ],
)
def test_hostile_clocks_rejected(clock: object) -> None:
    with pytest.raises(ValidationError):
        mono.Input.model_validate({"observations": [obs(clock, 1)]})


def test_hostile_values_and_group_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        mono.Input.model_validate({"observations": [obs("2026-01-01T10:00:00Z", True)]})
    with pytest.raises(ValidationError):
        mono.Input.model_validate({"observations": [obs("2026-01-01T10:00:00Z", float("nan"))]})
    with pytest.raises(ValidationError):
        mono.Input.model_validate(
            {"observations": [obs("2026-01-01T10:00:00Z", i, f"g{i}") for i in range(257)]}
        )


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "observations": [
            obs("2026-01-01T10:00:00Z", 2, "a"),
            obs("2026-01-01T09:59:00Z", 3, "a"),
            obs("2026-01-01T10:00:00Z", 4, "a"),
        ],
        "allow_duplicate_event_times": True,
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
