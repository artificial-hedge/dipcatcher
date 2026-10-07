"""Adversarial probes for the per-security elapsed-grid audit.

An attacker who can reorder, duplicate, or offset rows must not be able to
hide a gap: off-grid rows never substitute for missing grid points, out-of-order
input is flagged rather than silently sorted, and timezone-equivalent instants
cannot fake or erase occupancy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_panel_gaps as gaps,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def row(clock: str, security: str = "SYNTHETIC_A") -> dict[str, str]:
    return {"security_id": security, "event_time": clock}


def test_off_grid_rows_cannot_cover_missing_grid_points(context: OperationContext) -> None:
    """Rows between grid points leave the grid holes fully visible."""
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:00:00Z"),
                    row("2026-01-01T10:00:30Z"),
                    row("2026-01-01T10:03:00Z"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.off_grid_rows == 1
    assert result.missing_grid_points == 2
    assert not result.passed
    span = [f for f in result.diagnostics if f.code == "missing_grid_span"]
    assert span[0].first_missing_time.isoformat() == "2026-01-01T10:01:00+00:00"


def test_trailing_off_grid_row_leaves_a_missing_span(context: OperationContext) -> None:
    """The last row need not be on-grid; slots before it are still owed."""
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:00:00Z"),
                    row("2026-01-01T10:03:30Z"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.expected_grid_points == 4
    assert result.missing_grid_points == 3
    assert {f.code for f in result.diagnostics} == {"off_grid", "missing_grid_span"}


def test_out_of_order_input_is_flagged_not_sorted(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:02:00Z"),
                    row("2026-01-01T10:01:00Z"),
                    row("2026-01-01T10:00:00Z"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.out_of_order_rows == 2
    assert result.missing_grid_points == 0
    assert not result.passed


def test_equivalent_instants_share_one_clock(context: OperationContext) -> None:
    """The same instant in two offsets is a duplicate, not two observations."""
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:00:00Z"),
                    row("2026-01-01T05:00:00-05:00"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.duplicate_rows == 1
    assert result.observed_grid_points == 1


def test_security_ids_are_case_sensitive_and_isolated(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:00:00Z", "a"),
                    row("2026-01-01T10:05:00Z", "a"),
                    row("2026-01-01T10:00:00Z", "A"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.security_count == 2
    a_missing = sum(f.missing_points for f in result.diagnostics if f.security_id == "a")
    assert a_missing == 4
    assert all(f.security_id != "A" or f.code != "missing_grid_span" for f in result.diagnostics)


def test_microsecond_misalignment_cannot_pass_as_grid(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input.model_validate(
            {
                "observations": [
                    row("2026-01-01T10:00:00Z"),
                    row("2026-01-01T10:01:00.000001Z"),
                ],
                "interval_seconds": 60,
            }
        ),
        context,
    )
    assert result.off_grid_rows == 1
    assert not result.passed


@pytest.mark.parametrize("interval", [0, -60, 0.5, True, 31_536_001, "60"])
def test_hostile_intervals_rejected(interval: object) -> None:
    with pytest.raises(ValidationError):
        gaps.Input.model_validate(
            {"observations": [row("2026-01-01T10:00:00Z")], "interval_seconds": interval}
        )


@pytest.mark.parametrize("clock", ["2026-01-01T10:00:00", "soon", "", None, [1]])
def test_hostile_clocks_rejected(clock: object) -> None:
    with pytest.raises(ValidationError):
        gaps.Input.model_validate({"observations": [row(clock)], "interval_seconds": 60})


def test_findings_are_deterministic(context: OperationContext) -> None:
    args = {
        "observations": [
            row("2026-01-01T10:03:00Z"),
            row("2026-01-01T10:00:00Z"),
            row("2026-01-01T10:00:00Z", "b"),
            row("2026-01-01T10:02:30Z", "b"),
        ],
        "interval_seconds": 60,
    }
    first = gaps.execute(gaps.Input.model_validate(args), context)
    second = gaps.execute(gaps.Input.model_validate(args), context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
