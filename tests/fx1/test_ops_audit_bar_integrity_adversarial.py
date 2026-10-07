"""Adversarial probes for the OHLCV integrity audit.

Every bypass tried here must fail closed: smuggled numeric-looking strings,
nonfinite markers in any casing, inverted ranges hidden behind plausible
open/close prices, and envelope escapes. The audit also must not repair the
supplied rows in place — corrupt inputs stay corrupt and stay flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_bar_integrity as bars,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def bar(**updates: object) -> dict[str, object]:
    return {"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 0, **updates}


@pytest.mark.parametrize(
    "smuggled",
    ["12.5", "0", "-3", "1e9", " 10.0 ", "12,5", "1_000"],
)
def test_numeric_looking_strings_stay_violations(context: OperationContext, smuggled: str) -> None:
    """A string cell is never silently coerced into a valid price."""
    result = bars.execute(bars.Input(bars=[bar(open=smuggled)]), context)
    assert not result.passed
    assert result.invalid_rows == 1
    assert result.violation_counts == {"nonnumeric_value": 1}
    assert result.diagnostics[0].field == "open"


@pytest.mark.parametrize(
    "marker",
    ["NaN", "nan ", " INF", "+Inf", "-inf", "Infinity", "-INFINITY", "+infinity"],
)
def test_nonfinite_markers_flagged_in_any_casing(context: OperationContext, marker: str) -> None:
    result = bars.execute(bars.Input(bars=[bar(close=marker)]), context)
    assert result.violation_counts == {"nonfinite_value": 1}
    assert result.invalid_rows == 1


def test_inverted_range_reports_even_with_plausible_open_close(
    context: OperationContext,
) -> None:
    """high < low must flag even though open/close look individually sane."""
    result = bars.execute(
        bars.Input(bars=[bar(open=11.0, high=9.0, low=10.0, close=11.0)]), context
    )
    assert result.violation_counts == {"inverted_range": 1}
    assert result.diagnostics[0].field == "high"
    assert not result.passed


def test_envelope_escape_on_both_sides_reports_each(context: OperationContext) -> None:
    result = bars.execute(bars.Input(bars=[bar(open=13.0, close=8.0, high=12.0, low=9.0)]), context)
    assert result.violation_counts == {"outside_envelope": 2}
    assert {d.field for d in result.diagnostics} == {"open", "close"}


def test_corrupt_row_is_flagged_and_never_repaired(context: OperationContext) -> None:
    request = bars.Input(bars=[bar(open=-4.0, high="oops", volume=-2.0), bar(), bar(close=0.0)])
    result = bars.execute(request, context)
    assert result.valid_rows == 1
    assert result.invalid_rows == 2
    assert request.bars[0].open == -4.0
    assert request.bars[0].high == "oops"
    assert request.bars[0].volume == -2.0
    assert request.bars[2].close == 0.0


def test_missing_cells_cannot_hide_envelope_checks(context: OperationContext) -> None:
    """A missing high skips the envelope check only because it is already a
    violation — the row can never pass."""
    result = bars.execute(bars.Input(bars=[bar(high=None, open=999.0)]), context)
    assert result.violation_counts == {"missing_value": 1}
    assert result.invalid_rows == 1


def test_results_are_deterministic_and_json_finite(context: OperationContext) -> None:
    request = bars.Input(bars=[bar(open=-1), bar(), bar(high=8.0, low=20.0), bar(volume=-0.5)])
    first = bars.execute(request, context)
    second = bars.execute(request, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)


def test_input_bounds_reject_hostile_shapes() -> None:
    with pytest.raises(ValidationError):
        bars.Input(bars=[])
    with pytest.raises(ValidationError):
        bars.Input.model_validate({"bars": [bar()] * 10_001})
    with pytest.raises(ValidationError):
        bars.Input.model_validate({"bars": [bar()], "max_diagnostics": 201})
    with pytest.raises(ValidationError):
        bars.Input.model_validate({"bars": [bar(close=True)]})
    with pytest.raises(ValidationError):
        bars.Input.model_validate({"bars": [bar(volume=None)], "extra": 1})
