"""Synthetic correctness fixtures for the independent data-audit operations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_bar_integrity as bars,
)
from fx1.operations import (
    audit_duplicate_keys as duplicates,
)
from fx1.operations import (
    audit_panel_gaps as gaps,
)
from fx1.operations import (
    audit_point_in_time as pit,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def bar(**updates: object) -> dict[str, object]:
    return {"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 0, **updates}


def observation(**updates: object) -> dict[str, object]:
    return {
        "event_time": "2026-01-01T10:00:00Z",
        "available_time": "2026-01-01T10:01:00Z",
        "ingested_time": "2026-01-01T10:02:00Z",
        **updates,
    }


def panel_row(clock: str, security: str = "SYNTHETIC_A") -> dict[str, str]:
    return {"security_id": security, "event_time": clock}


def test_bar_envelope_endpoints_and_zero_volume_are_valid(context: OperationContext) -> None:
    result = bars.execute(
        bars.Input(bars=[bar(open=9, close=12), bar(low=10, high=10, close=10)]), context
    )
    assert result.passed
    assert result.valid_rows == 2
    assert result.violation_count == 0


def test_bar_bad_values_and_ranges_report_without_repair(context: OperationContext) -> None:
    request = bars.Input(
        bars=[
            bar(open=0, high=2, low=3, close=4, volume=-1),
            bar(open=None, high="Infinity", low="NaN", close="oops", volume="2"),
            bar(open=8, close=13),
        ],
        max_diagnostics=3,
    )
    result = bars.execute(request, context)
    assert result.invalid_rows == 3
    assert result.violation_counts == {
        "nonpositive_price": 1,
        "negative_volume": 1,
        "inverted_range": 1,
        "missing_value": 1,
        "nonfinite_value": 2,
        "nonnumeric_value": 2,
        "outside_envelope": 2,
    }
    assert result.violation_count == 10
    assert len(result.diagnostics) == 3
    assert result.omitted_diagnostics == 7
    assert request.bars[0].open == 0
    assert request.bars[1].high == "Infinity"
    json.dumps(result.model_dump(), allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_bar_json_numbers_are_strict_and_finite(value: object) -> None:
    with pytest.raises(ValidationError):
        bars.Input(bars=[bar(close=value)])


def test_point_in_time_timezone_offsets_and_equal_boundaries(context: OperationContext) -> None:
    result = pit.execute(
        pit.Input(
            observations=[
                observation(
                    event_time="2026-01-01T15:30:00+05:30",
                    available_time="2026-01-01T10:00:00Z",
                    ingested_time="2026-01-01T05:00:00-05:00",
                )
            ],
            decision_time="2026-01-01T10:00:00Z",
            require_ingestion_by_decision=True,
        ),
        context,
    )
    assert result.passed
    assert result.ingestion_required
    assert result.ingestion_by_decision_required


def test_point_in_time_reports_all_violations_after_diagnostic_limit(
    context: OperationContext,
) -> None:
    result = pit.execute(
        pit.Input(
            observations=[
                observation(
                    event_time="2026-01-01T10:05:00Z",
                    available_time="2026-01-01T10:04:00Z",
                    ingested_time="2026-01-01T10:03:00Z",
                ),
                observation(ingested_time=None),
            ],
            decision_time="2026-01-01T10:02:00Z",
            require_completed_events=True,
            require_ingestion_by_decision=True,
            max_diagnostics=1,
        ),
        context,
    )
    assert result.invalid_rows == 2
    assert result.violation_count == 6
    assert set(result.violation_counts) == {
        "available_before_event",
        "event_after_decision",
        "available_after_decision",
        "ingested_before_availability",
        "ingested_after_decision",
        "missing_ingestion",
    }
    assert result.omitted_diagnostics == 5


def test_historical_availability_does_not_require_same_day_system_ingestion(
    context: OperationContext,
) -> None:
    args = {
        "observations": [observation(ingested_time="2026-03-01T00:00:00Z")],
        "decision_time": "2026-01-01T10:01:00Z",
    }
    assert pit.execute(pit.Input.model_validate(args), context).passed
    result = pit.execute(
        pit.Input.model_validate({**args, "require_ingestion_by_decision": True}), context
    )
    assert result.violation_counts == {"ingested_after_decision": 1}


def test_optional_ingestion_missing_and_before_availability(context: OperationContext) -> None:
    request = pit.Input(
        observations=[observation(ingested_time=None)], decision_time="2026-01-01T10:02:00Z"
    )
    assert pit.execute(request, context).passed
    request.require_ingestion = True
    assert pit.execute(request, context).violation_counts == {"missing_ingestion": 1}
    request.observations[0].ingested_time = request.observations[0].event_time
    assert pit.execute(request, context).violation_counts == {"ingested_before_availability": 1}


@pytest.mark.parametrize(
    "field", ["event_time", "available_time", "ingested_time", "decision_time"]
)
def test_point_in_time_naive_timestamps_are_not_assumed_utc(field: str) -> None:
    args = {"observations": [observation()], "decision_time": "2026-01-01T10:02:00Z"}
    if field == "decision_time":
        args[field] = "2026-01-01T10:02:00"
    else:
        args["observations"] = [observation(**{field: "2026-01-01T10:02:00"})]
    with pytest.raises(ValidationError):
        pit.Input.model_validate(args)


def test_panel_gaps_keep_security_grids_separate(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input(
            observations=[
                panel_row("2026-01-01T10:00:00Z"),
                panel_row("2026-01-01T10:03:00Z"),
                panel_row("2026-01-01T10:00:30Z", "SYNTHETIC_B"),
                panel_row("2026-01-01T10:01:30Z", "SYNTHETIC_B"),
            ],
            interval_seconds=60,
        ),
        context,
    )
    assert result.security_count == 2
    assert result.expected_grid_points == 6
    assert result.observed_grid_points == 4
    assert result.missing_grid_points == 2
    assert result.off_grid_rows == 0
    finding = result.diagnostics[0]
    assert finding.code == "missing_grid_span"
    assert finding.missing_points == 2
    assert finding.first_missing_time.isoformat() == "2026-01-01T10:01:00+00:00"
    assert finding.last_missing_time.isoformat() == "2026-01-01T10:02:00+00:00"


def test_panel_off_grid_duplicates_and_order_are_independent(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input(
            observations=[
                panel_row("2026-01-01T10:02:30Z"),
                panel_row("2026-01-01T10:00:00Z"),
                panel_row("2026-01-01T15:30:00+05:30"),
                panel_row("2026-01-01T10:00:00.000001Z"),
            ],
            interval_seconds=60,
            max_diagnostics=1,
        ),
        context,
    )
    assert result.duplicate_rows == 1
    assert result.out_of_order_rows == 1
    assert result.off_grid_rows == 2
    assert result.missing_grid_points == 2
    assert result.expected_grid_points == 3
    assert result.observed_grid_points == 1
    assert result.finding_count == 5
    assert len(result.diagnostics) == 1
    assert result.omitted_diagnostics == 4


def test_panel_singleton_and_regular_series_pass(context: OperationContext) -> None:
    result = gaps.execute(
        gaps.Input(
            observations=[
                panel_row("2026-01-01T10:00:00Z"),
                panel_row("2026-01-01T10:01:00Z"),
                panel_row("2026-01-01T10:02:00Z"),
                panel_row("2026-01-01T00:00:00Z", "SYNTHETIC_B"),
            ],
            interval_seconds=60,
        ),
        context,
    )
    assert result.passed
    assert result.expected_grid_points == 4
    assert result.observed_grid_points == 4


def test_panel_very_long_gaps_are_compressed_without_grid_materialization(
    context: OperationContext,
) -> None:
    result = gaps.execute(
        gaps.Input(
            observations=[
                panel_row("0001-01-01T00:00:00Z"),
                panel_row("9999-12-31T23:59:59Z"),
            ],
            interval_seconds=1,
        ),
        context,
    )
    assert result.missing_grid_points > 300_000_000_000
    assert result.finding_count == 1
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].missing_points == result.missing_grid_points


@pytest.mark.parametrize("interval", [0, -1, 0.5, True, 31_536_001])
def test_panel_interval_is_explicit_bounded_whole_seconds(interval: object) -> None:
    with pytest.raises(ValidationError):
        gaps.Input.model_validate(
            {"observations": [panel_row("2026-01-01T00:00:00Z")], "interval_seconds": interval}
        )


def test_panel_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        gaps.Input(observations=[panel_row("2026-01-01T00:00:00")], interval_seconds=60)


def test_duplicate_composite_keys_and_bounded_examples(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input(
            rows=[
                {"security_id": "A", "version": 1},
                {"security_id": "A", "version": 2},
                {"security_id": "A", "version": 1},
                {"security_id": "B", "version": 1},
                {"security_id": "A", "version": 1},
                {"security_id": "B", "version": 1},
            ],
            keys=["security_id", "version"],
            max_examples=1,
            max_indices_per_group=2,
        ),
        context,
    )
    assert result.comparable_rows == 6
    assert result.distinct_keys == 3
    assert result.duplicate_key_groups == 2
    assert result.duplicate_rows == 3
    assert result.rows_in_duplicate_groups == 5
    assert result.omitted_duplicate_groups == 1
    assert result.duplicate_examples[0].row_indices == [0, 2]
    assert result.duplicate_examples[0].row_count == 3
    assert result.duplicate_examples[0].omitted_row_indices == 1


def test_duplicate_numeric_boolean_and_string_semantics(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input(
            rows=[{"key": value} for value in [1, 1.0, True, "1", 0.0, -0.0]], keys=["key"]
        ),
        context,
    )
    assert result.distinct_keys == 4
    assert result.duplicate_key_groups == 2
    assert result.duplicate_rows == 2
    assert result.duplicate_examples[0].row_indices == [0, 1]
    assert result.duplicate_examples[1].row_indices == [4, 5]


@pytest.mark.parametrize(
    ("policy", "invalid", "duplicate_rows", "distinct"),
    [("reject", 2, 0, 1), ("equal", 0, 1, 2), ("distinct", 0, 0, 3)],
)
def test_duplicate_null_policy_is_explicit(
    context: OperationContext, policy: str, invalid: int, duplicate_rows: int, distinct: int
) -> None:
    result = duplicates.execute(
        duplicates.Input.model_validate(
            {
                "rows": [{"key": None}, {"key": None}, {"key": "a"}],
                "keys": ["key"],
                "null_policy": policy,
            }
        ),
        context,
    )
    assert result.null_key_rows == 2
    assert result.invalid_key_rows == invalid
    assert result.duplicate_rows == duplicate_rows
    assert result.distinct_keys == distinct
    assert result.passed == (policy == "distinct")


def test_duplicate_missing_fields_cannot_be_conflated_with_null(context: OperationContext) -> None:
    result = duplicates.execute(
        duplicates.Input(
            rows=[{}, {"key": None}, {}, {"key": "A"}],
            keys=["key"],
            null_policy="distinct",
            max_examples=1,
        ),
        context,
    )
    assert not result.passed
    assert result.invalid_key_rows == 2
    assert result.comparable_rows == 2
    assert result.invalid_key_examples[0].missing_fields == ["key"]
    assert result.invalid_key_examples[0].null_fields == []
    assert result.omitted_invalid_examples == 1


@pytest.mark.parametrize("value", [[1], {"nested": 1}, float("nan"), float("inf")])
def test_duplicate_key_cells_must_be_finite_json_scalars(value: object) -> None:
    with pytest.raises(ValidationError):
        duplicates.Input.model_validate({"rows": [{"key": value}], "keys": ["key"]})


def test_duplicate_keys_reject_repeated_names_and_wide_rows() -> None:
    with pytest.raises(ValidationError):
        duplicates.Input(rows=[{"key": 1}], keys=["key", "key"])
    with pytest.raises(ValidationError):
        duplicates.Input(rows=[{str(i): i for i in range(65)}], keys=["0"])


@pytest.mark.parametrize(
    ("module", "list_key", "row", "other"),
    [
        (bars, "bars", bar(), {}),
        (pit, "observations", observation(), {"decision_time": "2026-01-01T10:02:00Z"}),
        (gaps, "observations", panel_row("2026-01-01T00:00:00Z"), {"interval_seconds": 60}),
        (duplicates, "rows", {"key": 1}, {"keys": ["key"]}),
    ],
)
def test_audit_input_size_limits_and_extra_fields(module, list_key, row, other) -> None:
    with pytest.raises(ValidationError):
        module.Input.model_validate({list_key: [], **other})
    with pytest.raises(ValidationError):
        module.Input.model_validate({list_key: [row] * 10_001, **other})
    with pytest.raises(ValidationError):
        module.Input.model_validate({list_key: [row], **other, "unexpected": True})


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        (bars, {"bars": [bar(close="NaN")]}),
        (pit, {"observations": [observation()], "decision_time": "2026-01-01T10:02:00Z"}),
        (gaps, {"observations": [panel_row("2026-01-01T00:00:00Z")], "interval_seconds": 60}),
        (duplicates, {"rows": [{"key": 1}], "keys": ["key"]}),
    ],
)
def test_operations_have_schema_and_finite_hashed_results(context, module, arguments) -> None:
    manifest = module.OPERATION.describe()
    assert manifest["kind"] == "skill"
    assert manifest["input_schema"]["additionalProperties"] is False
    result = module.OPERATION.invoke(arguments, context)
    assert result["market_evidence"] is False
    assert result["research_only"] is True
    assert len(result["input_sha256"]) == 64
    assert len(result["output_sha256"]) == 64
    assert module.OPERATION.invoke(arguments, context) == result
    json.dumps(result, allow_nan=False)
