"""Adversarial probes for the point-in-time audit.

The decision clock is the security boundary: availability after the decision
is future leakage and must flag under every flag combination. Naive or epoch
clocks cannot be allowed to smuggle in an assumed timezone. Ingestion lineage
violations are opt-in but exact when required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_point_in_time as pit,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def observation(**updates: object) -> dict[str, object]:
    return {
        "event_time": "2026-01-01T10:00:00Z",
        "available_time": "2026-01-01T10:01:00Z",
        "ingested_time": "2026-01-01T10:02:00Z",
        **updates,
    }


def test_availability_after_decision_flags_under_every_flag_mix(
    context: OperationContext,
) -> None:
    for flags in [
        {},
        {"require_completed_events": True},
        {"require_ingestion": True},
        {"require_ingestion_by_decision": True},
    ]:
        result = pit.execute(
            pit.Input.model_validate(
                {
                    "observations": [observation(available_time="2026-01-01T10:02:00.000001Z")],
                    "decision_time": "2026-01-01T10:02:00Z",
                    **flags,
                }
            ),
            context,
        )
        assert not result.passed
        assert result.violation_counts["available_after_decision"] == 1


def test_decision_boundary_is_exact(context: OperationContext) -> None:
    """Equality at the decision instant is allowed; a microsecond later is not."""
    clean = pit.execute(
        pit.Input.model_validate(
            {
                "observations": [observation(available_time="2026-01-01T10:02:00Z")],
                "decision_time": "2026-01-01T10:02:00Z",
                "require_completed_events": True,
            }
        ),
        context,
    )
    assert clean.passed
    leaked = pit.execute(
        pit.Input.model_validate(
            {
                "observations": [
                    observation(
                        available_time="2026-01-01T10:02:00.000001Z",
                        ingested_time="2026-01-01T10:02:00.000001Z",
                    )
                ],
                "decision_time": "2026-01-01T10:02:00Z",
            }
        ),
        context,
    )
    assert leaked.violation_counts == {"available_after_decision": 1}


def test_completed_event_boundary_exact(context: OperationContext) -> None:
    """An event exactly at the decision clock is completed, not future."""
    result = pit.execute(
        pit.Input.model_validate(
            {
                "observations": [
                    observation(
                        event_time="2026-01-01T10:02:00Z",
                        available_time="2026-01-01T10:02:00Z",
                    )
                ],
                "decision_time": "2026-01-01T10:02:00Z",
                "require_completed_events": True,
            }
        ),
        context,
    )
    assert result.passed


def test_ingested_before_available_is_impossible_lineage(
    context: OperationContext,
) -> None:
    result = pit.execute(
        pit.Input.model_validate(
            {
                "observations": [observation(ingested_time="2026-01-01T10:00:30Z")],
                "decision_time": "2026-01-01T10:02:00Z",
            }
        ),
        context,
    )
    assert result.violation_counts == {"ingested_before_availability": 1}


def test_late_ingestion_only_flags_under_ingestion_constraint(
    context: OperationContext,
) -> None:
    args = {
        "observations": [observation(ingested_time="2026-01-03T00:00:00Z")],
        "decision_time": "2026-01-01T10:02:00Z",
    }
    assert pit.execute(pit.Input.model_validate(args), context).passed
    flagged = pit.execute(
        pit.Input.model_validate({**args, "require_ingestion_by_decision": True}), context
    )
    assert flagged.violation_counts == {"ingested_after_decision": 1}
    assert flagged.ingestion_required


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_time", "2026-01-01T10:00:00"),
        ("available_time", "10:01"),
        ("ingested_time", "yesterday"),
        ("decision_time", [1767225600]),
        ("decision_time", ""),
    ],
)
def test_clocks_cannot_be_naive_or_text(
    context: OperationContext, field: str, value: object
) -> None:
    args: dict[str, object] = {
        "observations": [observation()],
        "decision_time": "2026-01-01T10:02:00Z",
    }
    if field == "decision_time":
        args["decision_time"] = value
    else:
        args["observations"] = [observation(**{field: value})]
    with pytest.raises(ValidationError):
        pit.Input.model_validate(args)


def test_offsets_normalize_to_the_same_instant(context: OperationContext) -> None:
    """Equivalent instants in different offsets are the same clock."""
    args = {
        "observations": [
            observation(
                event_time="2026-01-01T15:30:00+05:30",
                available_time="2026-01-01T10:00:00Z",
                ingested_time="2026-01-01T05:00:00-05:00",
            )
        ],
        "decision_time": "2026-01-01T10:00:00Z",
        "require_completed_events": True,
        "require_ingestion_by_decision": True,
    }
    assert pit.execute(pit.Input.model_validate(args), context).passed
    result = pit.execute(
        pit.Input.model_validate({**args, "decision_time": "2026-01-01T09:59:59Z"}),
        context,
    )
    assert result.violation_counts["available_after_decision"] == 1


def test_diagnostics_bounded_but_counts_exact(context: OperationContext) -> None:
    result = pit.execute(
        pit.Input.model_validate(
            {
                "observations": [
                    observation(
                        event_time="2026-01-01T11:00:00Z",
                        available_time="2026-01-01T11:01:00Z",
                        ingested_time="2026-01-01T10:00:00Z",
                    ),
                    observation(ingested_time=None),
                ],
                "decision_time": "2026-01-01T10:02:00Z",
                "require_completed_events": True,
                "require_ingestion_by_decision": True,
                "max_diagnostics": 1,
            }
        ),
        context,
    )
    assert result.invalid_rows == 2
    assert len(result.diagnostics) == 1
    assert result.omitted_diagnostics == result.violation_count - 1


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    request = pit.Input.model_validate(
        {
            "observations": [observation(available_time="2026-01-03T00:00:00Z")],
            "decision_time": "2026-01-01T10:02:00Z",
        }
    )
    first = pit.execute(request, context)
    second = pit.execute(request, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)


def test_epoch_clocks_resolve_to_utc_instants(context: OperationContext) -> None:
    """Epoch numbers are honest unambiguous UTC instants, not naive clocks."""
    args = {
        "observations": [
            observation(available_time=1767225690.5),  # 10:01:30.5Z
        ],
        "decision_time": 1767225720,  # 10:02:00Z
    }
    result = pit.execute(pit.Input.model_validate(args), context)
    assert result.passed
    result = pit.execute(
        pit.Input.model_validate({**args, "decision_time": 1767225600}),
        context,
    )
    assert result.violation_counts["available_after_decision"] == 1
