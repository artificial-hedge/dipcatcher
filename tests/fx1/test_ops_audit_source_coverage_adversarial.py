"""Adversarial probes for source/security coverage at a decision clock.

Future availability must never satisfy coverage (point-in-time leakage);
future events are excluded separately; staleness is measured on the caller's
selected clock with an exact boundary; and the latest-record selection must
follow the documented priority, never first-seen order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fx1.operations import (
    audit_source_coverage as cov,
)
from fx1.operations.base import OperationContext


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def observation(source: str = "s", security: str = "a", **updates: object) -> dict[str, object]:
    return {
        "source": source,
        "security_id": security,
        "event_time": "2026-01-01T10:00:00Z",
        "available_time": "2026-01-01T10:00:00Z",
        **updates,
    }


def run(args: dict[str, object], context: OperationContext) -> cov.Output:
    return cov.execute(cov.Input.model_validate(args), context)


def test_unavailable_records_cannot_cover_a_pair(context: OperationContext) -> None:
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [
                observation(available_time="2026-01-01T10:03:00Z"),
                observation(available_time="2026-01-01T10:00:30Z"),
            ],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    assert result.unavailable_records == 1
    assert result.eligible_record_count == 1
    assert result.covered_expected_pairs == 1


def test_all_unavailable_pair_reports_missing_not_covered(
    context: OperationContext,
) -> None:
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [observation(available_time="2026-01-01T11:00:00Z")],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    assert result.missing_expected_pairs == 1
    assert result.assessment == "incomplete"
    assert result.complete_expected_coverage is False
    assert result.expected_pairs[0].selected_row_index is None


def test_future_event_is_excluded_even_when_available(context: OperationContext) -> None:
    """An announced-but-incomplete event cannot satisfy coverage."""
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [
                observation(
                    event_time="2026-01-01T12:00:00Z",
                    available_time="2026-01-01T10:00:00Z",
                )
            ],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    assert result.future_event_records_excluded == 1
    assert result.missing_expected_pairs == 1


def test_staleness_boundary_is_exact(context: OperationContext) -> None:
    args: dict[str, object] = {
        "expected_pairs": [{"source": "s", "security_id": "a"}],
        "observations": [observation()],
        "decision_time": "2026-01-01T10:01:00Z",
    }
    at_limit = run({**args, "max_age_seconds": 60.0}, context)
    assert at_limit.expected_pairs[0].status == "covered"
    over = run({**args, "max_age_seconds": 59.999}, context)
    assert over.expected_pairs[0].status == "stale"
    assert over.stale_expected_pairs == 1
    assert over.assessment == "incomplete"


def test_staleness_clock_switches_the_measured_age(context: OperationContext) -> None:
    args: dict[str, object] = {
        "expected_pairs": [{"source": "s", "security_id": "a"}],
        "observations": [
            observation(
                event_time="2026-01-01T09:00:00Z",
                available_time="2026-01-01T10:00:00Z",
            )
        ],
        "decision_time": "2026-01-01T10:01:00Z",
        "max_age_seconds": 120.0,
    }
    event_clock = run({**args, "staleness_clock": "event_time"}, context)
    assert event_clock.expected_pairs[0].status == "stale"
    avail_clock = run({**args, "staleness_clock": "available_time"}, context)
    assert avail_clock.expected_pairs[0].status == "covered"


def test_selection_prefers_latest_event_then_availability_then_earliest_row(
    context: OperationContext,
) -> None:
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [
                observation(
                    event_time="2026-01-01T10:00:00Z", available_time="2026-01-01T10:00:00Z"
                ),
                observation(
                    event_time="2026-01-01T10:00:00Z", available_time="2026-01-01T10:00:30Z"
                ),
                observation(
                    event_time="2026-01-01T09:59:00Z", available_time="2026-01-01T10:00:59Z"
                ),
                observation(
                    event_time="2026-01-01T10:00:00Z", available_time="2026-01-01T10:00:30Z"
                ),
            ],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    pair = result.expected_pairs[0]
    assert pair.selected_row_index == 1
    assert pair.eligible_records == 4


def test_unexpected_pairs_are_reported_but_never_subtract_coverage(
    context: OperationContext,
) -> None:
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [
                observation(),
                observation(source="rogue", security="z"),
            ],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    assert result.assessment == "covered"
    assert result.unexpected_eligible_pair_count == 1
    assert result.eligible_unexpected_records == 1
    assert result.unexpected_eligible_pairs[0].source == "rogue"


def test_unexpected_examples_are_bounded(context: OperationContext) -> None:
    result = run(
        {
            "expected_pairs": [{"source": "s", "security_id": "a"}],
            "observations": [observation(security=f"u{i}") for i in range(5)] + [observation()],
            "decision_time": "2026-01-01T10:01:00Z",
            "max_unexpected_examples": 2,
        },
        context,
    )
    assert result.unexpected_eligible_pair_count == 5
    assert len(result.unexpected_eligible_pairs) == 2
    assert result.omitted_unexpected_examples == 3


def test_no_expected_pairs_is_unassessed_not_green(context: OperationContext) -> None:
    result = run(
        {
            "expected_pairs": [],
            "observations": [observation()],
            "decision_time": "2026-01-01T10:01:00Z",
        },
        context,
    )
    assert result.assessment == "no_expected_pairs"
    assert result.complete_expected_coverage is None


@pytest.mark.parametrize(
    "clock",
    [
        "2026-01-01T10:01:00",
        "1767225660",
        "10:01",
        "x" * 65,
        1767225660,
        None,
    ],
)
def test_clocks_cannot_be_naive_epochs_or_oversized(
    context: OperationContext, clock: object
) -> None:
    with pytest.raises(ValidationError):
        cov.Input.model_validate(
            {
                "expected_pairs": [{"source": "s", "security_id": "a"}],
                "observations": [observation(available_time=clock)],
                "decision_time": "2026-01-01T10:01:00Z",
            }
        )


def test_duplicate_expected_pairs_rejected() -> None:
    with pytest.raises(ValidationError):
        cov.Input.model_validate(
            {
                "expected_pairs": [
                    {"source": "s", "security_id": "a"},
                    {"source": "s", "security_id": "a"},
                ],
                "observations": [observation()],
                "decision_time": "2026-01-01T10:01:00Z",
            }
        )


def test_paging_is_bounded_and_deterministic(context: OperationContext) -> None:
    expected = [{"source": "s", "security_id": f"p{i:02d}"} for i in range(7)]
    args: dict[str, object] = {
        "expected_pairs": expected,
        "observations": [observation(security=f"p{i:02d}") for i in range(7)],
        "decision_time": "2026-01-01T10:01:00Z",
        "limit": 3,
    }
    page0 = run({**args, "offset": 0}, context)
    page1 = run({**args, "offset": 3}, context)
    page2 = run({**args, "offset": 6}, context)
    assert [p.security_id for p in page0.expected_pairs] == ["p00", "p01", "p02"]
    assert page0.next_offset == 3
    assert [p.security_id for p in page1.expected_pairs] == ["p03", "p04", "p05"]
    assert page1.next_offset == 6
    assert [p.security_id for p in page2.expected_pairs] == ["p06"]
    assert page2.next_offset is None


def test_results_deterministic_and_json_finite(context: OperationContext) -> None:
    args = {
        "expected_pairs": [{"source": "s", "security_id": "a"}],
        "observations": [observation(), observation(security="u")],
        "decision_time": "2026-01-01T10:01:00Z",
    }
    first = run(args, context)
    second = run(args, context)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    json.dumps(first.model_dump(mode="json"), allow_nan=False)
