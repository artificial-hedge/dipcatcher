"""Unit tests for ``fx1.operations.time_weighted_mean``.

The operation integrates a *held* value over an explicit time window. The
contract it must keep is the one that matters for a live decision path:

* an observation changes the held state only after **both** its event time and
  its availability time have passed;
* a late-arriving old event cannot overwrite a newer event or rewrite history;
* values are held on ``(activation_time, next_activation_time]``;
* missing initial coverage **fails closed** instead of backfilling.

These are all failure modes that would silently produce a plausible-looking but
wrong average, so they are pinned explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fx1.operations.base import OperationContext
from fx1.operations.time_weighted_mean import Input, Observation, execute

CONTEXT = OperationContext(workspace_root=Path(__file__).resolve())

QUERY = datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 2, hour, minute, 0, tzinfo=UTC)


def _obs(
    event: datetime,
    available: datetime,
    value: float,
) -> Observation:
    return Observation(event_time=event, available_time=available, value=value)


def _run(observations: list[Observation], lookback: int = 3600):
    return execute(
        Input(observations=observations, query_time=QUERY, lookback_seconds=lookback),
        CONTEXT,
    )


# --------------------------------------------------------------------------- #
# Single-segment integration
# --------------------------------------------------------------------------- #


def test_constant_value_integrates_to_itself() -> None:
    """A value held for the whole window averages to that value."""
    out = _run([_obs(_at(10), _at(10), 2.0)])
    assert out.time_weighted_mean == pytest.approx(2.0)
    assert out.integral_value_seconds == pytest.approx(2.0 * 3600)
    assert out.covered_seconds == pytest.approx(3600.0)
    assert out.requested_seconds == 3600
    assert out.coverage_fraction == pytest.approx(1.0)
    assert out.window_start == _at(11)
    assert out.query_time == QUERY
    assert out.positive_duration_segments == 1
    assert out.initial_observation_index == 0
    assert out.contributing_observation_indices == [0]
    assert out.rows_not_activated_before_query == 0


def test_two_segments_are_weighted_by_duration() -> None:
    """1.0 for the first half-hour, 3.0 for the second half-hour."""
    out = _run(
        [
            _obs(_at(10), _at(10), 1.0),
            _obs(_at(11, 30), _at(11, 30), 3.0),
        ]
    )
    # 1.0 * 1800 + 3.0 * 1800 = 7200 over 3600 seconds.
    assert out.time_weighted_mean == pytest.approx(2.0)
    assert out.integral_value_seconds == pytest.approx(7200.0)
    assert out.positive_duration_segments == 2
    assert out.contributing_observation_indices == [0, 1]
    assert out.initial_observation_index == 0


# --------------------------------------------------------------------------- #
# Availability governs activation
# --------------------------------------------------------------------------- #


def test_late_arrival_does_not_rewrite_past_intervals() -> None:
    """A value that only becomes observable at 11:45 applies from 11:45 onward."""
    out = _run(
        [
            _obs(_at(10), _at(10), 1.0),
            _obs(_at(11, 30), _at(11, 45), 3.0),
        ]
    )
    # 1.0 * 2700 + 3.0 * 900 = 5400 over 3600 seconds.
    assert out.time_weighted_mean == pytest.approx(1.5)
    assert out.integral_value_seconds == pytest.approx(5400.0)
    assert out.positive_duration_segments == 2


def test_old_event_arriving_late_cannot_overwrite_a_newer_event() -> None:
    """The newer event (11:00) wins; the late old event (10:30) never applies."""
    out = _run(
        [
            _obs(_at(11), _at(10), 5.0),
            _obs(_at(10, 30), _at(11, 30), 100.0),
        ]
    )
    assert out.time_weighted_mean == pytest.approx(5.0)
    assert out.contributing_observation_indices == [0]
    assert out.positive_duration_segments == 1


def test_row_activated_exactly_at_the_query_clock_contributes_no_weight() -> None:
    """Activation at the query instant is not observable yet — it is excluded."""
    out = _run(
        [
            _obs(_at(10), _at(10), 1.0),
            _obs(QUERY, QUERY, 99.0),
        ]
    )
    assert out.time_weighted_mean == pytest.approx(1.0)
    assert out.rows_not_activated_before_query == 1
    assert out.contributing_observation_indices == [0]


# --------------------------------------------------------------------------- #
# Fail-closed behaviour
# --------------------------------------------------------------------------- #


def test_missing_initial_coverage_fails_closed() -> None:
    """No observation observable at the window start is an error, not a backfill."""
    with pytest.raises(ValueError, match="no observable initial value"):
        _run([_obs(_at(11, 30), _at(11, 30), 1.0)])


def test_duplicate_event_times_are_rejected() -> None:
    """Ambiguous revisions are unsupported and must not be silently merged."""
    with pytest.raises(ValueError, match="event_time must be unique"):
        _run(
            [
                _obs(_at(10), _at(10), 1.0),
                _obs(_at(10), _at(10), 2.0),
            ]
        )


def test_naive_datetimes_are_rejected() -> None:
    """An explicit timezone offset is required — an implicit clock is ambiguous."""
    with pytest.raises(ValueError, match="timezone offset"):
        _run([_obs(datetime(2024, 1, 2, 10), datetime(2024, 1, 2, 10), 1.0)])


def test_empty_observation_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        Input(observations=[], query_time=QUERY, lookback_seconds=3600)


def test_max_contributor_availability_is_reported() -> None:
    """The output records the latest availability that fed the integral."""
    out = _run(
        [
            _obs(_at(10), _at(10, 5), 1.0),
            _obs(_at(11, 30), _at(11, 40), 3.0),
        ]
    )
    assert out.max_contributor_available_time == _at(11, 40)
    assert out.hold_policy == "latest_observable_event_without_staleness_cutoff"
