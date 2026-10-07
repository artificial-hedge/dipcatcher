"""SYNTHETIC adversarial probes for point-in-time select/join operations.

Every fixture is synthetic and deterministic; results are correctness checks,
never market evidence. These tests attack the clocks: observations published
after the decision must be invisible, events dated after the decision must not
leak even when published early, equal-clock conflicts must surface as
ambiguity or fail closed, and every tie must resolve deterministically.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from fx1.operations import (
    join_asof_observations as jao,
)
from fx1.operations import (
    resolve_security_identity as rsi,
)
from fx1.operations import (
    select_asof_revisions as sar,
)
from fx1.operations import (
    select_universe_membership as sumod,
)
from fx1.operations import (
    time_weighted_mean as twm,
)
from fx1.operations.base import OperationContext

T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def context(tmp_path: Path) -> OperationContext:
    return OperationContext(workspace_root=tmp_path)


def _hours(n: float) -> datetime:
    return T0 + timedelta(hours=n)


# ------------------------- join_asof_observations -----------------------------


def test_join_never_uses_rows_published_after_the_decision(context: OperationContext):
    """A row whose availability clock is later than the decision cannot match."""
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(10),
            available_time=_hours(200),  # published after the decision
            revision_id="late",
            source="s",
            values={"v": 1.0},
        ),
        jao.Observation(
            security_id="X",
            event_time=_hours(8),
            available_time=_hours(12),
            revision_id="early",
            source="s",
            values={"v": 0.5},
        ),
    ]
    out = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(100))],
        ),
        context,
    )
    assert out.joined[0].matched is True
    assert out.joined[0].observation_row_index == 1
    assert out.no_eligible_observation_count == 0


def test_join_future_event_published_early_does_not_leak(context: OperationContext):
    """An event dated after the decision is invisible even if already ingested."""
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(50),  # event happens after the decision
            available_time=_hours(10),
            revision_id="r1",
            source="s",
            values={"v": 9.0},
        )
    ]
    out = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(30))],
        ),
        context,
    )
    assert out.joined[0].matched is False
    assert out.joined[0].unmatched_reason == "no_eligible_observation"
    assert out.matched_count == 0


def test_join_activation_boundary_is_inclusive(context: OperationContext):
    """A row usable exactly at the decision clock (max(event, avail) == t) matches."""
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(20),
            revision_id="r1",
            source="s",
            values={"v": 1.0},
        )
    ]
    out = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(20))],
        ),
        context,
    )
    assert out.joined[0].matched is True
    assert out.joined[0].observation_row_index == 0


def test_join_same_clock_ties_pick_greatest_revision_then_earliest_row(
    context: OperationContext,
):
    """Identical (event, avail) clocks order by revision_id then input index.

    Re-publishing the same facts under a new revision is a legitimate tie;
    the input validator only rejects same-clock rows whose facts differ.
    """
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(10),
            available_time=_hours(12),
            revision_id="r009",
            source="s",
            values={"v": 1.0},
        ),
        jao.Observation(
            security_id="X",
            event_time=_hours(10),
            available_time=_hours(12),
            revision_id="r010",
            source="s",
            values={"v": 1.0},
        ),
        jao.Observation(
            security_id="X",
            event_time=_hours(10),
            available_time=_hours(12),
            revision_id="r009",
            source="s",
            values={"v": 1.0},
        ),
    ]
    out = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(50))],
        ),
        context,
    )
    assert out.joined[0].observation_row_index == 1  # greatest revision wins


def test_join_output_order_is_input_query_order_not_sweep_order(
    context: OperationContext,
):
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(1),
            available_time=_hours(2),
            revision_id="r1",
            source="s",
            values={"v": 1.0},
        )
    ]
    queries = [
        jao.Query(security_id="X", decision_time=_hours(100)),
        jao.Query(security_id="X", decision_time=_hours(3)),
        jao.Query(security_id="Z", decision_time=_hours(2)),
    ]
    out = jao.execute(jao.Input(observations=rows, queries=queries), context)
    assert [j.query_row_index for j in out.joined] == [0, 1, 2]
    assert out.joined[0].matched and out.joined[0].event_age_seconds == pytest.approx(99 * 3600)
    assert out.joined[1].matched
    assert out.joined[2].matched is False


def test_join_lookback_boundary_uses_event_age(context: OperationContext):
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(10),
            available_time=_hours(90),
            revision_id="r1",
            source="s",
            values={"v": 1.0},
        )
    ]
    # event age = 90h, inside a 90h limit and outside an 89h limit
    inside = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(100))],
            max_event_age_seconds=90 * 3600.0,
        ),
        context,
    )
    outside = jao.execute(
        jao.Input(
            observations=rows,
            queries=[jao.Query(security_id="X", decision_time=_hours(100))],
            max_event_age_seconds=89 * 3600.0,
        ),
        context,
    )
    assert inside.joined[0].matched
    assert outside.joined[0].matched is False
    assert outside.joined[0].unmatched_reason == "outside_lookback"
    assert outside.joined[0].event_age_seconds == pytest.approx(90 * 3600.0)
    assert outside.joined[0].observation_row_index is None


def test_join_equal_clock_conflicting_values_fail_closed() -> None:
    with pytest.raises(ValidationError, match="conflicting source or values"):
        jao.Input(
            observations=[
                {
                    "security_id": "X",
                    "event_time": "2026-01-01T05:00:00Z",
                    "available_time": "2026-01-01T06:00:00Z",
                    "revision_id": "a",
                    "source": "s",
                    "values": {"v": 1.0},
                },
                {
                    "security_id": "X",
                    "event_time": "2026-01-01T05:00:00Z",
                    "available_time": "2026-01-01T06:00:00Z",
                    "revision_id": "b",
                    "source": "s",
                    "values": {"v": 2.0},
                },
            ],
            queries=[{"security_id": "X", "decision_time": "2026-01-02T00:00:00Z"}],
        )


def test_join_naive_clock_and_nonfinite_value_rejected() -> None:
    with pytest.raises(ValidationError):
        jao.Query.model_validate({"security_id": "X", "decision_time": "2026-01-02T00:00:00"})
    with pytest.raises(ValidationError):
        jao.Observation.model_validate(
            {
                "security_id": "X",
                "event_time": "2026-01-01T00:00:00Z",
                "available_time": "2026-01-01T00:00:00Z",
                "revision_id": "a",
                "source": "s",
                "values": {"v": float("nan")},
            }
        )


def test_join_global_counts_cover_queries_outside_the_page(context: OperationContext):
    rows = [
        jao.Observation(
            security_id="X",
            event_time=_hours(1),
            available_time=_hours(2),
            revision_id="r",
            source="s",
            values={"v": 1.0},
        )
    ]
    queries = [jao.Query(security_id="X", decision_time=_hours(10 + i)) for i in range(5)]
    out = jao.execute(jao.Input(observations=rows, queries=queries, offset=2, limit=2), context)
    assert out.matched_count == 5
    assert [j.query_row_index for j in out.joined] == [2, 3]
    assert out.next_offset == 4


# ------------------------- select_asof_revisions ------------------------------


def test_revisions_after_the_decision_are_invisible(context: OperationContext):
    rows = [
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(10),
            revision_id="old",
            source="s",
            values={"v": 1.0},
        ),
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(50),
            revision_id="new",
            source="s",
            values={"v": 2.0},
        ),
    ]
    out = sar.execute(sar.Input(records=rows, decision_time=_hours(20)), context)
    assert out.unavailable_rows == 1
    assert out.selected[0].input_row_index == 0
    assert out.selected[0].record.values["v"] == 1.0
    assert out.superseded_or_duplicate_rows == 0

    later = sar.execute(sar.Input(records=rows, decision_time=_hours(60)), context)
    assert later.selected[0].input_row_index == 1  # newer vintage now observable
    assert later.superseded_or_duplicate_rows == 1


def test_revision_label_tie_break_is_lexicographic_not_numeric(
    context: OperationContext,
):
    """'r9' beats 'r10' lexicographically — the policy is explicit string order."""
    rows = [
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(10),
            revision_id="r10",
            source="s",
            values={"v": 1.0},
        ),
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(10),
            revision_id="r9",
            source="s",
            values={"v": 1.0},
        ),
    ]
    out = sar.execute(sar.Input(records=rows, decision_time=_hours(20)), context)
    assert out.selected[0].input_row_index == 1
    assert out.selected[0].eligible_vintage_rows == 2


def test_revision_full_tie_picks_earliest_input_row(context: OperationContext):
    rows = [
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(10),
            revision_id="r1",
            source="s",
            values={"v": 7.0},
        ),
        sar.Record(
            security_id="X",
            event_time=_hours(5),
            available_time=_hours(10),
            revision_id="r1",
            source="s",
            values={"v": 7.0},
        ),
    ]
    out = sar.execute(sar.Input(records=rows, decision_time=_hours(20)), context)
    assert out.selected[0].input_row_index == 0
    assert out.superseded_or_duplicate_rows == 1


def test_announced_future_event_allowed_unless_flagged(context: OperationContext):
    rows = [
        sar.Record(
            security_id="X",
            event_time=_hours(60),
            available_time=_hours(5),
            revision_id="r1",
            source="s",
            values={"v": 1.0},
        )
    ]
    allowed = sar.execute(sar.Input(records=rows, decision_time=_hours(10)), context)
    blocked = sar.execute(
        sar.Input(records=rows, decision_time=_hours(10), require_event_by_decision=True),
        context,
    )
    assert allowed.selected_event_count == 1
    assert blocked.selected_event_count == 0
    assert blocked.future_event_rows_excluded == 1
    assert blocked.eligible_rows == 0


def test_selection_output_is_sorted_security_then_event(context: OperationContext):
    rows = [
        sar.Record(
            security_id="B",
            event_time=_hours(1),
            available_time=_hours(2),
            revision_id="r",
            source="s",
            values={"v": 1.0},
        ),
        sar.Record(
            security_id="A",
            event_time=_hours(9),
            available_time=_hours(2),
            revision_id="r",
            source="s",
            values={"v": 1.0},
        ),
        sar.Record(
            security_id="A",
            event_time=_hours(3),
            available_time=_hours(2),
            revision_id="r",
            source="s",
            values={"v": 1.0},
        ),
    ]
    out = sar.execute(sar.Input(records=rows, decision_time=_hours(10)), context)
    keys = [(s.record.security_id, s.record.event_time) for s in out.selected]
    assert keys == sorted(keys)
    assert out.ordering == "security_id_then_event_time_ascending"


def test_revision_seed_driven_crosscheck_against_naive_scan(context: OperationContext):
    rng = random.Random(20261007)
    rows = []
    used: set[tuple[str, datetime, datetime]] = set()
    for i in range(60):
        while True:
            sec = rng.choice(["A", "B", "C"])
            ev = _hours(rng.randint(0, 40))
            av = ev + timedelta(hours=rng.randint(0, 15))
            if (sec, ev, av) not in used:
                used.add((sec, ev, av))
                break
        rows.append(
            sar.Record(
                security_id=sec,
                event_time=ev,
                available_time=av,
                revision_id=f"r{rng.randint(1, 7)}",
                source="s",
                values={"v": float(i)},
            )
        )
    decision = _hours(30)
    out = sar.execute(sar.Input(records=rows, decision_time=decision), context)
    naive: dict[tuple[str, datetime], tuple[datetime, str, int]] = {}
    for i, r in enumerate(rows):
        if r.available_time > decision:
            continue
        key = (r.security_id, r.event_time)
        priority = (r.available_time, r.revision_id, -i)
        if key not in naive or priority > naive[key]:
            naive[key] = priority
    assert out.selected_event_count == len(naive)
    got = {(s.record.security_id, s.record.event_time): s.input_row_index for s in out.selected}
    for key, (_, _, idx) in ((k, v) for k, v in naive.items()):
        assert got[key] == -idx


# ------------------------ resolve_security_identity ---------------------------


def _mapping(
    mapping_id: str,
    security_id: str,
    available_hours: float,
    *,
    valid_from: float = 0,
    valid_to: float | None = None,
    revision_id: str = "r1",
    ticker: str = "AAA",
    exchange: str = "XN",
    source: str = "s",
) -> rsi.Mapping:
    """Build a Mapping; valid_from/valid_to are absolute hours after T0."""
    return rsi.Mapping(
        mapping_id=mapping_id,
        security_id=security_id,
        ticker=ticker,
        exchange=exchange,
        valid_from=_hours(valid_from),
        valid_to=None if valid_to is None else _hours(valid_to),
        available_time=_hours(available_hours),
        revision_id=revision_id,
        source=source,
    )


def test_identity_latest_available_vintage_replaces_older(context: OperationContext):
    """The newest vintage at the decision wins; an older name is superseded."""
    rows = [
        _mapping("m1", "OLD", 10),
        _mapping("m1", "NEW", 30),
    ]
    query = rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(20))
    out = rsi.execute(rsi.Input(mappings=rows, queries=[query]), context)
    assert out.resolutions[0].status == "resolved"
    assert out.resolutions[0].security_id == "OLD"

    late = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(40))],
        ),
        context,
    )
    assert late.resolutions[0].security_id == "NEW"


def test_identity_conflicting_same_clock_records_are_ambiguous(
    context: OperationContext,
):
    rows = [
        _mapping("m1", "S1", 10, revision_id="r1"),
        _mapping("m1", "S2", 10, revision_id="r2"),
    ]
    out = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(20))],
        ),
        context,
    )
    r = out.resolutions[0]
    assert r.status == "ambiguous"
    assert r.conflicting_mapping_ids == 1
    assert r.candidate_security_count == 2
    assert r.security_id is None


def test_identity_multiple_ids_agreeing_on_one_security_resolve(
    context: OperationContext,
):
    rows = [
        _mapping("m1", "SAME", 10),
        _mapping("m2", "SAME", 15, revision_id="r2"),
    ]
    out = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(20))],
        ),
        context,
    )
    r = out.resolutions[0]
    assert r.status == "resolved"
    assert r.security_id == "SAME"
    assert r.selected_mapping_row_index == 1  # latest availability wins
    assert r.candidate_mapping_rows == 2


def test_identity_effective_interval_half_open(context: OperationContext):
    rows = [_mapping("m1", "S1", 5, valid_from=10, valid_to=20)]
    at_from = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(10))],
        ),
        context,
    )
    at_to = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(20))],
        ),
        context,
    )
    assert at_from.resolutions[0].status == "resolved"
    assert at_to.resolutions[0].status == "missing"
    assert at_to.resolutions[0].candidate_mapping_rows == 0


def test_identity_no_vintage_before_decision_is_missing_not_confused(
    context: OperationContext,
):
    rows = [_mapping("m1", "S1", 50)]
    out = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(10))],
        ),
        context,
    )
    assert out.resolutions[0].status == "missing"
    assert out.missing_count == 1 and out.ambiguous_count == 0


def test_identity_case_sensitive_matching_no_normalization(context: OperationContext):
    rows = [_mapping("m1", "S1", 5, ticker="aaa")]
    out = rsi.execute(
        rsi.Input(
            mappings=rows,
            queries=[rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(10))],
        ),
        context,
    )
    assert out.resolutions[0].status == "missing"


def test_identity_work_budget_fails_closed(context: OperationContext):
    rows = [
        _mapping("m1", "S1", 5),
        _mapping("m2", "S2", 5),
        _mapping("m3", "S3", 5),
    ]
    queries = [rsi.Query(ticker="AAA", exchange="XN", decision_time=_hours(10))]
    with pytest.raises(ValueError, match="work_budget"):
        rsi.execute(rsi.Input(mappings=rows, queries=queries, work_budget=2), context)


# ------------------------ select_universe_membership --------------------------


def _membership(
    membership_id: str,
    security_id: str,
    state: Literal["included", "excluded"],
    effective_from: float,
    effective_to: float | None,
    available_hours: float,
    revision_id: str = "r1",
    source: str = "s",
) -> sumod.Membership:
    """Build a Membership; effective_from/effective_to are absolute hours after T0."""
    return sumod.Membership(
        membership_id=membership_id,
        security_id=security_id,
        state=state,
        effective_from=_hours(effective_from),
        effective_to=None if effective_to is None else _hours(effective_to),
        available_time=_hours(available_hours),
        revision_id=revision_id,
        source=source,
    )


def test_membership_expired_exclusion_reveals_older_inclusion(
    context: OperationContext,
):
    """An expired temporary exclusion must not shadow a still-active inclusion."""
    rows = [
        _membership("u_in", "S1", "included", 0, None, 5),
        _membership("u_out", "S1", "excluded", 10, 20, 8),
    ]
    # exclusion active on [10,20): decision 15 excluded, decision 25 re-included
    during = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(15)), context)
    after = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(25)), context)
    assert during.selections[0].status == "excluded"
    assert after.selections[0].status == "included"
    assert after.selections[0].selected_row_index == 0


def test_membership_same_clock_conflicting_states_are_ambiguous(
    context: OperationContext,
):
    rows = [
        _membership("u1", "S1", "included", 5, None, 10, revision_id="r1"),
        _membership("u1", "S1", "excluded", 5, None, 10, revision_id="r2"),
    ]
    out = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(20)), context)
    sel = out.selections[0]
    assert sel.status == "ambiguous"
    assert sel.conflicting_membership_ids == 1
    assert sel.selected_row_index is None
    assert set(sel.supporting_row_indices) == {0, 1}


def test_membership_unavailable_rows_never_participate(context: OperationContext):
    rows = [
        _membership("u1", "S1", "included", 0, None, 100),  # arrives later
        _membership("u1", "S1", "excluded", 0, None, 10),
    ]
    out = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(50)), context)
    assert out.unavailable_input_rows == 1
    assert out.selections[0].status == "excluded"


def test_membership_all_unavailable_security_is_missing(context: OperationContext):
    rows = [_membership("u1", "S1", "included", 0, None, 100)]
    out = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(50)), context)
    assert out.selections[0].status == "missing"
    assert out.security_count == 1


def test_membership_distinct_id_same_effective_from_conflict_is_ambiguous(
    context: OperationContext,
):
    rows = [
        _membership("u_in", "S1", "included", 5, None, 10),
        _membership("u_out", "S1", "excluded", 5, None, 10),
    ]
    out = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(20)), context)
    assert out.selections[0].status == "ambiguous"


def test_membership_latest_effective_from_wins(context: OperationContext):
    rows = [
        _membership("u1", "S1", "excluded", 5, None, 10),
        _membership("u2", "S1", "included", 15, None, 10),
    ]
    out = sumod.execute(sumod.Input(memberships=rows, decision_time=_hours(20)), context)
    sel = out.selections[0]
    assert sel.status == "included"
    assert sel.selected_row_index == 1


def test_membership_explicit_security_list_paginates_sorted(
    context: OperationContext,
):
    rows = [
        _membership("u1", "B", "included", 0, None, 5),
        _membership("u2", "A", "excluded", 0, None, 5),
    ]
    out = sumod.execute(
        sumod.Input(
            memberships=rows,
            decision_time=_hours(10),
            security_ids=["C", "A", "B"],
            offset=1,
            limit=1,
        ),
        context,
    )
    assert out.security_count == 3
    assert [s.security_id for s in out.selections] == ["B"]
    assert out.next_offset == 2


# --------------------------- time_weighted_mean -------------------------------


def _obs(event: float, available: float, value: float) -> twm.Observation:
    return twm.Observation(
        event_time=T0 + timedelta(seconds=event),
        available_time=T0 + timedelta(seconds=available),
        value=value,
    )


def test_twm_missing_initial_coverage_fails_closed(context: OperationContext) -> None:
    # row activates at 360, inside window [300,400) -> nothing held at window start
    obs = [_obs(350.0, 360.0, 1.0)]
    with pytest.raises(ValueError, match="no observable initial value"):
        twm.execute(
            twm.Input(
                observations=obs,
                query_time=T0 + timedelta(seconds=400),
                lookback_seconds=100,
            ),
            context,
        )


def test_twm_late_arriving_old_event_cannot_overwrite(context: OperationContext):
    """An old event published late is observable but cannot rewrite history."""
    obs = [
        _obs(50.0, 60.0, 1.0),  # holds from t=60
        _obs(20.0, 200.0, 9.0),  # older event arriving at t=200 — must not displace
    ]
    out = twm.execute(
        twm.Input(
            observations=obs,
            query_time=T0 + timedelta(seconds=400),
            lookback_seconds=300,
        ),
        context,
    )
    # window [100,400]: 1.0 holds the whole window -> mean 1.0
    assert out.time_weighted_mean == pytest.approx(1.0)


def test_twm_activation_uses_max_of_event_and_availability(
    context: OperationContext,
):
    obs = [_obs(50.0, 150.0, 2.0)]  # event early, available late
    out = twm.execute(
        twm.Input(
            observations=obs,
            query_time=T0 + timedelta(seconds=400),
            lookback_seconds=200,
        ),
        context,
    )
    # window [200,400]: activation at 150 <= start -> holds whole window
    assert out.time_weighted_mean == pytest.approx(2.0)
    assert out.contributing_observation_indices == [0]


def test_twm_row_activated_at_query_clock_gets_zero_weight(context: OperationContext):
    obs = [
        _obs(10.0, 20.0, 1.0),
        _obs(30.0, 400.0, 9.0),  # activates exactly at query clock
    ]
    out = twm.execute(
        twm.Input(
            observations=obs,
            query_time=T0 + timedelta(seconds=400),
            lookback_seconds=200,
        ),
        context,
    )
    assert out.time_weighted_mean == pytest.approx(1.0)
    assert out.rows_not_activated_before_query == 1


def test_twm_hand_computed_integration(context: OperationContext):
    obs = [
        _obs(0.0, 0.0, 2.0),  # holds [start=100,250)
        _obs(200.0, 250.0, 8.0),  # activates 250, holds (250,400]
    ]
    out = twm.execute(
        twm.Input(
            observations=obs,
            query_time=T0 + timedelta(seconds=400),
            lookback_seconds=300,
        ),
        context,
    )
    # 2.0 for 150s + 8.0 for 150s -> (300+1200)/300 = 5.0
    assert out.time_weighted_mean == pytest.approx(5.0)
    assert out.integral_value_seconds == pytest.approx(1500.0)
    assert out.positive_duration_segments == 2
    assert out.coverage_fraction == 1.0
    assert out.max_contributor_available_time == T0 + timedelta(seconds=250)


def test_twm_duplicate_event_times_fail_closed() -> None:
    obs = [_obs(0.0, 0.0, 1.0), _obs(0.0, 5.0, 2.0)]
    with pytest.raises(ValidationError, match="event_time must be unique"):
        twm.Input(
            observations=obs,
            query_time=T0 + timedelta(seconds=400),
            lookback_seconds=100,
        )


def test_twm_seed_driven_crosscheck(context: OperationContext):
    rng = random.Random(99)
    for _ in range(60):
        n = rng.randint(2, 8)
        events = rng.sample(range(0, 250), n)
        obs = [_obs(float(e), float(rng.randint(0, 350)), rng.uniform(-10, 10)) for e in events]
        query = T0 + timedelta(seconds=400)
        lb = rng.randint(100, 300)
        start = query - timedelta(seconds=lb)
        act = [
            (max(o.event_time, o.available_time), o.event_time, i, o)
            for i, o in enumerate(obs)
            if max(o.event_time, o.available_time) < query
        ]
        if not any(a <= start for a, _, _, _ in act):
            continue
        out = twm.execute(
            twm.Input(observations=obs, query_time=query, lookback_seconds=lb), context
        )
        act.sort(key=lambda t: (t[0], t[1]))
        held = max((it for it in act if it[0] <= start), key=lambda t: t[1])
        cur_ev, cur_v = held[1], held[3].value
        integral = 0.0
        seg = start
        for a, e, _i, o in act:
            if a <= start or e <= cur_ev:
                continue
            integral += cur_v * (a - seg).total_seconds()
            seg, cur_ev, cur_v = a, e, o.value
        integral += cur_v * (query - seg).total_seconds()
        assert out.integral_value_seconds == pytest.approx(integral)
        assert out.time_weighted_mean == pytest.approx(integral / lb)
