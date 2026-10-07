"""Adversarial probes for the book/core-sim audit lane.

Each test pins one defect class fixed in this lane on seeded SYNTHETIC
inputs (no real tape):

- LOBSTER event-0 double-apply: orderbook row i is the book state AFTER
  message i, so seeding row 0 must not re-apply event 0
  (``lobster.validate_reconstruction`` / ``tape_measurements``,
  ``spread_dynamics``, ``queue_jump``, ``quote_place``,
  ``sim_real_ledger``, ``lob_exec`` warmup).
- EXECUTION-hidden blindness: LOBSTER type-5 prints are market-order
  arrivals — sign streams, MO ledgers, exec-time series and fill
  clustering must count them (``lobster.tape_measurements``,
  ``sim_real_ledger``, ``cancel_cluster``, ``metaorder_detect``,
  ``spread_response``).
- Seeded-book sweep: ids already in ``sim._orders`` at construction are
  seed liquidity, not submissions (``queue_jump``, ``quote_place``).
- Post-step touch: an event's classification belongs to the touch it
  saw BEFORE ``sim.step()`` (``cancel_gradient``, ``cancel_lead``,
  ``marketable_limit``).
- NaN sealed into receipts: payloads must serialize under strict JSON —
  non-finite stats emit as null, never literal ``NaN``
  (``queue_depletion``, ``excursion_geometry``).
- Receipt-convention drift: ``kind="<name>"`` / ``schema="<name>.v1"``,
  and unsealed member receipts contribute no claims (``book_depletion``,
  ``zone_map``).
- First-event state attribution: ``QueueTrajectory.q0`` is the pre-state
  of the first event, not a hardcoded ``q_max // 2`` (``queue_reactive``).
- Truncated depletion scan: an unrecovered depletion ends its own
  episode, not the scan (``lob_resilience``).
- Units honesty: ``sim.mid`` is price units; the drift contract is ticks
  (``stale_quote``).
- Warmup replay off-by-one: ``exec_on_tape`` replays events 1..start-1
  and resyncs each applied row (``lob_exec``).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quant_fund.microstructure import (
    cancel_gradient as cancel_gradient_mod,
)
from quant_fund.microstructure import (
    cancel_lead as cancel_lead_mod,
)
from quant_fund.microstructure import (
    marketable_limit as marketable_limit_mod,
)
from quant_fund.microstructure import (
    stale_quote as stale_quote_mod,
)
from quant_fund.microstructure.book_depletion import (
    BOOK_DEPLETION_KIND,
    BOOK_DEPLETION_SCHEMA,
)
from quant_fund.microstructure.cancel_cluster import lobster_cancel_cluster
from quant_fund.microstructure.cancel_gradient import sim_cancel_gradient
from quant_fund.microstructure.cancel_lead import sim_cancel_lead
from quant_fund.microstructure.excursion_geometry import excursion_geometry_bench
from quant_fund.microstructure.lob_exec import exec_on_tape
from quant_fund.microstructure.lob_resilience import _resilience_stats
from quant_fund.microstructure.lobster import (
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    LobsterBook,
    LobsterEvent,
    tape_measurements,
    validate_reconstruction,
)
from quant_fund.microstructure.marketable_limit import (
    _aggression_stats,
    sim_marketable_limit,
)
from quant_fund.microstructure.metaorder_detect import lobster_metaorders
from quant_fund.microstructure.queue_depletion import queue_depletion_bench
from quant_fund.microstructure.queue_jump import sim_queue_jump
from quant_fund.microstructure.queue_reactive import (
    QueueTrajectory,
    estimate_rates,
    simulate_queue,
)
from quant_fund.microstructure.quote_place import sim_placement
from quant_fund.microstructure.sim_real_ledger import measure_lobster
from quant_fund.microstructure.spread_response import lobster_spread_response
from quant_fund.microstructure.stale_quote import sim_stale_quote
from quant_fund.microstructure.zone_map import zone_map
from tests.unit.microstructure.test_sim_real_ledger import _ev, _write_pair


def _tape(tmp_path: Path, events: list[LobsterEvent]) -> tuple[Path, Path]:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    _write_pair(msg, ob, events, levels=4)
    return msg, ob


class _StubOrder:
    __slots__ = ("level", "side")

    def __init__(self, side: str, level: int) -> None:
        self.side = side
        self.level = level


class _StubSim:
    """Scripted stand-in for ``ZILobSimulator``.

    Script entries are ``(kind, post_orders, post_bid, post_ask)``: each
    ``step()`` replaces the order set and touch — the post-event state
    convention. ``mid``/``t``/``trades`` cover the lead/stale probes; an
    optional ``inject`` hook mutates the sim inside a step (e.g. records
    a fill).
    """

    def __init__(
        self,
        orders: dict[int, _StubOrder],
        bb: int | None,
        ba: int | None,
        script: list[tuple[str, dict[int, _StubOrder], int | None, int | None]],
        *,
        mid0: float | None = None,
        mid_step: float = 0.0,
        t_step: float = 0.1,
        inject: Callable[[_StubSim], None] | None = None,
    ) -> None:
        self._orders = dict(orders)
        self.best_bid_level = bb
        self.best_ask_level = ba
        self._script = list(script)
        self._mid = mid0
        self._mid_step = mid_step
        self._t_step = t_step
        self._inject = inject
        self.t = 0.0
        self.trades: list[SimpleNamespace] = []

    @property
    def mid(self) -> float | None:
        return self._mid

    def step(self) -> str:
        kind, orders, bb, ba = self._script.pop(0)
        self._orders = dict(orders)
        self.best_bid_level = bb
        self.best_ask_level = ba
        self.t += self._t_step
        if self._mid is not None:
            self._mid += self._mid_step
        if self._inject is not None:
            self._inject(self)
        return kind


# ---------------------------------------------------------------------------
# LOBSTER event-0 double-apply
# ---------------------------------------------------------------------------


def test_validate_reconstruction_no_event0_double_apply(tmp_path: Path) -> None:
    """On a consistent tape, every applied row must match — the event-0
    double-apply showed up as an immediate level-size divergence."""
    events = [
        _ev(0, SUBMISSION, 1, 100, 49000, 1),
        _ev(1, SUBMISSION, 2, 50, 50000, -1),
        _ev(2, SUBMISSION, 3, 25, 49100, 1),
        _ev(3, EXECUTION_HIDDEN, 3, 25, 49100, 1),
        _ev(4, SUBMISSION, 4, 30, 49200, 1),
    ]
    msg, ob = _tape(tmp_path, events)
    out = validate_reconstruction(msg, ob)
    assert out["n_events"] == 5  # all rows consumed (row 0 as the seed)
    assert out["n_compared"] == 3  # hidden/halt rows skip the book check
    assert out["n_match"] == 3
    assert out["n_resync_events"] == 0
    assert out["first_mismatch"] is None


# ---------------------------------------------------------------------------
# EXECUTION_HIDDEN inclusion in aggressor streams
# ---------------------------------------------------------------------------


def test_tape_measurements_counts_hidden_executions(tmp_path: Path) -> None:
    """The sign stream must cover the full aggressor tape, including
    hidden-liquidity fills — type-5 events are still MO prints."""
    events = [
        _ev(0, SUBMISSION, 1, 100, 49000, 1),
        _ev(1, SUBMISSION, 2, 50, 50000, -1),
        _ev(2, SUBMISSION, 3, 25, 49100, 1),
        _ev(3, EXECUTION_HIDDEN, 3, 25, 49100, 1),
        _ev(4, EXECUTION, 2, 10, 50000, -1),
    ]
    msg, ob = _tape(tmp_path, events)
    out = tape_measurements(msg, ob)
    assert out["n_executions"] == 2  # type 4 + type 5


def test_sim_real_ledger_counts_hidden_execs(tmp_path: Path) -> None:
    msg, ob = _tape(
        tmp_path,
        [
            _ev(0, SUBMISSION, 1, 100, 49000, 1),
            _ev(1, SUBMISSION, 2, 50, 50000, -1),
            _ev(2, EXECUTION_HIDDEN, 1, 25, 49000, 1),
            _ev(3, EXECUTION, 2, 10, 50000, -1),
            _ev(4, SUBMISSION, 4, 30, 49200, 1),
        ],
    )
    out = measure_lobster(msg, ob, tick_units=100.0)
    assert out["n_trades"] == 2
    assert out["mo_fraction"] == 2 / 5


def test_cancel_cluster_counts_hidden_execs(tmp_path: Path) -> None:
    _tape(
        tmp_path,
        [
            _ev(0, SUBMISSION, 1, 100, 49000, 1),
            _ev(1, SUBMISSION, 2, 50, 50000, -1),
            _ev(2, EXECUTION_HIDDEN, 1, 25, 49000, 1),
            _ev(3, DELETE, 1, 75, 49000, 1),
        ],
    )
    out = lobster_cancel_cluster(tmp_path)
    assert out["n_execs"] == 1  # the type-5 print is an exec timestamp


def test_metaorders_count_hidden_fills(tmp_path: Path) -> None:
    _tape(
        tmp_path,
        [
            _ev(0, SUBMISSION, 1, 100, 49000, 1),
            _ev(1, SUBMISSION, 2, 50, 50000, -1),
            _ev(2, EXECUTION_HIDDEN, 1, 25, 49000, 1),
            _ev(3, EXECUTION_HIDDEN, 1, 25, 49000, 1),
            _ev(4, EXECUTION, 1, 25, 49000, 1),
        ],
    )
    out = lobster_metaorders(tmp_path)
    assert out["n_execs"] == 3  # two hidden + one visible, same sign


def test_spread_response_includes_hidden_execs(tmp_path: Path) -> None:
    msg, ob = _tape(
        tmp_path,
        [
            _ev(0, SUBMISSION, 1, 100, 49000, 1),
            _ev(1, SUBMISSION, 2, 50, 50000, -1),
            _ev(2, EXECUTION_HIDDEN, 1, 25, 49000, 1),
            _ev(3, EXECUTION, 2, 10, 50000, -1),
        ],
    )
    out = lobster_spread_response(msg, ob)
    assert out["n_execs"] == 2


# ---------------------------------------------------------------------------
# Seeded-book sweep: construction-time orders are not submissions
# ---------------------------------------------------------------------------


def test_sim_queue_jump_does_not_sweep_seed_book() -> None:
    """After a single step at most one new order can exist; the seeded
    book (~30 orders) must not be counted as submissions."""
    out = sim_queue_jump(horizon=1, seed=7)
    assert out["n_submissions"] <= 1


def test_sim_placement_does_not_sweep_seed_book() -> None:
    out = sim_placement(horizon=1, seed=7)
    assert out["n_submissions"] <= 1


# ---------------------------------------------------------------------------
# lob_exec warmup replay: row 0 is the seed; replay starts at event 1
# ---------------------------------------------------------------------------


def test_exec_on_tape_replays_warmup_from_event_1() -> None:
    """Event 0 deepens the seeded touch. Applied twice it doubles the
    seeded depth; the child's walk then fills at the touch instead of
    walking a level — measurable as a zero liquidity cost."""
    events = [
        _ev(0, SUBMISSION, 1, 30, 50000, -1),  # seeds ask 30@50000
        _ev(1, SUBMISSION, 2, 40, 50100, -1),  # ask 40@50100
        _ev(2, SUBMISSION, 3, 40, 49000, 1),  # bid 40@49000
        _ev(3, SUBMISSION, 4, 10, 48000, 1),  # deep bid
        _ev(4, SUBMISSION, 5, 10, 47900, 1),
    ]
    book = LobsterBook()
    snapshots = []
    for ev in events:
        book.apply(ev)
        snapshots.append((book.top("ask", 4), book.top("bid", 4)))
    out = exec_on_tape(
        events,
        snapshots,
        start=3,
        parent_size=50,
        n_children=1,
        events_per_child=1,
        side="buy",
    )
    assert out is not None
    # Correct book: 30@50000 then 20@50100 -> walk = 20*100/50 = 40 (raw).
    assert out["fill_fraction"] == 1.0
    assert out["liquidity_cost_ticks"] == pytest.approx(0.4)
    assert out["is_total_ticks"] == pytest.approx(5.4)


# ---------------------------------------------------------------------------
# Post-step touch: classification belongs to the pre-event book
# ---------------------------------------------------------------------------


def _cancel_touch_script(
    n_cycles: int,
) -> tuple[dict[int, _StubOrder], int, int, list]:
    """Alternate: cancel the best bid (touch drops deep), then a buy at
    the old touch re-forms it. Every cancel is exactly at the pre-event
    touch (distance 0); a post-step read puts it at distance 1+."""
    sell = _StubOrder("sell", 10)
    deep = _StubOrder("buy", 1)
    orders = {1: _StubOrder("buy", 5), 2: sell, 3: deep}
    script = []
    oid = 10
    for _ in range(n_cycles):
        script.append(("cancel", {2: sell, 3: deep}, 1, 10))
        script.append(("limit", {2: sell, 3: deep, oid: _StubOrder("buy", 5)}, 5, 10))
        oid += 1
    return orders, 5, 10, script


def test_cancel_gradient_uses_pre_event_touch(monkeypatch: pytest.MonkeyPatch) -> None:
    orders, bb, ba, script = _cancel_touch_script(50)
    stub = _StubSim(orders, bb, ba, script)
    monkeypatch.setattr(cancel_gradient_mod, "ZILobSimulator", lambda *a, **k: stub)
    out = sim_cancel_gradient(horizon=100)
    assert out["ok"] is True
    assert out["at_touch_share"] == 1.0  # every cancel sat at its own touch


def test_cancel_lead_uses_pre_event_touch(monkeypatch: pytest.MonkeyPatch) -> None:
    orders, bb, ba, script = _cancel_touch_script(50)
    stub = _StubSim(orders, bb, ba, script, mid0=100.0)
    monkeypatch.setattr(cancel_lead_mod, "ZILobSimulator", lambda *a, **k: stub)
    out = sim_cancel_lead(horizon=100)
    assert out["n_cancels_at_touch"] == 50  # not zeroed by the dropped touch


def test_marketable_limit_uses_pre_event_touch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scripted submission lands strictly inside the pre-step
    spread; a post-step read would call the same order 'at own touch'."""
    orders = {1: _StubOrder("buy", 0), 2: _StubOrder("sell", 100)}
    script = []
    oid = 10
    current = dict(orders)
    for _ in range(60):
        bb = max(o.level for o in current.values() if o.side == "buy")
        ba = min(o.level for o in current.values() if o.side == "sell")
        lvl = bb + max(1, (ba - bb) // 3)
        nxt = dict(current)
        nxt[oid] = _StubOrder("buy", lvl)
        script.append(("limit", nxt, lvl, ba))
        current = nxt
        oid += 1
        if ba - lvl <= 2:
            current = dict(orders)
    stub = _StubSim(orders, 0, 100, script)
    monkeypatch.setattr(marketable_limit_mod, "ZILobSimulator", lambda *a, **k: stub)
    out = sim_marketable_limit(horizon=60)
    assert out["ok"] is True
    # ~92% of scripted submissions land strictly inside the pre-spread
    # (the periodic reset steps land behind); a post-step read would
    # reclassify every inside order as 'at own touch' (share 0).
    assert out["share"]["inside"] > 0.8
    assert out["share"]["at_own_touch"] == 0.0


def test_marketable_limit_resting_excludes_boundary() -> None:
    """rel == 0 is exactly at the opposite touch — marketable, not
    resting; it must not leak into the resting-size mean."""
    rel = np.asarray([0.0] * 30 + [-2.0] * 30)
    rel_own = np.asarray([5.0] * 30 + [1.0] * 30)
    sizes = np.asarray([2.0] * 30 + [4.0] * 30)
    out = _aggression_stats(rel, rel_own, sizes)
    assert out["share"]["marketable"] == pytest.approx(0.5)
    assert out["mean_size_resting"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# queue_reactive first-event state attribution
# ---------------------------------------------------------------------------


def test_first_event_attributed_to_trajectory_q0() -> None:
    traj = QueueTrajectory(
        times=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        states=np.asarray([1, 2, 1, 0, 1]),
        events=np.asarray([0, 0, 1, 2, 0]),  # limit, limit, cancel, market, limit
        q_max=4,
        q0=0,
    )
    est = estimate_rates(traj)
    # Event 0 (a limit) fired from state q0=0 — not the legacy midpoint 2.
    assert est.counts[0, 0] == 2.0  # limits from state 0: events 0 and 4
    assert est.counts[1, 0] == 1.0  # the event-1 limit fired from state 1
    assert est.counts[2, 1] == 1.0  # the event-2 cancel fired from state 2
    assert est.dwell[0] == pytest.approx(1.0)  # dt[0]=0 by diff-prepend convention


def test_simulate_queue_carries_q0() -> None:
    lam_l, lam_c, lam_m = [0.5] * 5, [0.3] * 5, [0.1] * 5
    traj = simulate_queue(
        np.asarray(lam_l),
        np.asarray(lam_c),
        np.asarray(lam_m),
        horizon=50.0,
        seed=3,
        q0=0,
    )
    assert traj.q0 == 0


# ---------------------------------------------------------------------------
# Strict-JSON seals (no literal NaN in signed payloads)
# ---------------------------------------------------------------------------


def test_queue_depletion_payload_is_strict_json() -> None:
    out = queue_depletion_bench(horizon=500.0, probe_interval=10.0, seed=0)
    blob = json.dumps(out, allow_nan=False)  # raises on any literal NaN
    assert len(out["payload_sha256"]) == 64
    assert "NaN" not in blob


def test_excursion_geometry_payload_is_strict_json() -> None:
    out = excursion_geometry_bench(horizon=200.0, inventory_cap=5, seed=0)
    blob = json.dumps(out, allow_nan=False)
    assert len(out["payload_sha256"]) == 64
    assert "NaN" not in blob


# ---------------------------------------------------------------------------
# Receipt conventions: kind/schema split + seal-gated evidence
# ---------------------------------------------------------------------------


def test_book_depletion_kind_schema_split() -> None:
    assert BOOK_DEPLETION_SCHEMA == "book_depletion.v1"
    assert BOOK_DEPLETION_KIND == "book_depletion"


def test_zone_map_unsealed_receipt_contributes_no_claims(tmp_path: Path) -> None:
    src = Path("receipts")
    for name in (
        "wave24_map.json",
        "band_occupancy.json",
        "zone_embargo.json",
        "zone_stability.json",
    ):
        shutil.copy(src / name, tmp_path / name)
    # Tamper one member: a flipped claim invalidates its seal.
    embargo = tmp_path / "zone_embargo.json"
    body = json.loads(embargo.read_text())
    body["claims"]["zone_reaches_occupancy"] = not body["claims"].get(
        "zone_reaches_occupancy", True
    )
    embargo.write_text(json.dumps(body, indent=1))
    out = zone_map(tmp_path)
    assert out["kind"] == "zone_map"
    lane = next(l_lane for l_lane in out["lanes"] if l_lane["receipt"] == "zone_embargo.json")
    assert lane["sealed"] is False
    assert lane["claims"] is None  # unsealed: no evidence reaches the map
    assert out["claims"]["all_member_receipts_sealed"] is False
    assert out["claims"]["all_lanes_present"] is False
    # The surviving lanes still carry their claims.
    assert any(l_lane["claims"] is not None for l_lane in out["lanes"])


# ---------------------------------------------------------------------------
# lob_resilience: an unrecovered depletion ends its episode, not the scan
# ---------------------------------------------------------------------------


def test_unrecovered_depletion_does_not_truncate_scan() -> None:
    out = _resilience_stats([0.0, 0.1, 0.2], [100, 20, 4])
    assert out["n_depletions"] == 2  # 100->20 and 20->4 are both depletions
    assert out["n_unrecovered"] == 2
    assert out["n_recovered"] == 0


# ---------------------------------------------------------------------------
# stale_quote sim arm: drift measured in ticks, not dollars
# ---------------------------------------------------------------------------


def test_sim_stale_quote_reports_tick_units(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scripted mid rises 0.01 per 0.1s step: a +1s forward drift is
    0.10 dollars = 10 ticks at tick=0.01. A dollar-unit leak reports
    0.10 mislabeled as ticks."""

    def _inject(sim: _StubSim) -> None:
        sim.trades.append(SimpleNamespace(t=sim.t, aggressor="buy", maker_t_submit=sim.t - 0.05))

    script = [("market", {}, 0, 100)] * 80
    stub = _StubSim({}, 0, 100, script, mid0=100.0, mid_step=0.01, inject=_inject)
    monkeypatch.setattr(stale_quote_mod, "ZILobSimulator", lambda *a, **k: stub)
    out = sim_stale_quote(horizon=80)
    assert out["ok"] is True
    assert out["n_with_drift"] >= 50
    # j+10 or j+11 steps ahead depending on accumulated-t drift; the
    # dollar-unit leak would report ~0.1 mislabeled as ticks.
    assert out["mean_signed_dmid_ticks"] == pytest.approx(10.0, abs=1.5)


# ---------------------------------------------------------------------------
# Direct pin on the row-0 convention: seed then apply double-counts
# ---------------------------------------------------------------------------


def test_seed_then_apply_double_counts_event0() -> None:
    """The defect mechanism, pinned at the engine level: a seeded row-0
    state already contains event 0's effect."""
    book = LobsterBook()
    book.seed([(50000, 5)], [(49000, 50)])  # row 0 already reflects event 0
    book.apply(LobsterEvent(0.0, SUBMISSION, 1, 50, 49000, 1))
    assert book.top("bid", 1) == [(49000, 100)]  # applying again doubles it
