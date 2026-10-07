"""micro_bench1 adversarial probes — honesty/determinism audit lane.

Probes the defect classes found auditing the wave-23 bench modules:

- channel-label honesty: removals fired outside the dispatched-event
  wrappers (hit_flee, touch_pull, maker expiry, repost drains) must land
  on the "cxl" channel, not a wrapper flag — a "?" kind used to crash
  the channel cells once those knobs were armed;
- causal bucketing: distance-to-touch must be measured against the
  PRE-event touch snapshot, never the post-event state;
- window boundaries: sim events are 1-indexed 1..horizon, so the last
  event must remain reachable as a follower and anchors whose window
  starts exactly at the horizon still count;
- per-fill anchor normalization: a multi-level sweep prints several
  fills under one event index — anchors are fills, not distinct events;
- degenerate inputs: empty buckets / zero horizons yield ``None`` or an
  ``unmeasured`` divergence, never a crash or a silently absent entry;
- fill-index integrity: a trade landing on a one-sided-book step must
  still be recorded under its own step index.

All inputs are SYNTHETIC and RNG-seeded; no tape files are read.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

import quant_fund.microstructure.crown_density_bench as cm
from quant_fund.microstructure.aftermath_flow_bench import (
    _CHANNELS,
    _WINDOWS,
    _dist_bucket,
    _empty_cell,
    _finish,
    _FlowSim,
    _window_of,
    sim_aftermath,
)
from quant_fund.microstructure.cancel_gradient_bench import (
    CANCEL_GRADIENT_BENCH_SCHEMA,
    cancel_gradient_bench,
)
from quant_fund.microstructure.continuation_attr_bench import _attr_totals, _AttrSim
from quant_fund.microstructure.improve_flow_bench import improve_flow_bench
from quant_fund.microstructure.initiative_fade_bench import initiative_fade_bench
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator

_HIT_FLEE = {"hit_flee_frac": 1.0, "hit_flee_band": 2, "hit_flee_window": 200}


def _drive(sim: ZILobSimulator, n: int) -> None:
    for _ in range(n):
        sim.step()


def test_flowsim_removal_kinds_follow_cause() -> None:
    """Armed hit_flee + touch_pull: every removal is a real channel.

    Under the wrapper-flag bookkeeping the flee removals (fired before
    the event dispatch) logged "?" — a kind the channel cells do not
    contain — and touch_pull removals inside ``_consume_best`` logged
    "fill". The cause-based label must keep fill kinds in 1:1 lockstep
    with visible fills (dark fills bypass ``_remove_resting_at``).
    """
    cfg = _calibrated(11, {**_HIT_FLEE, "touch_pull": 0.6})
    sim = _FlowSim(cfg, _split(3.0, 12))
    _drive(sim, 6000)
    kinds = {k for k, _s, _l, _e in sim.flow_log}
    assert kinds <= set(_CHANNELS)
    assert sim.n_hit_flees > 0  # the unwrapped path was armed and fired
    n_fill_kind = sum(1 for k, _s, _l, _e in sim.flow_log if k == "fill")
    assert n_fill_kind == sim.n_fills - sim.n_dark_fills
    n_cxl_kind = sum(1 for k, _s, _l, _e in sim.flow_log if k == "cxl")
    assert n_cxl_kind >= sim.n_touch_pulls + sim.n_hit_flees


def test_sim_aftermath_survives_unwrapped_removals() -> None:
    """hit_flee removals logged between events must not crash the pane."""
    cfg = _calibrated(3, {**_HIT_FLEE, "touch_pull": 0.5})
    out = sim_aftermath(cfg, _split(3.0, 4), 4000)
    assert set(out["windows"]) == {"1_10", "10_50", "50_200"}


def test_sim_aftermath_matches_independent_recomputation() -> None:
    """Recompute the panes with an independent seeded replica.

    The replica buckets each mutation against the touch snapshot taken
    BEFORE its event (touch_before[e2 - 1] for the 1-indexed event e2)
    and lets the final event remain a follower; exact equality with
    sim_aftermath's windows pins both the causal indexing and the
    inclusive horizon bound.
    """
    cfg = _calibrated(7, {"touch_pull": 0.5})
    horizon = 4000
    ref = sim_aftermath(cfg, _split(3.0, 8), horizon)

    sim = _FlowSim(_calibrated(7, {"touch_pull": 0.5}), _split(3.0, 8))
    touch_before: list[tuple[int | None, int | None]] = []
    for _ in range(horizon):
        touch_before.append((sim.best_bid_level, sim.best_ask_level))
        sim.step()
    by_ev: dict[int, list[tuple[str, str, int]]] = {}
    for kind, side, lvl, ev in sim.flow_log:
        by_ev.setdefault(ev, []).append((kind, side, lvl))

    # Independent recomputation: group by follower event first and walk
    # anchors backwards — a structurally different traversal producing
    # the same bucket multiset, so exact equality pins the product
    # loop's causal indexing rather than its implementation.
    # fills as a per-row list, not a dict on ev — a multi-fill event's
    # anchors each re-run the same window (per-fill normalization).
    fill_rows = [(ev, side == "buy") for kind, side, _l, ev in sim.flow_log if kind == "fill"]
    cells = {f"{lo}_{hi}": _empty_cell() for lo, hi in _WINDOWS}
    n_anchor = {f"{lo}_{hi}": 0 for lo, hi in _WINDOWS}
    for ev, _hb in fill_rows:
        for lo, hi in _WINDOWS:
            if ev + lo <= horizon:
                n_anchor[f"{lo}_{hi}"] += 1
    for e2, muts in by_ev.items():
        pre_bb, pre_ba = touch_before[e2 - 1]
        for ev, hit_is_buy in fill_rows:
            wname = _window_of(e2 - ev)
            if wname is None:
                continue
            for kind2, side2, lvl in muts:
                own_touch = pre_bb if side2 == "buy" else pre_ba
                if own_touch is None:
                    continue
                dist = own_touch - lvl if side2 == "buy" else lvl - own_touch
                rel = "hit" if (side2 == "buy") == hit_is_buy else "unhit"
                cells[wname][rel][kind2][_dist_bucket(dist)] += 1
    assert ref["windows"] == _finish(cells, n_anchor)


def test_attrsim_channel_labels_follow_cause() -> None:
    cfg = _calibrated(5, {**_HIT_FLEE, "touch_pull": 0.6})
    sim = _AttrSim(cfg, _split(3.0, 6))
    _drive(sim, 6000)
    chans = {ch for _e, ch, _s in sim.mut_log}
    assert chans <= {"lo", "fill", "cxl"}
    n_fill = sum(1 for _e, ch, _s in sim.mut_log if ch == "fill")
    assert n_fill == sim.n_fills - sim.n_dark_fills


def test_attr_totals_multi_fill_anchor_counts_per_fill() -> None:
    """A 3-fill sweep event is three anchors, matching the tape's rows."""
    recs = [(7, 1.0, 12, "lo", "hit", 0.6)] * 3
    out = _attr_totals(recs, {7: 3})
    assert out["n_anchor_fills"] == 3
    assert out["k200_per_channel_ticks"]["lo"] == pytest.approx(0.6)
    assert out["windows"]["1_10"]["n_anchor_fills"] == 3
    assert out["windows"]["1_10"]["per_channel_ticks"]["lo"]["hit"] == pytest.approx(0.6)


def test_attr_totals_multi_fill_anchor_unweighted_is_inflated() -> None:
    """Guard the semantics: without multiplicity the same records are a
    single anchor and the per-fill channel inflates 3x — the defect."""
    recs = [(7, 1.0, 12, "lo", "hit", 0.6)] * 3
    out = _attr_totals(recs)
    assert out["n_anchor_fills"] == 1
    assert out["k200_per_channel_ticks"]["lo"] == pytest.approx(1.8)


def test_cancel_gradient_degenerate_horizon() -> None:
    """horizon=0: every occupancy bucket empty -> propensity unmeasured,
    claims fail closed, receipt still seals."""
    out = cancel_gradient_bench(horizon=0.0)
    assert out["schema"] == CANCEL_GRADIENT_BENCH_SCHEMA
    assert out["divergences"]
    assert all("unmeasured" in d or "vs" in d for d in out["divergences"])
    assert all(v is False for v in out["claims"].values())


def test_sim_crown_onesided_step_fill_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fill on a one-sided-book step keeps its own row index.

    Scripted book: fill at step 1 empties the ask side; the book is
    refilled at step 2 where another fill lands. Deferring the trade
    drain past the empty check would record both fills under index 2 and
    merge them into one event's sweep footprint (sweep_p_ge2 = 1.0).
    """
    tr1 = types.SimpleNamespace(aggressor="buy", level=110, t=0.0)
    tr2 = types.SimpleNamespace(aggressor="buy", level=105, t=0.0)
    steps = [
        ({90: [1, 1]}, {110: [1]}, []),
        ({90: [1]}, {}, [tr1]),
        ({90: [1]}, {105: [1]}, [tr2]),
    ]

    class _ScriptedSim:
        def __init__(self) -> None:
            self.n_events = 0
            self.trades: list[Any] = []
            self._bids: dict[int, list[int]] = {}
            self._asks: dict[int, list[int]] = {}

        def step(self) -> None:
            b, a, trs = steps[self.n_events]
            self.n_events += 1
            self._bids, self._asks = b, a
            self.trades.extend(trs)

        def event_counts(self) -> dict[str, int]:
            return {
                "n_hidden_fills": 0,
                "n_mo_units": 0,
                "n_mo_arrivals": 0,
                "n_lo_capped": 0,
            }

    fake = _ScriptedSim()
    monkeypatch.setattr(cm, "ZILobSimulator", lambda cfg, flow: fake)
    out = cm._sim_crown(
        "scripted", None, horizon=3, seed=0, collect_counts=True, flow_intensity=None
    )
    assert out["n_fills"] == 2
    assert out["n_reveals"] == 0  # both reveal rows are sentinel-masked
    assert out["sweep_p_ge2"] == 0.0


def test_improve_flow_horizon_is_threaded() -> None:
    """The horizon parameter must reach the sim loop — a receipt that
    reports horizon=1.0 must not have measured a 1500s run."""
    out = improve_flow_bench(horizon=1.0)
    assert out["horizon"] == 1.0
    for arm in out["arms"].values():
        assert arm["n_lo_arrivals"] < 50


def test_initiative_fade_horizon_is_threaded() -> None:
    """Bucket edges follow the run horizon, not the module constant."""
    out = initiative_fade_bench(horizon=9.0)
    edges = [b["t_lo"] for b in out["arms"]["flat"]["thirds"]]
    assert edges == pytest.approx([0.0, 3.0, 6.0])
    assert out["horizon"] == 9.0
