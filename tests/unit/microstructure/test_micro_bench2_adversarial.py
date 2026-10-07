"""Adversarial probes for the micro_bench2 audit lane.

Each probe pins a defect class found during the line-level audit:
LOBSTER event-0 double-apply (seed row is the post-message-0 state),
vacuous-true claims on absent arms, mutation-aliasing of module
constants into sealed payloads, post-fill window normalization,
within-step reseed resolution, and horizon/seed threading. All inputs
are synthetic and deterministic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.instant_decomp_bench import lobster_instant_decomp
from quant_fund.microstructure.joint_fit_bench import _TARGET, _TOL, joint_fit_bench
from quant_fund.microstructure.joint_stability_bench import joint_stability_bench
from quant_fund.microstructure.level_gap_bench import _gap_stats, level_gap_bench
from quant_fund.microstructure.lo_response_bench import lobster_lo_response
from quant_fund.microstructure.lobster import (
    LobsterEvent,
    validate_reconstruction,
)
from quant_fund.microstructure.maker_age_bench import _arm as maker_age_arm
from quant_fund.microstructure.repost_frontier_bench import repost_frontier_bench
from quant_fund.microstructure.reseed_hazard_bench import sim_reseed
from quant_fund.microstructure.spread_decomp import decompose
from quant_fund.microstructure.sweep_width_bench import _arm as sweep_width_arm
from quant_fund.microstructure.tape_stats import load_tape_stats
from quant_fund.microstructure.tape_surgery import _facts_seeded
from quant_fund.microstructure.zone_churn_bench import _mean as churn_mean
from quant_fund.microstructure.zone_embargo_bench import _cell_draws
from quant_fund.microstructure.zone_ttl_bench import _mean as ttl_mean


def _ev(t: float, typ: int, oid: int, size: int, price: int, direction: int) -> LobsterEvent:
    return LobsterEvent(
        time_s=t, event_type=typ, order_id=oid, size=size, price=price, direction=direction
    )


def _write_tape(tmp_path: Path, messages: list[str], rows: list[str]) -> tuple[Path, Path]:
    msg = tmp_path / "msg.csv"
    ob = tmp_path / "ob.csv"
    msg.write_text("\n".join(messages) + "\n")
    ob.write_text("\n".join(rows) + "\n")
    return msg, ob


def test_tape_surgery_event_zero_not_double_applied(tmp_path: Path) -> None:
    """The seed row is already post-event-0; replaying it doubles its size."""
    ev0 = _ev(34200.1, 1, 1, 100, 2000500, -1)  # SUB ask 100@2000500 (in seed row)
    ev1 = _ev(34200.2, 1, 2, 25, 2000100, 1)  # SUB bid
    ev2 = _ev(34200.3, 4, 1, 100, 2000500, -1)  # EXECUTION on the ask (buy MO)
    ev3 = _ev(34200.4, 1, 3, 10, 2000050, 1)  # SUB bid
    events = [ev0, ev1, ev2, ev3]
    out = _facts_seeded(
        events,
        seed_asks=[(2000500, 100)],
        seed_bids=[(2000000, 50)],
        skip_apply_of=ev0,
    )
    assert out["n_kept"] == 4
    assert out["n_exec"] == 1
    # i=0 sample reads the seed book exactly once: 100 + 50 = 150.
    assert out["touch_depth_med_shares"] == 150.0
    assert out["spread_mean_ticks"] == 5.0


def test_validate_reconstruction_no_phantom_resync(tmp_path: Path) -> None:
    """Double-applying event 0 fabricates a resync at event 1."""
    msg, ob = _write_tape(
        tmp_path,
        [
            "34200.1,1,1,100,2000500,-1",
            "34200.2,1,2,25,2000100,1",
        ],
        [
            "2000500,100,2000000,50",
            "2000500,100,2000100,25,-1,0,2000000,50",
        ],
    )
    out = validate_reconstruction(msg, ob)
    assert out["n_resync_events"] == 0
    assert out["n_match"] == out["n_compared"] == 1


def test_lo_response_windows_count_events_not_windows(tmp_path: Path) -> None:
    """submissions_per_event divides by event count, not window count."""
    messages = [
        "34200.1,1,1,50,2000000,1",  # SUB bid (seed-row event)
        "34200.2,1,2,100,2000500,-1",  # SUB ask
        "34200.3,4,2,100,2000500,-1",  # EXECUTION (buy aggressor)
        "34200.4,1,3,25,2000400,1",  # SUB bid, unhit side, 4 ticks out
        "34200.5,1,4,30,2000800,-1",  # SUB ask, hit side, 3 ticks out
        "34200.6,4,5,50,2000500,-1",  # EXECUTION (buy aggressor)
        "34200.7,1,6,10,2000600,-1",  # SUB ask, hit side, 2 ticks in
    ]
    rows = [
        "2000500,50,2000000,50",
        "2000500,150,2000000,50",
        "2000500,50,2000000,50",
        "2000500,50,2000400,25,-1,0,2000000,50",
        "2000500,50,2000400,25,2000800,30,2000000,50",
        "2000800,30,2000400,25,-1,0,2000000,50",
        "2000600,10,2000400,25,2000800,30,2000000,50",
    ]
    msg, ob = _write_tape(tmp_path, messages, rows)
    out = lobster_lo_response(msg, ob, horizons=(1, 3))
    assert out["ok"] is True
    h1, h3 = out["post_fill"]["1"], out["post_fill"]["3"]
    # h=1: windows {3} and {6} -> 2 events, 2 submissions.
    assert h1["n_submissions"] == 2
    assert h1["submissions_per_event"] == pytest.approx(1.0)
    assert h1["share_unhit"] == pytest.approx(0.5)
    assert h1["mean_dist_unhit"] == pytest.approx(4.0)
    assert h1["mean_dist_hit"] == pytest.approx(2.0)
    # h=3: windows {3,4,5} and {6} -> 4 events, 3 submissions.
    # (Counting windows instead of events would report 3/2 = 1.5.)
    assert h3["n_submissions"] == 3
    assert h3["submissions_per_event"] == pytest.approx(0.75)
    assert h3["share_unhit"] == pytest.approx(1 / 3)
    assert h3["mean_dist_hit"] == pytest.approx(2.5)
    # Unconditional: only event 1 is a measurable baseline submission.
    assert out["unconditional"]["submissions_per_event"] == pytest.approx(1 / 3)


def test_instant_decomp_partial_fill_is_not_empty(tmp_path: Path) -> None:
    """A fill that leaves size at the touch is not an emptying."""
    messages = ["34200.1,1,1,50,2000000,1"]
    rows = ["2000500,50,2000000,50"]
    t = 1
    for k in range(10):
        messages.append(f"3420{1 + k}.1,1,{10 + k},100,2000500,-1")
        rows.append("2000500,150,2000000,50")
        # k==4 is a partial fill (40 left); the rest empty the touch.
        size = 110 if k == 4 else 150
        messages.append(f"3420{1 + k}.2,4,{100 + k},{size},2000500,-1")
        rows.append("2000500,40,2000000,50" if k == 4 else "2000600,30,2000000,50")
        t += 1
    msg, ob = _write_tape(tmp_path, messages, rows)
    out = lobster_instant_decomp(msg, ob)
    assert out["ok"] is True
    assert out["n_fills"] == 10
    assert out["p_empty_touch"] == pytest.approx(0.9)
    assert out["mean_gap_ticks_when_empty"] == pytest.approx(1.0)


def test_repost_frontier_tape_claims_fail_closed() -> None:
    """Absent tape arm must not make tape-dependent claims vacuously true."""
    out = repost_frontier_bench(tape_dir=None, horizon=1500, seed=7)
    assert out["tape"] is None
    assert out["claims"]["tape_reseeds_majority"] is False
    assert out["claims"]["tape_reseed_returns_to_touch"] is False
    assert out["claims"]["joint_undeerseeds"] is False
    assert len(out["receipt_sha256"]) == 64


def test_sim_reseed_episode_accounting() -> None:
    """Resolutions cannot exceed episodes; touch-share is a proper fraction."""
    out = sim_reseed("probe", {}, horizon=1500, seed=11)
    assert out["n_emptied"] >= 0
    if out["reseed_rate_500"] is not None:
        assert 0.0 <= out["reseed_rate_500"] <= 1.0
        assert out["reseed_as_touch_share"] is None or (0.0 <= out["reseed_as_touch_share"] <= 1.0)
    # Determinism: same seed -> identical episode ledger.
    again = sim_reseed("probe", {}, horizon=1500, seed=11)
    assert again == out


def test_horizon_threaded_as_prefix() -> None:
    """A smaller horizon run is a strict prefix of the larger one."""
    short = maker_age_arm(13, sized=False, horizon=120.0)
    long = maker_age_arm(13, sized=False, horizon=600.0)
    assert short["n_fills"] <= long["n_fills"]
    assert maker_age_arm(13, sized=False, horizon=300.0) == maker_age_arm(
        13, sized=False, horizon=300.0
    )
    sw_short = sweep_width_arm(13, sized=False, horizon=120.0)
    sw_long = sweep_width_arm(13, sized=False, horizon=600.0)
    assert sw_short["n_bursts"] <= sw_long["n_bursts"]


def test_zone_embargo_seed_threading() -> None:
    """Draws record the actual run seed (seed*1000 + lane seed)."""
    draws = _cell_draws(0, 280, 0.6, 2.0, horizon=1000, seed=7)
    assert [d["run_seed"] for d in draws] == [7007, 7011]
    for d in draws:
        assert isinstance(d["pins"], dict)
        assert d["n_fills"] >= 0


def test_zone_means_none_not_zero() -> None:
    """A cell whose draws are all missing reports None, not 0.0."""
    for fn in (churn_mean, ttl_mean):
        assert fn([None, None]) is None
        assert fn([1.0, None, 3.0]) == pytest.approx(2.0)


def test_joint_fit_payload_copies_constants() -> None:
    """Sealed payload must not alias the module's mutable target dicts."""
    out = joint_fit_bench(horizon=300, seed=7)
    assert out["targets"] == dict(_TARGET)
    assert out["targets"] is not _TARGET
    assert out["tolerances"] is not _TOL
    out["targets"]["instant"] = -999.0
    assert _TARGET["instant"] != -999.0


def test_joint_stability_reports_used_band() -> None:
    """The pin tolerates 3x the tape ceiling — the payload must say so."""
    out = joint_stability_bench(horizon=500)
    assert out["tape_pins"]["spread_band"] == [9, 21]
    assert out["tape_pins"]["spread_band_used"] == [9, 63]


def test_spread_decomp_mean_spread_populated() -> None:
    rng = np.random.default_rng(7)
    signs = rng.choice([-1.0, 1.0], size=40)
    mids = 200.0 + np.cumsum(rng.normal(0.0, 0.4, size=40))
    fit = decompose(signs, mids, np.diff(mids), horizon_trades=5, mean_spread_ticks=3.25)
    assert fit.mean_spread_ticks == pytest.approx(3.25)
    fit2 = decompose(signs, mids, np.diff(mids), horizon_trades=5)
    assert math.isnan(fit2.mean_spread_ticks)


def test_level_gap_fails_closed_on_empty_tape(tmp_path: Path) -> None:
    """An orderbook file with no rows leaves the tape claim off, not crashed."""
    (tmp_path / "SYN_x_orderbook_1.csv").write_text("")
    out = level_gap_bench(tmp_path, ticker="SYN", horizon=300, seed=3)
    assert out["real"]["ok"] is False
    assert out["claims"]["tape_near_touch_is_sparse"] is False
    assert len(out["receipt_sha256"]) == 64


def test_gap_stats_empty() -> None:
    assert _gap_stats([]) == {"ok": False, "n": 0}


def test_tape_stats_rejects_bad_values(tmp_path: Path) -> None:
    def _root(events_per_s: float, occ: float) -> Path:
        (tmp_path / "round_lot_syn.json").write_text(
            json.dumps({"real": {"size_hist": {"1-10": 5}}})
        )
        (tmp_path / "event_matrix_syn.json").write_text(
            json.dumps({"real": {"events_per_s": {"sub": events_per_s}}})
        )
        (tmp_path / "spread_dynamics_syn.json").write_text(
            json.dumps({"real": {"spread_occupancy": {"1": occ}, "mean_spread_ticks": 3.0}})
        )
        return tmp_path

    root = _root(0.5, 0.4)
    ts = load_tape_stats(root, "syn")
    assert ts.rate("sub") == pytest.approx(0.5)
    with pytest.raises(ValueError):
        load_tape_stats(_root(-1.0, 0.4), "syn")
    with pytest.raises(ValueError):
        load_tape_stats(_root(0.5, float("nan")), "syn")


def test_streak_calibrate_degenerate_arm_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero-fill arm returns ok=False stats — claims must fail closed."""
    import quant_fund.microstructure.streak_calibrate_bench as m

    monkeypatch.setattr(m, "_arm", lambda seed, **kw: {"ok": False, "reason": "no_runs"})
    out = m.streak_calibrate_bench(seed=1)
    assert all(v is False for v in out["claims"].values())
    assert any("degenerate" in d for d in out["divergences"])
    assert len(out["receipt_sha256"]) == 64
