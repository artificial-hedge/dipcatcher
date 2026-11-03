"""Crash resume replays only the unprocessed window and keeps the live claim false."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.loop import (
    _bar_maps,
    _paper_resume_fingerprint,
    run_paper_loop,
)


def _cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml").model_copy(deep=True)
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def _bars(n_days: int = 4) -> pl.DataFrame:
    rows = []
    for day in range(n_days):
        stamp = datetime(2024, 1, 1 + day, tzinfo=UTC)
        for security_id, price in (("A", 100.0 + day), ("B", 50.0 + day)):
            rows.append(
                {
                    "security_id": security_id,
                    "event_time": stamp,
                    "open": price,
                    "close": price,
                    "close_total_return": price,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n_days: int = 3, scale: float = 1.0) -> pl.DataFrame:
    rows = []
    for day in range(n_days):
        stamp = datetime(2024, 1, 1 + day, tzinfo=UTC)
        rows.append({"event_time": stamp, "security_id": "A", "target_weight": 0.1 * scale})
        rows.append({"event_time": stamp, "security_id": "B", "target_weight": -0.05 * scale})
    return pl.DataFrame(rows)


def _panel_fn(weights: pl.DataFrame):
    def _fn(dt: datetime, _cfg: object) -> dict[str, float]:
        rows = weights.filter(pl.col("event_time") == dt)
        return {
            str(row["security_id"]): float(row["target_weight"])
            for row in rows.iter_rows(named=True)
        }

    return _fn


def _plant_unmarked(broker) -> None:
    """A held name with no fresh bar must not become an execution."""
    broker.shares["GHOST"] = 1.0
    broker.last_marks["GHOST"] = 10.0


def test_resume_replays_only_the_unprocessed_window(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars()
    weights = _weights()
    first = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        shadow_weights=_weights(scale=0.5),
        max_steps=2,
        prefer_latest=False,
        run_id="resume-window",
        initial_nav=100_000.0,
    )
    assert first.metrics["live_pnl_claim"] is False
    assert first.metrics["n_steps"] == 2
    assert first.promotion["would_promote_live"] is False
    assert first.source_note == "SYNTHETIC"

    state_path = Path(first.paths["broker_state"])
    state = json.loads(state_path.read_text())
    assert state["step"] == 2
    assert isinstance(state["resume_fingerprint"], str)
    state["mark_ages"] = None
    state_path.write_text(json.dumps(state))

    resumed = run_paper_loop(
        bars,
        cfg,
        champion_fn=_panel_fn(weights),
        shadow_weights=_weights(scale=0.5),
        challenger_weights={"scale_0.25": _weights(scale=0.25)},
        max_steps=1,
        resume=True,
        resume_run_id="resume-window",
        prepare_broker=_plant_unmarked,
        initial_nav=100_000.0,
    )
    assert resumed.metrics["resumed"] is True
    assert resumed.metrics["n_steps_this_run"] == 1
    assert resumed.metrics["n_steps"] == 3
    assert resumed.metrics["n_extra_challengers"] == 1
    assert resumed.metrics["live_pnl_claim"] is False
    assert resumed.metrics["research_only"] is True
    assert resumed.metrics["kill_tripped_mid_run"] is False
    assert resumed.promotion["would_promote_live"] is False
    assert resumed.promotion["data_source"] == "SYNTHETIC"
    assert resumed.source_note == "SYNTHETIC"
    if resumed.orders.height:
        assert "GHOST" not in resumed.orders["security_id"].to_list()


def test_resume_refuses_altered_bars_and_a_missing_exec_cursor(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars()
    weights = _weights()
    run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        max_steps=1,
        prefer_latest=False,
        run_id="moved-bars",
        initial_nav=100_000.0,
    )
    moved = bars.with_columns(
        pl.when(pl.col("event_time") == bars["event_time"][0])
        .then(pl.col("close") + 7.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        run_paper_loop(
            moved,
            cfg,
            champion_weights=weights,
            max_steps=1,
            resume=True,
            resume_run_id="moved-bars",
            initial_nav=100_000.0,
        )

    cursor = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        max_steps=1,
        prefer_latest=False,
        run_id="missing-exec",
        initial_nav=100_000.0,
    )
    state_path = Path(cursor.paths["broker_state"])
    state = json.loads(state_path.read_text())
    state["last_exec"] = None
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="last_exec"):
        run_paper_loop(
            bars,
            cfg,
            champion_weights=weights,
            max_steps=1,
            resume=True,
            resume_run_id="missing-exec",
            initial_nav=100_000.0,
        )


def test_wall_clock_flat_book_and_midrun_halt_stop_cleanly(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars()
    weights = _weights()

    wall = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        use_wall_clock=True,
        run_id="wall",
        initial_nav=100_000.0,
    )
    assert wall.metrics["n_steps_this_run"] == 0
    assert wall.metrics["live_pnl_claim"] is False
    assert wall.promotion["would_promote_live"] is False

    def _flatten(broker) -> None:
        broker.cash = 0.0

    flat = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        prepare_broker=_flatten,
        max_steps=2,
        prefer_latest=False,
        run_id="flat",
        initial_nav=100_000.0,
    )
    assert flat.metrics["n_steps_this_run"] == 0
    assert flat.metrics["live_pnl_claim"] is False
    assert flat.promotion["would_promote_live"] is False

    halted = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        shadow_weights=_weights(scale=0.5),
        max_steps=2,
        prefer_latest=False,
        halt_after_steps=1,
        run_id="halt-shadow",
        initial_nav=100_000.0,
    )
    assert halted.metrics["kill_tripped_mid_run"] is True
    assert halted.metrics["n_steps_this_run"] == 2
    assert halted.metrics["live_pnl_claim"] is False
    assert halted.promotion["would_promote_live"] is False

    halted_book = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        max_steps=2,
        prefer_latest=False,
        halt_after_steps=1,
        run_id="halt-book",
        initial_nav=100_000.0,
    )
    assert halted_book.metrics["kill_tripped_mid_run"] is True
    assert halted_book.metrics["has_shadow"] is False
    assert halted_book.promotion["would_promote_live"] is False


def test_setup_without_a_shadow_keeps_the_claim_false(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    bars = _bars()
    weights = _weights()
    seen: list[str] = []

    def _note(broker) -> None:
        seen.append(broker.slot)

    quiet = run_paper_loop(
        bars,
        cfg,
        champion_fn=lambda _dt, _config: {},
        max_steps=0,
        prefer_latest=False,
        prepare_broker=_note,
        run_id="quiet",
    )
    assert seen == ["champion"]
    assert quiet.metrics["has_shadow"] is False
    assert quiet.metrics["n_steps_this_run"] == 0
    assert quiet.metrics["live_pnl_claim"] is False
    assert quiet.promotion["would_promote_live"] is False

    def _same(_dt: datetime, _config: object) -> dict[str, float]:
        return {"A": 0.0}

    overlap = run_paper_loop(
        bars,
        cfg,
        champion_fn=_same,
        challenger_fns={"same": _same},
        challenger_weights={"same": weights},
        max_steps=0,
        prefer_latest=False,
        run_id="overlap",
    )
    assert overlap.metrics["has_shadow"] is True
    assert overlap.metrics["n_extra_challengers"] == 0
    assert overlap.metrics["live_pnl_claim"] is False

    day = pl.DataFrame(
        {
            "security_id": ["A"],
            "open": [None],
            "close": [10.0],
            "close_total_return": [None],
            "adv": [None],
            "vol_20": [None],
        }
    )
    _marks, close_marks, advs, vols = _bar_maps(day, use_open=False)
    assert close_marks["A"] == 10.0
    assert advs["A"] == 1.0
    assert vols["A"] == 0.02

    bare = pl.DataFrame({"event_time": [datetime(2024, 1, 1, tzinfo=UTC)], "close": [1.0]})
    digest = _paper_resume_fingerprint(bare, cfg, bare["event_time"][0])
    assert len(digest) == 64
