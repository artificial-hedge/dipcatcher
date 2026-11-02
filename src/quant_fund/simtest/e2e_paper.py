"""Paper-session extension of the SYNTHETIC ingest-to-report smoke.

Several sessions of the same synthetic book go through
:class:`~quant_fund.paper.trading_session.PaperTradingSession`. The receipt
is an accounting check: a kill switch blocks new orders, an oversize order
is refused, a cancel still lands while halted, and a restart from the
checkpoint reproduces positions and cash. It is not market evidence.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.paper.trading_session import SessionLimits, run_session
from quant_fund.pipeline.dataset import clear_panel_cache, ensure_silver
from quant_fund.pipeline.e2e_smoke import (
    E2ESmokeError,
    load_smoke_config,
    require_finite,
    require_populated,
    require_rows,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.seeds import set_global_seed

_SESSIONS = 4
_NAMES = 2
_QTY = 1.0
_OVERSIZE = 11.0
_STARTING_CASH = 100_000.0
_BOOK_VERDICTS = {"mark": "marked", "submit": "accepted", "fill": "accepted"}
_STATE_NAME = "paper_session.json"


class E2EPaperError(E2ESmokeError):
    """The paper-session smoke was empty, non-finite, or refused its contract."""


def run_e2e_paper(
    root: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Trade the synthetic smoke book and return a digest-bearing receipt."""
    lake = Path(root)
    lake.mkdir(parents=True, exist_ok=True)
    cfg = load_smoke_config(lake, config_path=config_path, seed=seed)
    with _thread_limits():
        sessions = select_sessions(_load_bars(cfg))
    limits = _limits()
    state = lake / "metadata" / _STATE_NAME
    security_id = str(sessions[0]["quotes"][0]["security_id"])
    book = _book_state(state, sessions, limits)
    controls = _control(state, security_id, limits)
    restarted = _restart(state, limits)
    _require_restart(book, controls, restarted)
    return _receipt(cfg, sessions, restarted)


def select_sessions(bars: pl.DataFrame) -> list[dict[str, Any]]:
    """Last ``_SESSIONS`` dates and the first two names present on every date."""
    _require_book(bars)
    chosen = _session_times(bars)
    names = _names_for(bars, chosen)
    return [_session(bars, when, names) for when in chosen]


def main(argv: list[str] | None = None) -> int:
    """CLI entry: print the paper receipt JSON, or write a stage error to stderr."""
    parser = argparse.ArgumentParser(
        description="Deterministic SYNTHETIC paper-session correctness smoke"
    )
    parser.add_argument("--root", required=True, help="Writable lake root for this run")
    parser.add_argument(
        "--config", default=None, help="Config YAML (default configs/e2e_smoke.yaml)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the synthetic seed")
    args = parser.parse_args(argv)
    try:
        receipt = run_e2e_paper(args.root, config_path=args.config, seed=args.seed)
    except E2ESmokeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    sys.stdout.write(canonical_json_bytes(receipt).decode("utf-8"))
    sys.stdout.write("\n")
    return 0


def _load_bars(cfg: AppConfig) -> pl.DataFrame:
    set_global_seed(int(cfg.data.synthetic_seed))
    clear_panel_cache()
    return ensure_silver(cfg)


def _require_book(bars: pl.DataFrame) -> None:
    require_rows("paper-book", bars)
    require_populated("paper-book", bars, ("security_id", "event_time", "close", "source"))
    sources = {str(value) for value in bars.get_column("source").unique().to_list()}
    if sources != {"synthetic"}:
        raise E2EPaperError("paper-book: source is not synthetic")


def _session_times(bars: pl.DataFrame) -> list[datetime]:
    times = bars.get_column("event_time").unique().sort()
    if times.len() < _SESSIONS:
        raise E2EPaperError(f"paper-book: fewer than {_SESSIONS} sessions")
    tail = times.tail(_SESSIONS).to_list()
    if any(value is None for value in tail):
        raise E2EPaperError("paper-book: empty session time")
    return [cast(datetime, value) for value in tail]


def _names_for(bars: pl.DataFrame, times: Sequence[datetime]) -> list[str]:
    window = bars.filter(pl.col("event_time").is_in(list(times)))
    counts = window.group_by("security_id").len().sort("security_id")
    full = counts.filter(pl.col("len") == len(times))
    if full.height < _NAMES:
        raise E2EPaperError("paper-book: fewer than 2 complete names")
    return [str(value) for value in full.get_column("security_id").head(_NAMES).to_list()]


def _session(bars: pl.DataFrame, when: datetime, names: Sequence[str]) -> dict[str, Any]:
    return {
        "session_time": when.isoformat(),
        "quotes": [_quote(bars, when, name) for name in names],
    }


def _quote(bars: pl.DataFrame, when: datetime, name: str) -> dict[str, Any]:
    rows = bars.filter((pl.col("security_id") == name) & (pl.col("event_time") == when))
    if rows.height != 1:
        raise E2EPaperError(f"paper-book: expected one bar for {name}")
    price = float(rows.get_column("close")[0])
    if not math.isfinite(price) or price <= 0.0:
        raise E2EPaperError(f"paper-book: non-finite close for {name}")
    return {"security_id": name, "price": price}


def _limits() -> SessionLimits:
    return SessionLimits(
        max_order_quantity=10.0,
        max_order_notional=1_000_000.0,
        max_position_notional=1_000_000.0,
        max_gross_notional=5_000_000.0,
    )


def _book_state(
    state: Path, sessions: Sequence[Mapping[str, Any]], limits: SessionLimits
) -> dict[str, Any]:
    midpoint = _SESSIONS // 2
    opened = run_session(
        state_path=state,
        commands=_fill_commands(sessions, 0, midpoint),
        starting_cash=_STARTING_CASH,
        limits=limits,
    )
    _require_book_steps(opened, "paper-open")
    resumed = run_session(
        state_path=state,
        commands=_fill_commands(sessions, midpoint, _SESSIONS),
        resume=True,
        limits=limits,
    )
    _require_book_steps(resumed, "paper-resume")
    return resumed


def _fill_commands(
    sessions: Sequence[Mapping[str, Any]], start: int, stop: int
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for index in range(start, stop):
        commands.extend(_session_commands(index, sessions[index]))
    return commands


def _session_commands(index: int, session: Mapping[str, Any]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    when = str(session["session_time"])
    for quote in session["quotes"]:
        security_id = str(quote["security_id"])
        price = float(quote["price"])
        order_id = f"s{index}-{security_id}"
        commands.append({"op": "mark", "security_id": security_id, "price": price})
        commands.append(
            {
                "op": "submit",
                "order_id": order_id,
                "security_id": security_id,
                "side": "BUY",
                "quantity": _QTY,
            }
        )
        commands.append(
            {
                "op": "fill",
                "fill_id": f"f{index}-{security_id}",
                "order_id": order_id,
                "security_id": security_id,
                "quantity": _QTY,
                "price": price,
                "fill_time": when,
            }
        )
    return commands


def _require_book_steps(report: Mapping[str, Any], stage: str) -> None:
    steps = report["steps"]
    if not steps:
        raise E2EPaperError(f"{stage}: empty")
    for step in steps:
        expected = _BOOK_VERDICTS.get(str(step["op"]))
        if expected is None or step["verdict"] != expected:
            raise E2EPaperError(f"{stage}: {step['op']} {step['verdict']}")
    if not report["consistent"] or not report["nav_seam_parity"]:
        raise E2EPaperError(f"{stage}: seam failed")


def _control(state: Path, security_id: str, limits: SessionLimits) -> dict[str, Any]:
    report = run_session(
        state_path=state,
        commands=_control_commands(security_id),
        resume=True,
        limits=limits,
    )
    _require_controls(report)
    return report


def _control_commands(security_id: str) -> list[dict[str, Any]]:
    return [
        {
            "op": "submit",
            "order_id": "oversize",
            "security_id": security_id,
            "side": "BUY",
            "quantity": _OVERSIZE,
        },
        {
            "op": "submit",
            "order_id": "working",
            "security_id": security_id,
            "side": "BUY",
            "quantity": _QTY,
        },
        {"op": "kill", "reason": "e2e-paper"},
        {
            "op": "submit",
            "order_id": "halted",
            "security_id": security_id,
            "side": "BUY",
            "quantity": _QTY,
        },
        {"op": "cancel", "order_id": "working"},
    ]


def _require_controls(report: Mapping[str, Any]) -> None:
    steps = list(report["steps"])
    if len(steps) != 5:
        raise E2EPaperError("paper-controls: empty or short")
    _expect(steps[0], "rejected", "max_order_quantity")
    _expect(steps[1], "accepted", None)
    _expect(steps[2], "killed", None)
    _expect(steps[3], "rejected", "kill switch")
    _expect(steps[4], "cancelled", None)
    if report["reconciliation"]["kill_state"] != "HALT_NEW_ORDERS":
        raise E2EPaperError("paper-controls: kill switch did not halt")


def _expect(step: Mapping[str, Any], verdict: str, reason_part: str | None) -> None:
    if step["verdict"] != verdict:
        raise E2EPaperError(f"paper-controls: {step['op']} {step['verdict']}")
    if reason_part is not None and reason_part not in str(step.get("reason", "")):
        raise E2EPaperError(f"paper-controls: {step['op']} reason")


def _restart(state: Path, limits: SessionLimits) -> dict[str, Any]:
    report = run_session(
        state_path=state,
        commands=[{"op": "checkpoint"}],
        resume=True,
        limits=limits,
    )
    steps = report["steps"]
    if len(steps) != 1 or steps[0]["verdict"] != "checkpointed":
        raise E2EPaperError("paper-restart: empty")
    if not report["nav_seam_parity"]:
        raise E2EPaperError("paper-restart: seam failed")
    return report


def _require_restart(
    book: Mapping[str, Any],
    controls: Mapping[str, Any],
    restarted: Mapping[str, Any],
) -> None:
    if not _same_book(book, controls) or not _same_book(controls, restarted):
        raise E2EPaperError("paper: restart positions or cash diverged")


def _same_book(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_positions, left_cash = _positions_cash(left)
    right_positions, right_cash = _positions_cash(right)
    if set(left_positions) != set(right_positions):
        return False
    quantities = (
        math.isclose(left_positions[key], right_positions[key], rel_tol=0.0, abs_tol=1e-9)
        for key in left_positions
    )
    return all(quantities) and math.isclose(left_cash, right_cash, rel_tol=0.0, abs_tol=1e-6)


def _positions_cash(report: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    raw = report["reconciliation"]["computed_positions"]
    if not isinstance(raw, dict) or not raw:
        raise E2EPaperError("paper: empty positions")
    positions = {str(key): float(value) for key, value in raw.items()}
    cash = float(report["reconciliation"]["computed_cash"])
    if any(not math.isfinite(qty) or qty == 0.0 for qty in positions.values()):
        raise E2EPaperError("paper: non-finite positions")
    if not math.isfinite(cash):
        raise E2EPaperError("paper: non-finite cash")
    return positions, cash


def _receipt(
    cfg: AppConfig,
    sessions: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    positions, cash = _positions_cash(report)
    names = [str(quote["security_id"]) for quote in sessions[0]["quotes"]]
    body: dict[str, Any] = {
        "schema": "e2e_paper.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "live_orders_enabled": False,
        "would_promote_live": False,
        "panel": {
            "n_assets": int(cfg.data.synthetic_n_assets),
            "n_days": int(cfg.data.synthetic_n_days),
            "seed": int(cfg.data.synthetic_seed),
        },
        "sessions": len(sessions),
        "names": names,
        "fills": len(sessions) * len(names),
        "controls": {
            "oversize_reason": "max_order_quantity",
            "halted_submit": True,
            "cancel_verdict": "cancelled",
            "restart_match": True,
            "kill_state": str(report["reconciliation"]["kill_state"]),
            "seam_ok": bool(report["nav_seam_parity"]),
        },
        "positions": positions,
        "cash": cash,
    }
    require_finite("paper-receipt", body)
    return {**body, "digest": hash_bytes(canonical_json_bytes(body))}


@contextmanager
def _thread_limits() -> Iterator[None]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        with nullcontext():
            yield
        return
    with threadpool_limits(limits=1):
        yield


if __name__ == "__main__":
    raise SystemExit(main())
