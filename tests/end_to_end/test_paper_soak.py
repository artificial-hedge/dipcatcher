"""Thirty-session SYNTHETIC paper soak."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.paper.trading_session import module_network_imports
from quant_fund.pipeline.e2e_smoke import E2ESmokeError
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent
from quant_fund.simtest.paper_soak import (
    E2ESoakError,
    mark_residual,
    run_paper_soak,
    select_soak_sessions,
    trade_residual,
)

pytestmark = [pytest.mark.e2e_smoke, pytest.mark.synthetic]


def _when(offset: int) -> datetime:
    return datetime(2024, 1, 2, 16, 0, tzinfo=UTC) + timedelta(days=offset)


def _row(name: str, offset: int, close: float) -> dict[str, object]:
    return {
        "security_id": name,
        "event_time": _when(offset),
        "close": close,
        "source": "synthetic",
    }


def test_soak_rejects_short_and_nonfinite_books() -> None:
    with pytest.raises(E2ESoakError, match="sessions must be an int"):
        run_paper_soak("/tmp/unused-soak", sessions=29)
    rows = [_row(name, offset, 10.0) for offset in range(30) for name in ("SEC_0001", "SEC_0002")]
    rows[0]["close"] = float("nan")
    with pytest.raises(E2ESmokeError, match="non-finite"):
        select_soak_sessions(pl.DataFrame(rows), 30)
    with pytest.raises(E2ESoakError, match="non-finite cash"):
        trade_residual(float("nan"), 1.0, 1.0)


def test_mark_residual_is_the_overnight_move_only() -> None:
    # Buy one share at 10. Book value stays 100: cash 90 plus one share at 10.
    assert trade_residual(100.0, 90.0, 10.0) == 0.0
    assert mark_residual(100.0, {}, {}, 90.0, {"A": 1.0}, {"A": 10.0}) == 0.0
    # The share is now marked at 12. Value rises by 2, which is the whole move.
    assert mark_residual(90.0, {"A": 1.0}, {"A": 10.0}, 78.0, {"A": 2.0}, {"A": 12.0}) == 0.0


def test_module_has_no_network_imports() -> None:
    source = Path("src/quant_fund/simtest/paper_soak.py").read_text(encoding="utf-8")
    assert module_network_imports(source) == []


def test_thirty_sessions_conserve_cash_and_positions(tmp_path: Path) -> None:
    started = time.perf_counter()
    first = run_paper_soak(tmp_path / "a")
    second = run_paper_soak(tmp_path / "b")
    elapsed = time.perf_counter() - started
    assert elapsed < 120.0, f"paper soak took {elapsed:.1f}s"
    assert first == second
    assert first["sessions"] >= 30
    assert first["days_checked"] == first["sessions"]
    assert first["data_label"] == "SYNTHETIC"
    assert first["live_pnl_claim"] is False
    assert first["would_promote_live"] is False
    assert first["max_cash_gap"] <= 1e-6
    assert first["max_position_gap"] <= 1e-9
    assert first["max_trade_residual"] <= 1e-6
    assert first["max_mark_residual"] <= 1e-6
    assert set(first["positions"]) == {"SEC_0001", "SEC_0002"}
    assert all(qty == float(first["sessions"]) for qty in first["positions"].values())
    assert first["cash"] < 1_000_000.0
    assert family_blob_forbidden_metrics_absent(first)
