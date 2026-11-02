"""Deterministic SYNTHETIC paper-session smoke."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from quant_fund.paper.trading_session import module_network_imports
from quant_fund.pipeline.e2e_smoke import E2ESmokeError
from quant_fund.research.catalog.registry import family_blob_forbidden_metrics_absent
from quant_fund.simtest.e2e_paper import E2EPaperError, main, select_sessions
from quant_fund.simtest.e2e_paper import run_e2e_paper as trade_book

pytestmark = [pytest.mark.e2e_smoke, pytest.mark.synthetic]


def _when(offset: int) -> datetime:
    return datetime(2024, 1, 2, 16, 0, tzinfo=UTC) + timedelta(days=offset)


def _bars(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _row(name: str, offset: int, close: float, source: str = "synthetic") -> dict[str, object]:
    return {
        "security_id": name,
        "event_time": _when(offset),
        "close": close,
        "source": source,
    }


def test_select_sessions_rejects_empty_nonfinite_and_short_books() -> None:
    with pytest.raises(E2ESmokeError, match="empty"):
        select_sessions(_bars([]))
    nan = [_row("SEC_0001", offset, 10.0) for offset in range(4)]
    nan[0]["close"] = float("nan")
    with pytest.raises(E2ESmokeError, match="non-finite"):
        select_sessions(_bars(nan))
    inf = [_row("SEC_0001", offset, 10.0) for offset in range(4)]
    inf[-1]["close"] = float("inf")
    with pytest.raises(E2ESmokeError, match="non-finite"):
        select_sessions(_bars(inf))
    short = [_row(name, offset, 10.0) for offset in range(3) for name in ("SEC_0001", "SEC_0002")]
    with pytest.raises(E2EPaperError, match="fewer than 4"):
        select_sessions(_bars(short))
    one = [_row("SEC_0001", offset, 10.0) for offset in range(4)]
    with pytest.raises(E2EPaperError, match="fewer than 2"):
        select_sessions(_bars(one))
    foreign = [
        _row(name, offset, 10.0, source="file") for offset in range(4) for name in ("A", "B")
    ]
    with pytest.raises(E2EPaperError, match="not synthetic"):
        select_sessions(_bars(foreign))


def test_select_sessions_keeps_two_names_over_four_dates() -> None:
    rows = [
        _row(name, offset, 10.0 + offset)
        for offset in range(6)
        for name in ("SEC_0002", "SEC_0001")
    ]
    sessions = select_sessions(_bars(rows))
    assert len(sessions) == 4
    assert [quote["security_id"] for quote in sessions[0]["quotes"]] == ["SEC_0001", "SEC_0002"]
    assert sessions[-1]["quotes"][0]["price"] == 15.0


def test_main_rejects_a_non_synthetic_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("data:\n  source: file\n  root: data\n")
    code = main(["--root", str(tmp_path / "lake"), "--config", str(bad)])
    assert code == 1
    assert "non-synthetic" in capsys.readouterr().err


def test_paper_module_has_no_network_imports() -> None:
    source = Path("src/quant_fund/simtest/e2e_paper.py").read_text(encoding="utf-8")
    assert module_network_imports(source) == []


def test_two_runs_share_a_digest(tmp_path: Path) -> None:
    started = time.perf_counter()
    first = trade_book(tmp_path / "a")
    second = trade_book(tmp_path / "b")
    elapsed = time.perf_counter() - started
    assert elapsed < 300.0, f"e2e paper took {elapsed:.1f}s"
    assert first == second
    assert len(first["digest"]) == 64
    assert first["data_label"] == "SYNTHETIC"
    assert first["live_pnl_claim"] is False
    assert first["live_orders_enabled"] is False
    assert first["would_promote_live"] is False
    assert first["sessions"] == 4
    assert first["fills"] == 8
    assert first["controls"]["oversize_reason"] == "max_order_quantity"
    assert first["controls"]["halted_submit"] is True
    assert first["controls"]["cancel_verdict"] == "cancelled"
    assert first["controls"]["restart_match"] is True
    assert first["controls"]["kill_state"] == "HALT_NEW_ORDERS"
    assert first["controls"]["seam_ok"] is True
    assert first["positions"] == {"SEC_0001": 4.0, "SEC_0002": 4.0}
    assert first["cash"] < 100_000.0
    assert family_blob_forbidden_metrics_absent(first)
    state = tmp_path / "a" / "metadata" / "paper_session.json"
    assert state.is_file()
    assert "SEC_0001" in state.read_text(encoding="utf-8")
