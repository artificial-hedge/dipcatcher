"""Ledger flush failures, stale valuation, and a torn checkpoint that must not publish."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from quant_fund.execution.order_recon import Fill, Order
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.paper.ledger import PaperLedger
from quant_fund.paper.loop import (
    StaleValuationError,
    _bounded_valuation_marks,
    _require_valuation_marks,
)

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


class _Shares:
    def __init__(self, shares: dict[str, float]) -> None:
        self.shares = shares


def test_negative_stale_window_and_held_marks_fail_closed() -> None:
    with pytest.raises(ValueError, match="max_stale_bars must be non-negative"):
        _bounded_valuation_marks(object(), {}, {}, max_stale_bars=-1)  # type: ignore[arg-type]

    _require_valuation_marks(_Shares({"FLAT": 0.0, "HELD": 1.0}), {"HELD": 10.0})
    with pytest.raises(StaleValuationError, match="HELD"):
        _require_valuation_marks(_Shares({"HELD": 2.0, "FLAT": 0.0}), {})


def test_flush_fsync_failure_keeps_the_previous_canonical_files(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = PaperLedger(tmp_path, "flush-fsync")
    ledger.record_shadow_equity({"event_time": "2024-01-01", "nav": 1.0})
    first = ledger.flush()
    previous_meta = first["meta"].read_bytes()
    previous_frame = first["shadow_equity"].read_bytes()
    previous_latest = first["latest"].read_bytes()

    def fail_fsync(_fd: int) -> None:
        raise OSError("simulated ledger fsync failure")

    monkeypatch.setattr("quant_fund.paper.ledger.os.fsync", fail_fsync)
    ledger.record_shadow_equity({"event_time": "2024-01-02", "nav": 1.1})
    ledger.set_meta(note="must-not-publish")
    with pytest.raises(OSError, match="simulated ledger fsync failure"):
        ledger.flush()

    assert first["meta"].read_bytes() == previous_meta
    assert first["shadow_equity"].read_bytes() == previous_frame
    assert first["latest"].read_bytes() == previous_latest
    assert b"must-not-publish" not in previous_meta
    assert list(ledger.root.glob(".shadow_equity.parquet.*")) == []
    assert list(ledger.root.glob(".meta.json.*")) == []


def test_flush_of_an_empty_frame_removes_the_published_parquet(tmp_path) -> None:
    ledger = PaperLedger(tmp_path, "flush-empty")
    ledger.record_shadow_equity({"event_time": "2024-01-01", "nav": 1.0})
    published = ledger.flush()
    assert published["shadow_equity"].is_file()

    ledger._shadow_equity.clear()
    again = ledger.flush()

    assert "shadow_equity" not in again
    assert not (ledger.root / "shadow_equity.parquet").exists()
    assert again["meta"].is_file()
    assert json.loads(again["latest"].read_text(encoding="utf-8"))["run_id"] == "flush-empty"


def test_checkpoint_replace_failure_leaves_the_last_resumable_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=1_000.0)
    run.submit(Order(order_id="o1", security_id="SEC_A", side="BUY", quantity=1.0))
    run.apply_fill(
        Fill(
            fill_id="f1",
            order_id="o1",
            security_id="SEC_A",
            quantity=1.0,
            price=10.0,
            fill_time=T0,
        )
    )
    run.mark("SEC_A", 10.0)
    run.checkpoint()
    previous = path.read_bytes()

    run.submit(Order(order_id="o2", security_id="SEC_A", side="SELL", quantity=1.0))

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated checkpoint replace failure")

    monkeypatch.setattr("quant_fund.paper.broker_adapter.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated checkpoint replace failure"):
        run.checkpoint()

    assert path.read_bytes() == previous
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert [row["order_id"] for row in stored["orders"]] == ["o1"]
    _resumed, seam = PaperBrokerAdapter.resume(path)
    assert seam["ok"] is True
    assert seam["nav_seam_parity"] is True
    assert seam["research_only"] is True
    assert seam["live_pnl_claim"] is False
