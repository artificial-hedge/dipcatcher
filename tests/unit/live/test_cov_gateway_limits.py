"""Fail-closed branches of the live pre-trade limit memory and the arm file.

Both modules are the last line before an order reaches a venue: ``risk_memory``
replays persisted rate/duplicate/floor state into a fresh process, and
``gateway_arm`` decides whether a state directory is armed at all. A malformed
store must refuse rather than arm an empty (unlimited) engine.
"""

from __future__ import annotations

import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_fund.live.config import Venue
from quant_fund.live.errors import LiveGateError
from quant_fund.live.gateway_arm import (
    LIVE_CAPITAL_PHRASE,
    arm_session,
    disarm_session,
    load_hmac_key,
    read_arm,
    zero_audit_head,
)
from quant_fund.live.risk_memory import export_pretrade, restore_pretrade
from quant_fund.pretrade.config import LimitConfig, PretradeConfig, SessionConfig
from quant_fund.pretrade.engine import PretradeEngine

HMAC_KEY = b"unit-test-hmac-key-32b!!!!!"
DIGEST = "a" * 64
NOW = datetime(2024, 1, 3, 15, 0, tzinfo=UTC)


def limit_config(**overrides: Any) -> LimitConfig:
    base: dict[str, Any] = {
        "max_order_notional": 1_000_000.0,
        "max_order_quantity": 10_000.0,
        "price_collar_bps": 100.0,
        "max_position_notional": 5_000_000.0,
        "max_position_quantity": 50_000.0,
        "max_gross_notional": 5_000_000.0,
        "max_net_notional": 5_000_000.0,
        "max_name_concentration": 1.0,
        "max_daily_loss_fraction": 0.2,
        "max_trailing_drawdown_fraction": 0.3,
        "max_orders_per_window": 8,
        "max_messages_per_window": 8,
        "rate_window_ns": 1_000_000_000,
        "duplicate_window_ns": 1_000_000_000,
        "max_reference_age_ns": 60_000_000_000,
        "max_mark_age_ns": 60_000_000_000,
        "pdt_equity_threshold": 25_000.0,
        "pdt_window_sessions": 5,
        "pdt_max_day_trades": 3,
        "settlement_ns": 86_400_000_000_000,
        "qty_tick": 1e-4,
        "price_tick": 1e-4,
        "cash_account": True,
        "restrict_to_settled_cash": True,
        "max_lots_per_symbol": 8,
        "ring_capacity": 16,
        "symbol_capacity": 2,
        "day_trade_capacity": 4,
        "unsettled_capacity": 4,
    }
    base.update(overrides)
    return LimitConfig.model_validate(base)


def engine(**overrides: Any) -> PretradeEngine:
    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )
    return PretradeEngine(
        config,
        hmac_key=HMAC_KEY,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=0,
    )


# ---------------------------------------------------------------- risk_memory


def test_export_restore_round_trip_keeps_the_limits() -> None:
    source = engine()
    source.ensure_symbol("AAPL")
    source.book.session_start = 1_000_000.0
    source.book.peak = 1_200_000.0
    source.book.last_ts = 5
    blob = export_pretrade(source)
    assert blob["session_start"] == pytest.approx(1_000_000.0)
    assert blob["peak"] == pytest.approx(1_200_000.0)
    assert blob["last_ts"] == 5
    assert blob["symbols"] == ["AAPL"]

    target = engine()
    restore_pretrade(target, blob)
    assert target.book.session_start == pytest.approx(1_000_000.0)
    assert target.book.peak == pytest.approx(1_200_000.0)
    assert target.book.last_ts == 5
    assert target.ensure_symbol("AAPL") == 0
    # Floors are recomputed from the restored start/peak, not left at infinity.
    assert target.book.daily_floor == pytest.approx(800_000.0)
    assert target.book.dd_floor == pytest.approx(840_000.0)


def test_restore_refuses_a_blob_that_is_not_an_object() -> None:
    with pytest.raises(LiveGateError, match="memory is malformed"):
        restore_pretrade(engine(), ["AAPL"])


def test_restore_refuses_a_symbol_list_that_is_not_a_list() -> None:
    with pytest.raises(LiveGateError, match="symbols are malformed"):
        restore_pretrade(engine(), {"symbols": "AAPL"})


def test_restore_refuses_a_symbol_that_is_not_a_string() -> None:
    with pytest.raises(LiveGateError, match="symbol could not be restored"):
        restore_pretrade(engine(), {"symbols": [5]})


def test_restore_refuses_more_symbols_than_the_capacity() -> None:
    blob = export_pretrade(engine())
    blob["symbols"] = ["AAPL", "MSFT", "NVDA"]
    with pytest.raises(LiveGateError, match="symbol could not be restored"):
        restore_pretrade(engine(), blob)


@pytest.mark.parametrize("field", ["session_start", "peak", "last_ts"])
def test_restore_refuses_missing_floors(field: str) -> None:
    blob = export_pretrade(engine())
    del blob[field]
    with pytest.raises(LiveGateError, match="floors are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_unparsable_floors() -> None:
    blob = export_pretrade(engine())
    blob["session_start"] = "yesterday"
    with pytest.raises(LiveGateError, match="floors are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_a_ring_that_is_not_a_list() -> None:
    blob = export_pretrade(engine())
    blob["order_ts"] = 3
    with pytest.raises(LiveGateError, match="order timestamps are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_a_ring_of_unparsable_stamps() -> None:
    blob = export_pretrade(engine())
    blob["message_ts"] = ["soon"]
    with pytest.raises(LiveGateError, match="message timestamps are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_a_ring_longer_than_the_ring_capacity() -> None:
    blob = export_pretrade(engine())
    blob["order_ts"] = list(range(17))
    with pytest.raises(LiveGateError, match="order ring does not fit"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_malformed_dup_memory() -> None:
    blob = export_pretrade(engine())
    blob["dupes"] = {}
    with pytest.raises(LiveGateError, match="duplicates are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_dup_entries_that_are_not_pairs() -> None:
    blob = export_pretrade(engine())
    blob["dupes"] = [[1, 2, 3]]
    with pytest.raises(LiveGateError, match="duplicates are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_refuses_unparsable_dup_entries() -> None:
    blob = export_pretrade(engine())
    blob["dupes"] = [[1, "later"]]
    with pytest.raises(LiveGateError, match="duplicates are malformed"):
        restore_pretrade(engine(), blob)


def test_restore_replays_duplicate_fingerprints() -> None:
    blob = export_pretrade(engine())
    blob["dupes"] = [[12345, 1]]
    target = engine()
    restore_pretrade(target, blob)
    assert target.book.r_live == 1
    # The replayed fingerprint is live, so the same pair is a duplicate again.
    assert target.book.r_fp[0] == 12345


# ----------------------------------------------------------------- arms/gate


def test_load_hmac_key_creates_a_32_byte_private_key(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "hmac.bin"
    key = load_hmac_key(path)
    assert len(key) == 32
    assert path.read_bytes() == key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # A second call reuses the existing key instead of rotating it.
    assert load_hmac_key(path) == key


def test_load_hmac_key_refuses_a_truncated_key(tmp_path: Path) -> None:
    path = tmp_path / "hmac.bin"
    path.write_bytes(b"short")
    with pytest.raises(LiveGateError, match="unusable"):
        load_hmac_key(path)


@pytest.mark.parametrize("actor", ["", "   ", "a" * 81, "bad\x01actor"])
def test_arm_session_refuses_an_unusable_actor(tmp_path: Path, actor: str) -> None:
    with pytest.raises(LiveGateError, match="actor must be 1..80 visible characters"):
        arm_session(
            tmp_path / "arm.json",
            venue="alpaca-paper",
            actor=actor,
            pretrade_sha256=DIGEST,
        )


def test_arm_session_refuses_a_missing_digest(tmp_path: Path) -> None:
    with pytest.raises(LiveGateError, match="pretrade digest is missing"):
        arm_session(tmp_path / "arm.json", venue="alpaca-paper", actor="ops", pretrade_sha256="abc")


def test_arm_session_refuses_the_wrong_live_phrase(tmp_path: Path) -> None:
    with pytest.raises(LiveGateError, match="confirm-live-capital"):
        arm_session(
            tmp_path / "arm.json",
            venue="alpaca-live",
            actor="ops",
            pretrade_sha256=DIGEST,
            confirm_live_capital="I authorize live capital",
        )


def test_arm_session_refuses_the_phrase_for_a_paper_venue(tmp_path: Path) -> None:
    with pytest.raises(LiveGateError, match="only used for venue alpaca-live"):
        arm_session(
            tmp_path / "arm.json",
            venue="alpaca-paper",
            actor="ops",
            pretrade_sha256=DIGEST,
            confirm_live_capital=LIVE_CAPITAL_PHRASE,
        )


def test_arm_session_writes_a_private_record(tmp_path: Path) -> None:
    path = tmp_path / "state" / "arm.json"
    payload = arm_session(
        path, venue="alpaca-paper", actor=" ops ", pretrade_sha256=DIGEST, now=NOW
    )
    assert payload["actor"] == "ops"
    assert payload["venue"] == "alpaca-paper"
    assert payload["armed_at"] == NOW.isoformat()
    assert payload["live_capital_confirmed"] is False
    assert payload["live_pnl_claim"] is False
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert read_arm(path, venue="alpaca-paper", pretrade_sha256=DIGEST) == payload


def test_arm_session_confirms_live_capital_with_the_exact_phrase(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    payload = arm_session(
        path,
        venue="alpaca-live",
        actor="ops",
        pretrade_sha256=DIGEST,
        confirm_live_capital=LIVE_CAPITAL_PHRASE,
    )
    assert payload["live_capital_confirmed"] is True
    assert read_arm(path, venue="alpaca-live", pretrade_sha256=DIGEST) == payload


def test_disarm_is_idempotent_and_leaves_positions_alone(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    first = disarm_session(path, actor="ops")
    assert first["armed"] is False
    assert first["positions_unchanged"] is True
    arm_session(path, venue="alpaca-paper", actor="ops", pretrade_sha256=DIGEST)
    disarm_session(path, actor="ops")
    assert not path.exists()
    with pytest.raises(LiveGateError, match="actor must be"):
        disarm_session(path, actor="")


def test_read_arm_returns_none_when_the_directory_is_not_armed(tmp_path: Path) -> None:
    assert read_arm(tmp_path / "missing.json", venue="alpaca-paper", pretrade_sha256=DIGEST) is None


def write_arm(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_arm(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "live_arm.v1",
        "venue": "alpaca-paper",
        "actor": "ops",
        "armed_at": NOW.isoformat(),
        "pretrade_sha256": DIGEST,
        "live_capital_confirmed": False,
        "live_pnl_claim": False,
        "research_evidence": False,
    }
    base.update(overrides)
    return base


def test_read_arm_refuses_a_file_that_is_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    write_arm(path, [])
    with pytest.raises(LiveGateError, match="is not a JSON object"):
        read_arm(path, venue="alpaca-paper", pretrade_sha256=DIGEST)


def test_read_arm_refuses_an_unsupported_schema(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    write_arm(path, valid_arm(schema="live_arm.v0"))
    with pytest.raises(LiveGateError, match="schema is not supported"):
        read_arm(path, venue="alpaca-paper", pretrade_sha256=DIGEST)


@pytest.mark.parametrize(
    "overrides",
    [
        {"live_pnl_claim": True},
        {"research_evidence": True},
        {"venue": "loopback"},
        {"pretrade_sha256": "b" * 64},
        {"actor": ""},
        {"actor": 5},
    ],
)
def test_read_arm_refuses_a_record_it_must_not_honor(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    path = tmp_path / "arm.json"
    write_arm(path, valid_arm(**overrides))
    with pytest.raises(LiveGateError):
        read_arm(path, venue="alpaca-paper", pretrade_sha256=DIGEST)


def test_read_arm_requires_live_confirmation_for_the_live_venue(tmp_path: Path) -> None:
    path = tmp_path / "arm.json"
    write_arm(path, valid_arm(venue="alpaca-live", live_capital_confirmed=False))
    with pytest.raises(LiveGateError, match="live-capital confirmation"):
        read_arm(path, venue=cast_venue("alpaca-live"), pretrade_sha256=DIGEST)


def cast_venue(value: str) -> Venue:
    return value  # type: ignore[return-value]


def test_zero_audit_head_is_the_empty_chain() -> None:
    assert zero_audit_head() == "0" * 64
