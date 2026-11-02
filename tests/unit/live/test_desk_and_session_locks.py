"""Live desk and session locks fail closed under contention and stale files."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.session import LiveGateError, _acquire, _release
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config


def _config() -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open(root: Path) -> LiveDesk:
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=_config(),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def test_held_desk_lock_is_not_unlinked(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    first = _open(root)
    try:
        owned = (root / ".lock").read_bytes()
        with pytest.raises(LiveDeskError, match="locked"):
            _open(root)
        assert (root / ".lock").read_bytes() == owned
    finally:
        first.close()
    reopened = _open(root)
    reopened.close()


def test_dead_desk_lock_is_reclaimed(tmp_path: Path) -> None:
    root = tmp_path / "desk"
    root.mkdir()
    done = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
    )
    (root / ".lock").write_text(done.stdout.strip(), encoding="ascii")
    desk = _open(root)
    try:
        assert (root / ".lock").read_text(encoding="ascii").strip() == str(os.getpid())
        assert desk.status()["live_pnl_claim"] is False
    finally:
        desk.close()


def test_session_file_lock_times_out(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    first = path.open("a+b")
    second = path.open("a+b")
    if path.stat().st_size == 0:
        first.write(b"\0")
        first.flush()
    _acquire(first, timeout=1.0)
    try:
        started = time.monotonic()
        with pytest.raises(LiveGateError, match="timed out"):
            _acquire(second, timeout=0.05)
        assert time.monotonic() - started < 1.0
    finally:
        _release(first)
        first.close()
        second.close()
    third = path.open("a+b")
    try:
        _acquire(third, timeout=0.5)
    finally:
        _release(third)
        third.close()
