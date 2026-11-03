#!/usr/bin/env python3
"""Step the wall clock and check freshness, the kill file, and the rate window.

Monotonic time stays put, so the HTTP bucket does not refill and the
pre-trade window does not flush. A backward step reports the stale SLO
instead of a negative age. The kill file stays latched. Promtool evaluates
``DipcatcherDataStale`` and ``DipcatcherKillSwitchBlocking``. Stdout is one
JSON object. This is simulated paper.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import yaml

from quant_fund.api.rate_limit import BUCKET, configured_rps, reset_bucket
from quant_fund.live.clock_guard import STALE_SECONDS, ClockGuard, freshness_seconds
from quant_fund.live.config import LiveConfig
from quant_fund.live.session import open_session
from quant_fund.utils.operator_kill import (
    ENV_PATH,
    HALT_NEW_ORDERS,
    orders_blocked,
    write_operator_kill,
)

_ROOT = Path(__file__).resolve().parents[1]
_PRETRADE = _ROOT / "configs" / "pretrade_risk.yaml"
_ALERTS = _ROOT / "deploy" / "observability" / "alerts.yml"
_DAY_NS = 86_400 * 1_000_000_000
_TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _workspace() -> Path | None:
    raw = os.environ.get("DIPCATCHER_TIME_DRILL_DIR", "/tmp/dipcatcher-ops15-time")
    path = Path(raw)
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(Path("/tmp")) or "dipcatcher" not in resolved.name:
        return None
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)
    return resolved


def _http_bucket() -> bool:
    """A wall jump with a frozen monotonic clock does not refill the bucket."""
    clock = {"mono": 1_000.0}
    saved_rate = os.environ.get("DIPCATCHER_RATE_LIMIT_RPS")
    saved_profile = os.environ.get("DIPCATCHER_PROFILE")
    os.environ["DIPCATCHER_RATE_LIMIT_RPS"] = "1"
    os.environ.pop("DIPCATCHER_PROFILE", None)
    reset_bucket()
    try:
        with mock.patch("quant_fund.api.rate_limit.time.monotonic", lambda: clock["mono"]):
            first, _retry = BUCKET.take()
            second, _retry = BUCKET.take()
            with mock.patch("quant_fund.api.rate_limit.time.time", lambda: 9_000_000_000.0):
                third, _retry = BUCKET.take()
        return first and not second and not third and configured_rps() == 1.0
    finally:
        _restore("DIPCATCHER_RATE_LIMIT_RPS", saved_rate)
        _restore("DIPCATCHER_PROFILE", saved_profile)
        reset_bucket()


def _restore(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _rate_window(root: Path) -> tuple[bool, bool, float]:
    """Fill the signed window, then step the wall clock forward and backward."""
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "loopback",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(_PRETRADE),
            "state_dir": str(root / "loopback"),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(config, _PRETRADE, state_dir=root / "loopback", now_ns=lambda: _TS)
    session.arm(actor="ada")
    for index in range(20):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0 + index * 0.0001,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
            client_order_id=f"t{index:02d}",
            at_ns=_TS,
        )
    guard = ClockGuard()
    guard.order_ns(_TS, 0)
    forward_ts = guard.order_ns(_TS + _DAY_NS, 1_000_000)
    forward = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="forward",
        at_ns=forward_ts,
    )
    back = ClockGuard()
    back.order_ns(_TS, 0)
    back.order_ns(_TS - _DAY_NS, 1_000_000)
    age = freshness_seconds(
        (_TS - _DAY_NS) / 1_000_000_000, _TS / 1_000_000_000, backward=back.backward
    )
    forward_age = freshness_seconds(
        (_TS + _DAY_NS) / 1_000_000_000, _TS / 1_000_000_000, backward=False
    )
    denied = (not forward.get("ok")) and "order_rate" in forward.get("reasons", [])
    return denied and forward_ts < _TS + _DAY_NS, back.backward, max(age, forward_age)


def _kill_holds(root: Path) -> bool:
    path = root / "kill_switch"
    saved_profile = os.environ.get("DIPCATCHER_PROFILE")
    saved_owner = os.environ.get("DIPCATCHER_WRITE_OWNER")
    os.environ.pop("DIPCATCHER_PROFILE", None)
    os.environ.pop("DIPCATCHER_WRITE_OWNER", None)
    try:
        write_operator_kill(path, HALT_NEW_ORDERS)
    finally:
        _restore("DIPCATCHER_PROFILE", saved_profile)
        _restore("DIPCATCHER_WRITE_OWNER", saved_owner)
    before = orders_blocked(path)
    guard = ClockGuard()
    guard.order_ns(_TS, 0)
    guard.order_ns(_TS + _DAY_NS, 1_000_000)
    guard.order_ns(_TS - _DAY_NS, 2_000_000)
    after = orders_blocked(path)
    return before == f"kill_switch:{HALT_NEW_ORDERS}" and after == before


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def _promtool(work: Path) -> int:
    document = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    wanted = {"DipcatcherDataStale", "DipcatcherKillSwitchBlocking"}
    notes: dict[str, dict[str, str]] = {}
    for group in document["groups"]:
        for rule in group["rules"]:
            name = rule.get("alert")
            if name in wanted:
                notes[str(name)] = {
                    str(key): str(value) for key, value in (rule.get("annotations") or {}).items()
                }
    if set(notes) != wanted:
        return 1
    case = {
        "rule_files": ["alerts.yml"],
        "evaluation_interval": "1m",
        "tests": [
            {
                "interval": "1m",
                "input_series": [
                    {"series": "dipcatcher_data_freshness_seconds", "values": "86400+0x30"},
                    {"series": "dipcatcher_kill_switch_blocking", "values": "1+0x30"},
                ],
                "alert_rule_test": [
                    {
                        "eval_time": "10m",
                        "alertname": name,
                        "exp_alerts": [
                            {
                                "exp_labels": {"severity": "page"},
                                "exp_annotations": notes[name],
                            }
                        ],
                    }
                    for name in sorted(wanted)
                ],
            }
        ],
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "time_drill.yml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    shutil.copyfile(_ALERTS, work / "alerts.yml")
    proc = subprocess.run(
        [
            *_docker(),
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "promtool",
            "-v",
            f"{work}:/rules:ro",
            "-w",
            "/rules",
            "prom/prometheus:v2.54.1",
            "test",
            "rules",
            "time_drill.yml",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return proc.returncode


def execute() -> dict[str, object]:
    """Run the wall-clock steps and the two alert rules."""
    root = _workspace()
    if root is None:
        return {"error": "path", "ok": False}
    saved = os.environ.get(ENV_PATH)
    os.environ.pop(ENV_PATH, None)
    try:
        http_ok = _http_bucket()
        rate_ok, backward, age = _rate_window(root)
        kill_ok = _kill_holds(root)
        prom = _promtool(root / "rules")
        ok = http_ok and rate_ok and backward and kill_ok and age >= STALE_SECONDS and prom == 0
        return {
            "backward": backward,
            "error": "" if ok else "time",
            "freshness_seconds": age,
            "http_held": http_ok,
            "kill_held": kill_ok,
            "ok": ok,
            "promtool_exit": prom,
            "rate_held": rate_ok,
        }
    finally:
        if saved is None:
            os.environ.pop(ENV_PATH, None)
        else:
            os.environ[ENV_PATH] = saved


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-time-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
