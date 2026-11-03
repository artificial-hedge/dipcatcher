#!/usr/bin/env python3
"""Drive the paper book and the loopback broker at 10x the lab rate.

``DIPCATCHER_RATE_LIMIT_RPS`` in ``deploy/prod.env`` is the expected rate.
The paper session must accept that many pre-trade decisions per second,
refuse notional breaches, and refuse every order after the kill file trips.
The loopback session must keep the signed pre-trade window and the kill
latch. This is simulated paper. It is not a live-trading claim. Stdout is
one JSON object and it does not include an order id or a secret.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_fund.live.config import LiveConfig
from quant_fund.live.session import open_session
from quant_fund.paper.trading_session import PaperTradingSession, SessionLimits
from quant_fund.utils.operator_kill import ENV_PATH, HALT_NEW_ORDERS, write_operator_kill

_ROOT = Path(__file__).resolve().parents[1]
_PRETRADE = _ROOT / "configs" / "pretrade_risk.yaml"
_ENV = _ROOT / "deploy" / "prod.env"
_MULTIPLE = 10
_TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def expected_rps(text: str) -> int:
    """The production request rate, or 0 when the file does not set one."""
    for line in text.splitlines():
        if line.startswith("DIPCATCHER_RATE_LIMIT_RPS="):
            raw = line.split("=", 1)[1].strip()
            if raw.isdecimal():
                return int(raw)
    return 0


def _paper_burst(root: Path, attempts: int) -> dict[str, object]:
    limits = SessionLimits(
        max_order_quantity=100.0,
        max_order_notional=1_000.0,
        max_position_notional=50_000.0,
        max_gross_notional=50_000.0,
    )
    book = PaperTradingSession.open(
        root / "paper-broker.json",
        resume=False,
        starting_cash=1_000_000.0,
        limits=limits,
        endpoint_kind="paper",
    )
    marked = book.apply_command({"op": "mark", "security_id": "AAPL", "price": 100.0})
    blocked = book.apply_command(
        {
            "op": "submit",
            "order_id": "notional",
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 100.0,
        }
    )
    started = time.perf_counter()
    accepted = 0
    for index in range(attempts):
        verdict = book.apply_command(
            {
                "op": "submit",
                "order_id": f"rate-{index}",
                "security_id": "AAPL",
                "side": "BUY",
                "quantity": 100.0,
            }
        )
        if verdict.get("verdict") == "accepted":
            accepted += 1
    elapsed = time.perf_counter() - started
    kill = root / "kill_switch"
    write_operator_kill(kill, HALT_NEW_ORDERS)
    os.environ[ENV_PATH] = str(kill)
    kill_accepted = 0
    try:
        for index in range(attempts):
            verdict = book.apply_command(
                {
                    "op": "submit",
                    "order_id": f"kill-{index}",
                    "security_id": "AAPL",
                    "side": "BUY",
                    "quantity": 1.0,
                }
            )
            if verdict.get("verdict") == "accepted":
                kill_accepted += 1
    finally:
        os.environ.pop(ENV_PATH, None)
    rate = 0.0 if elapsed <= 0.0 else attempts / elapsed
    return {
        "accepted_over_notional": accepted,
        "elapsed_s": round(elapsed, 6),
        "kill_accepted": kill_accepted,
        "mark_ok": marked.get("verdict") == "marked",
        "notional_reason": blocked.get("reason"),
        "rps": round(rate, 3),
    }


def _loopback_burst(root: Path, attempts: int) -> dict[str, object]:
    clock = {"now": _TS}

    def _now() -> int:
        return clock["now"]

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
    session = open_session(config, _PRETRADE, state_dir=root / "loopback", now_ns=_now)
    session.arm(actor="ada")
    notional = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=10_000.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="notional",
        at_ns=_TS,
    )
    allowed = 0
    rate_denied = 0
    claims = 0
    for index in range(attempts):
        clock["now"] = _TS + (index % 50) * 1_000_000
        result = session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0 + index * 0.0001,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
            client_order_id=f"lb{index:05d}",
            at_ns=clock["now"],
        )
        if result.get("live_pnl_claim") or result.get("research_evidence"):
            claims += 1
        if result.get("ok"):
            allowed += 1
        elif "order_rate" in result.get("reasons", []):
            rate_denied += 1
    position = float(session.broker.account().positions.get("AAPL", 0.0))
    session.kill(actor="ada", reason="load halt", at_ns=_TS)
    kill_allowed = 0
    for index in range(attempts):
        result = session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
            client_order_id=f"halt{index:05d}",
            at_ns=_TS + 2_000_000_000,
        )
        if result.get("ok"):
            kill_allowed += 1
    after = float(session.broker.account().positions.get("AAPL", 0.0))
    reasons = list(notional.get("reasons", []))
    return {
        "allowed": allowed,
        "claims": claims,
        "kill_allowed": kill_allowed,
        "notional_blocked": (not notional.get("ok")) and "max_order_notional" in reasons,
        "position_held": after == position,
        "rate_denied": rate_denied,
    }


def execute(root: Path | None = None) -> dict[str, object]:
    """Run both bursts. ``ok`` requires the paper path at 10x and both gates."""
    rate = expected_rps(_ENV.read_text(encoding="utf-8"))
    target = rate * _MULTIPLE
    if rate <= 0 or target <= 0:
        return {"error": "rate", "ok": False}
    directory = root or Path(
        os.environ.get("DIPCATCHER_ORDER_LOAD_DIR", "/tmp/dipcatcher-ops14-orders")
    )
    if not str(directory.resolve()).startswith("/tmp/dipcatcher"):
        return {"error": "path", "ok": False}
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "paper").mkdir(parents=True)
    (directory / "loopback").mkdir(parents=True)
    paper = _paper_burst(directory / "paper", target)
    loopback = _loopback_burst(directory / "loopback", target)
    paper_ok = (
        paper["mark_ok"]
        and paper["notional_reason"] == "max_order_notional"
        and paper["accepted_over_notional"] == 0
        and paper["kill_accepted"] == 0
        and float(paper["rps"]) >= target
    )
    loop_ok = (
        loopback["notional_blocked"]
        and loopback["allowed"] <= 20
        and loopback["rate_denied"] >= target - 20
        and loopback["kill_allowed"] == 0
        and loopback["position_held"]
        and loopback["claims"] == 0
    )
    return {
        "evidence": "simulated-paper",
        "expected_rps": rate,
        "live_pnl_claim": False,
        "loopback": loopback,
        "multiple": _MULTIPLE,
        "ok": paper_ok and loop_ok,
        "paper": paper,
        "target_rps": target,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-order-load takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
