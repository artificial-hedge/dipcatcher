#!/usr/bin/env python3
"""Seed N, bump the config to N+1, and downgrade without touching state bytes.

The ledger is a paper checkpoint. The rate store and audit log are copied
from one loopback submit. The journal is the synthetic order line. Digests
cover those four files and not ``config.json``. A newer state schema fails
closed. Stdout is one JSON object. This is simulated paper.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_fund.live.carry import content_digests, downgrade, upgrade
from quant_fund.live.config import LiveConfig
from quant_fund.live.session import open_session
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.paper.trading_session import PaperTradingSession

_ROOT = Path(__file__).resolve().parents[1]
_PRETRADE = _ROOT / "configs" / "pretrade_risk.yaml"
_TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
_FILES = (
    "ledger/broker_state.json",
    "journal/events.jsonl",
    "rate/window.json",
    "audit/audit.jsonl",
)


def _workspace() -> Path | None:
    raw = os.environ.get("DIPCATCHER_MIGRATE_DIR", "/tmp/dipcatcher-ops15-migrate")
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


def _seed(root: Path) -> str:
    for name in ("ledger", "journal", "rate", "audit", "loopback"):
        (root / name).mkdir()
    paper = root / "ledger" / "broker_state.json"
    book = PaperTradingSession.open(
        paper, resume=False, starting_cash=1_000_000.0, endpoint_kind="paper"
    )
    marked = book.apply_command({"op": "mark", "security_id": "AAPL", "price": 100.0})
    accepted = book.apply_command(
        {
            "op": "submit",
            "order_id": "carry-1",
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    )
    if marked.get("verdict") != "marked" or accepted.get("verdict") != "accepted":
        return "paper"
    (root / "journal" / "events.jsonl").write_text(
        '{"data_label":"SYNTHETIC","event":"submit","schema":"journal.v1"}\n',
        encoding="utf-8",
    )
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
    result = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="carry1",
        at_ns=_TS,
    )
    if not result.get("ok"):
        return "loopback"
    stored = json.loads((root / "loopback" / "state.json").read_text(encoding="utf-8"))
    pretrade = stored.get("pretrade")
    if not isinstance(pretrade, dict):
        return "rate"
    (root / "rate" / "window.json").write_text(
        json.dumps(pretrade, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "audit" / "audit.jsonl").write_bytes((root / "loopback" / "audit.jsonl").read_bytes())
    (root / "config.json").write_text(
        json.dumps({"config_schema": 1, "order_path": "paper", "state_schema": 1}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return ""


def _future_state_refused(root: Path, before: dict[str, str]) -> bool:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config["state_schema"] = 99
    (root / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    report = downgrade(root)
    after = content_digests(root)
    (root / "config.json").write_text(
        json.dumps({"config_schema": 1, "order_path": "paper", "state_schema": 1}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    message = str(report.get("error") or "")
    return (not report.get("ok")) and "newer than this process" in message and after == before


def _resume_ok(root: Path) -> bool:
    try:
        _adapter, seam = PaperBrokerAdapter.resume(root / "ledger" / "broker_state.json")
    except (OSError, ValueError):
        return False
    return bool(seam.get("ok"))


def execute() -> dict[str, object]:
    """Seed N, upgrade, downgrade, and refuse a newer state schema."""
    root = _workspace()
    if root is None:
        return {"error": "path", "ok": False}
    seeded = _seed(root)
    if seeded:
        return {"error": seeded, "ok": False}
    before = content_digests(root)
    if len(before) != len(_FILES) or any(not item for item in before.values()):
        return {"error": "seed", "ok": False}
    up = upgrade(root)
    mid = content_digests(root)
    down = downgrade(root)
    after = content_digests(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    carried = before == mid == after
    closed = _future_state_refused(root, before)
    resumed = _resume_ok(root)
    ok = (
        up.get("ok") is True
        and down.get("ok") is True
        and carried
        and closed
        and resumed
        and config.get("config_schema") == 1
        and "rate_limit_rps" not in config
    )
    return {
        "digests_unchanged": carried,
        "error": "" if ok else "migrate",
        "fail_closed": closed,
        "ok": ok,
        "resume_ok": resumed,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-migrate-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
