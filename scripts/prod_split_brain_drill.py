#!/usr/bin/env python3
"""Two paper processes cannot both submit, and a killed holder is fenced.

The lock file stays on disk. Recovery takes the flock and increments the
token. A recorded pid that is already dead is not a reason to delete the
file. Stdout of the drill is one JSON object. Research and simulated paper
only.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = Path(__file__).resolve()
_WORK = Path("/tmp/dipcatcher-ops17-split")


def _env() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "DIPCATCHER_PAPER_BROKER_URL",
        "DIPCATCHER_PAPER_BROKER_CA",
        "DIPCATCHER_PAPER_BROKER_TIMEOUT",
        "DIPCATCHER_REPLACE_HOLD",
        "DIPCATCHER_REPLACE_HOLD_SKIP",
    ):
        env.pop(name, None)
    return env


def _spawn(args: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_ROOT,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _readline(proc: subprocess.Popen[str], timeout: float) -> str:
    if proc.stdout is None:
        return ""
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if selector.select(0.2):
                return proc.stdout.readline()
            if proc.poll() is not None:
                return proc.stdout.readline()
        return ""
    finally:
        selector.close()


def _reap(proc: subprocess.Popen[str] | None) -> str:
    if proc is None:
        return ""
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    try:
        _out, err = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        return ""
    return err or ""


def _hold(state: Path) -> int:
    from quant_fund.paper.order_lease import acquire_order_lease

    token = acquire_order_lease(state)
    sys.stdout.write(json.dumps({"pid": os.getpid(), "token": token}) + "\n")
    sys.stdout.flush()
    while True:
        time.sleep(3600)
    return 0


def _submit(state: Path, prefix: str) -> int:
    from quant_fund.api.lifecycle import acceptance_block
    from quant_fund.observe.metrics import render_prometheus
    from quant_fund.paper.durable_book import run_orders
    from quant_fund.paper.order_lease import SplitBrainError

    try:
        code = run_orders(state, 1, prefix)
    except SplitBrainError:
        payload = {
            "acceptance": acceptance_block(),
            "error": "split_brain",
            "metrics": render_prometheus(),
            "ok": False,
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
        return 2
    sys.stdout.write(json.dumps({"exit": code, "ok": code == 0}) + "\n")
    sys.stdout.flush()
    return 0 if code == 0 else 1


def _race(state: Path, prefix: str) -> int:
    from quant_fund.paper.durable_book import run_orders
    from quant_fund.paper.order_lease import SplitBrainError, acquire_order_lease

    try:
        token = acquire_order_lease(state)
    except SplitBrainError:
        sys.stdout.write(json.dumps({"error": "split_brain", "ok": False, "prefix": prefix}) + "\n")
        sys.stdout.flush()
        return 2
    code = run_orders(state, 1, prefix)
    sys.stdout.write(
        json.dumps({"exit": code, "ok": code == 0, "prefix": prefix, "token": token}) + "\n"
    )
    sys.stdout.flush()
    release = state / "release-race"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if release.exists():
            return 0 if code == 0 else 1
        time.sleep(0.05)
    return 0 if code == 0 else 1


def _once(state: Path, order_id: str) -> int:
    from quant_fund.paper.durable_book import open_book, submit_once

    result = submit_once(open_book(state), state, order_id)
    sys.stdout.write(result + "\n")
    sys.stdout.flush()
    return 0 if result == "duplicate" else 1


def _load(line: str) -> dict[str, object]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _fail(error: str, detail: str) -> dict[str, object]:
    return {"detail": detail[-400:], "error": error, "ok": False}


def execute() -> dict[str, object]:
    from quant_fund.paper.durable_book import ledger_order_count, ledger_path
    from quant_fund.paper.order_lease import lock_path

    if _WORK.exists():
        shutil.rmtree(_WORK)
    state = _WORK / "book"
    state.mkdir(parents=True)
    holder: subprocess.Popen[str] | None = None
    blocked: subprocess.Popen[str] | None = None
    first: subprocess.Popen[str] | None = None
    second: subprocess.Popen[str] | None = None
    replay: subprocess.Popen[str] | None = None
    try:
        holder = _spawn(["hold", str(state)])
        held = _load(_readline(holder, 90))
        if held.get("token") != 1:
            return _fail("no_holder", _reap(holder))
        holder_pid = held.get("pid")
        blocked = _spawn(["submit", str(state), "blocked"])
        blocked_line = _load(_readline(blocked, 90))
        blocked_code = blocked.wait(timeout=30)
        if blocked_code != 2 or blocked_line.get("error") != "split_brain":
            return _fail("both_submitted", _reap(blocked) + _reap(holder))
        os.kill(holder.pid, signal.SIGKILL)
        killed = holder.wait(timeout=30)
        lock = lock_path(state)
        stale = _load(lock.read_text(encoding="utf-8"))
        inode = lock.stat().st_ino
        if killed != -signal.SIGKILL or not lock.is_file() or stale.get("pid") != holder_pid:
            return _fail("stale_lock", _reap(holder))
        if stale.get("token") != 1:
            return _fail("stale_token", "")
        first = _spawn(["race", str(state), "reca"])
        second = _spawn(["race", str(state), "recb"])
        left = _load(_readline(first, 90))
        right = _load(_readline(second, 90))
        winners = [row for row in (left, right) if row.get("ok") is True]
        losers = [row for row in (left, right) if row.get("ok") is not True]
        if len(winners) != 1 or len(losers) != 1:
            return _fail("race", json.dumps({"left": left, "right": right}))
        winner = winners[0]
        loser = losers[0]
        if lock.stat().st_ino != inode or winner.get("token") != 2:
            return _fail("fence", json.dumps(winner))
        (state / "release-race").write_text("1\n", encoding="utf-8")
        first.wait(timeout=30)
        second.wait(timeout=30)
        prefix = winner.get("prefix")
        order_id = f"{prefix}-0"
        path = ledger_path(state)
        other = loser.get("prefix")
        count = ledger_order_count(path, order_id)
        other_count = ledger_order_count(path, f"{other}-0")
        blocked_orders = ledger_order_count(path, "blocked-0")
        replay = _spawn(["once", str(state), order_id])
        duplicate = _readline(replay, 90).strip()
        replay_code = replay.wait(timeout=30)
        again = ledger_order_count(path, order_id)
        metrics = blocked_line.get("metrics")
        gauge = isinstance(metrics, str) and "dipcatcher_split_brain 1" in metrics
        ok = (
            blocked_line.get("acceptance") == "split_brain"
            and gauge
            and count == 1
            and other_count == 0
            and duplicate == "duplicate"
            and again == 1
            and replay_code == 0
            and loser.get("error") == "split_brain"
            and blocked_orders == 0
        )
        return {
            "acceptance": blocked_line.get("acceptance"),
            "blocked_orders": blocked_orders,
            "both_submitted": False,
            "detail": "",
            "duplicate": duplicate,
            "error": "" if ok else "orders",
            "gauge": gauge,
            "killed": killed,
            "ledger_count": again,
            "lock_inode_same": lock.stat().st_ino == inode,
            "lock_present_after_kill": True,
            "loser_orders": other_count,
            "loser_prefix": other,
            "ok": ok,
            "stale_pid": holder_pid,
            "stale_token": stale.get("token"),
            "winner_prefix": prefix,
            "winner_token": winner.get("token"),
        }
    finally:
        for proc in (replay, second, first, blocked, holder):
            _reap(proc)


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "hold":
        return _hold(Path(argv[2]))
    if len(argv) == 4 and argv[1] == "submit":
        return _submit(Path(argv[2]), argv[3])
    if len(argv) == 4 and argv[1] == "race":
        return _race(Path(argv[2]), argv[3])
    if len(argv) == 4 and argv[1] == "once":
        return _once(Path(argv[2]), argv[3])
    if len(argv) != 1:
        sys.stderr.write("prod-split-brain-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
