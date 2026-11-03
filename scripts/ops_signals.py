#!/usr/bin/env python3
"""Write operator gauges for the node-exporter textfile collector.

The file is ``dipcatcher.prom``. A missing directory is skipped by the
callers. This process does not stamp epochs or sign pins. Reconcile reads
the same six corpora as ``owner_restamp`` and writes 1 when any tree moved.
``refresh`` republishes reconcile plus the backup and canary sidecars so
the series stay present.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_GAUGES = (
    "dipcatcher_reconcile_drift",
    "dipcatcher_backup_last_success_timestamp",
    "dipcatcher_canary_abort",
)
_ROOTS = ("/tmp/dipcatcher", "/var/lib/dipcatcher", "/var/log/dipcatcher")


def signals_dir(path: Path) -> Path | None:
    """A dipcatcher directory, or ``None`` when the path is outside that set."""
    if path.is_symlink():
        return None
    try:
        resolved = path.resolve()
    except OSError:
        return None
    text = str(resolved)
    if resolved.is_symlink():
        return None
    if any(
        text == root or text.startswith(root + "/") or text.startswith(root + "-")
        for root in _ROOTS
    ):
        return resolved
    return None


def render_signals(gauges: dict[str, int]) -> str:
    """Prometheus text for the known gauges that are present."""
    lines = [f"{name} {gauges[name]}" for name in _GAUGES if name in gauges]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def read_gauges(path: Path) -> dict[str, int]:
    """Parse a prom file. Unknown names and malformed lines are dropped."""
    if path.is_symlink() or not path.is_file():
        return {}
    found: dict[str, int] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw = line.partition(" ")
        if separator != " " or name not in _GAUGES or not raw.isdecimal():
            continue
        found[name] = int(raw)
    return found


def write_prom(directory: Path, updates: dict[str, int]) -> str | None:
    """Merge ``updates`` into ``dipcatcher.prom``. Return an error code or ``None``."""
    located = signals_dir(directory)
    if located is None:
        return "path"
    for name, value in updates.items():
        if name not in _GAUGES or value < 0:
            return "gauge"
    try:
        located.mkdir(parents=True, exist_ok=True)
    except OSError:
        return "io"
    target = located / "dipcatcher.prom"
    if target.is_symlink():
        return "symlink"
    merged = read_gauges(target)
    merged.update(updates)
    body = render_signals(merged).encode("utf-8")
    temporary = located / f".dipcatcher.prom.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(temporary, flags, 0o644)
    except OSError:
        return "io"
    try:
        os.write(descriptor, body)
        os.fsync(descriptor)
    except OSError:
        os.close(descriptor)
        return "io"
    os.close(descriptor)
    try:
        os.replace(temporary, target)
    except OSError:
        return "io"
    return None


def read_state(directory: Path, name: str) -> int | None:
    """An integer sidecar next to the prom file, or ``None``."""
    located = signals_dir(directory)
    if located is None:
        return None
    path = located / name
    if path.is_symlink() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not text.isdecimal():
        return None
    return int(text)


def write_state(directory: Path, name: str, value: int) -> None:
    """Remember one integer so a later refresh republishes the same gauge."""
    located = signals_dir(directory)
    if located is None or value < 0:
        return
    path = located / name
    if path.is_symlink():
        return
    temporary = located / f".{name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(f"{value}\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        return


def refresh_gauges(directory: Path, drifted: list[str]) -> dict[str, int]:
    """Gauges a timer republishes. A missing backup stamp stays 0."""
    backup = read_state(directory, "backup.stamp")
    canary = read_state(directory, "canary.state")
    return {
        "dipcatcher_reconcile_drift": reconcile_gauge(drifted),
        "dipcatcher_backup_last_success_timestamp": 0 if backup is None else backup,
        "dipcatcher_canary_abort": 1 if canary == 1 else 0,
    }


def publish(directory: str, updates: dict[str, int]) -> None:
    """Write when ``directory`` is set. A refused path is ignored."""
    raw = directory.strip()
    if not raw:
        return
    if write_prom(Path(raw), updates) is not None:
        return
    if "dipcatcher_backup_last_success_timestamp" in updates:
        write_state(Path(raw), "backup.stamp", updates["dipcatcher_backup_last_success_timestamp"])
    if "dipcatcher_canary_abort" in updates:
        write_state(Path(raw), "canary.state", updates["dipcatcher_canary_abort"])


def reconcile_gauge(drifted: list[str]) -> int:
    """1 when any corpus drifted, else 0."""
    return 1 if drifted else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Publish operator textfile gauges")
    parser.add_argument("action", choices=("reconcile", "backup", "canary", "refresh"))
    parser.add_argument("--dir", default=os.environ.get("DIPCATCHER_TEXTFILE_DIR", ""))
    parser.add_argument("--canary", choices=("0", "1"), default="0")
    parser.add_argument("--backup-unix", type=int, default=0)
    args = parser.parse_args(argv[1:])
    if not args.dir:
        sys.stderr.write("DIPCATCHER_TEXTFILE_DIR is unset\n")
        return 1
    if args.action in {"reconcile", "refresh"}:
        from owner_restamp import drift_report, live_trees  # noqa: PLC0415

        from quant_fund.research.corpus_epoch import load_heads_pin  # noqa: PLC0415

        root = Path(__file__).resolve().parents[1]
        report = drift_report(
            load_heads_pin(root / "quality" / "epoch_heads.json"), live_trees(root)
        )
        drifted = report.get("drifted")
        names = drifted if isinstance(drifted, list) else []
        if args.action == "refresh":
            updates = refresh_gauges(Path(args.dir), [str(item) for item in names])
        else:
            updates = {"dipcatcher_reconcile_drift": reconcile_gauge([str(item) for item in names])}
    elif args.action == "backup":
        stamp = args.backup_unix or int(time.time())
        updates = {"dipcatcher_backup_last_success_timestamp": stamp}
    else:
        updates = {"dipcatcher_canary_abort": int(args.canary)}
    error = write_prom(Path(args.dir), updates)
    sys.stdout.write(json.dumps({"ok": error is None, "error": error or ""}, sort_keys=True) + "\n")
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
