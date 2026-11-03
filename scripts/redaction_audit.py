#!/usr/bin/env python3
"""Scan logs, metrics, and alert payloads for secrets and PII.

Stdout is one JSON object. Findings name a kind and a path. They do not
include the secret, the address, or the account id. ``drill`` runs the
rotation drill and the soak, then scans what those runs wrote.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}")
_LABEL = re.compile(r'(?i)(?:account_id|account|acct|email|api_key|apikey|token|secret)="([^"]+)"')
_SECRET_NAMES = (
    "QUANT_API_KEY",
    "RESEARCH_API_KEY",
    "DIPCATCHER_ALERT_WEBHOOK_SECRET",
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
)
_MIN = 8


def findings(text: str, needles: tuple[str, ...]) -> list[str]:
    """Kinds of leak in ``text``. The matched text is not returned."""
    found: list[str] = []
    if any(len(needle) >= _MIN and needle in text for needle in needles):
        found.append("needle")
    if _EMAIL.search(text):
        found.append("email")
    if _SSN.search(text):
        found.append("ssn")
    if _BEARER.search(text):
        found.append("bearer")
    if _LABEL.search(text):
        found.append("account_label")
    return found


def needles_from_env() -> tuple[str, ...]:
    """Secret values currently configured. The tuple is for comparison only."""
    from quant_fund.utils.secret_tokens import read_secret_tokens

    found: list[str] = []
    for name in _SECRET_NAMES:
        path = os.environ.get(f"{name}_FILE", "").strip()
        if path:
            found.extend(read_secret_tokens(path) or ())
            continue
        plain = os.environ.get(name, "").strip()
        if plain:
            found.append(plain)
    unique = [item for item in dict.fromkeys(found) if len(item) >= _MIN]
    return tuple(unique)


def scan_files(paths: list[Path], needles: tuple[str, ...]) -> list[dict[str, object]]:
    """Scan existing files. A directory is walked. Missing paths are skipped."""
    hits: list[dict[str, object]] = []
    for path in _files(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            hits.append({"kinds": ["unreadable"], "path": str(path)})
            continue
        kinds = findings(text, needles)
        if kinds:
            hits.append({"kinds": kinds, "path": str(path)})
    return hits


def _files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_file():
            found.append(path)
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            found.append(child)
    return found


def _run(argv: list[str], stdout: Path, stderr: Path) -> int:
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        proc = subprocess.run(argv, cwd=_ROOT, stdout=out, stderr=err, check=False)
    return proc.returncode


def _work_dir() -> Path | None:
    """Writable directory for drill stdout. Refuses anything outside /tmp/dipcatcher."""
    raw = os.environ.get("DIPCATCHER_REDACTION_WORK", "/tmp/dipcatcher-ops14-redaction")
    path = Path(raw)
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(Path("/tmp")) or "dipcatcher" not in resolved.name:
        return None
    if resolved.exists():
        for child in sorted(resolved.rglob("*"), reverse=True):
            if child.is_symlink():
                return None
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def drill(log_dir: Path) -> dict[str, object]:
    """Run rotation and soak, then scan their output and the log directory."""
    work = _work_dir()
    if work is None:
        return {"error": "work", "findings": [], "missing": [], "ok": False}
    rotate_code = _run(
        ["uv", "run", "--quiet", "python", "scripts/prod_rotate.py", "drill"],
        work / "rotate-drill.json",
        work / "rotate-drill.err",
    )
    soak_code = _run(
        ["uv", "run", "--quiet", "python", "scripts/prod_soak.py"],
        work / "soak.out",
        work / "soak.err",
    )
    hits = scan_files([log_dir, work], needles_from_env())
    required = [
        log_dir / "alerts.json",
        log_dir / "metrics-lab-a.prom",
        log_dir / "metrics-research-a.prom",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    ok = rotate_code == 0 and soak_code == 0 and not hits and not missing
    return {
        "findings": hits,
        "missing": missing,
        "ok": ok,
        "rotate_exit": rotate_code,
        "soak_exit": soak_code,
    }


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "drill":
        log_dir = Path(os.environ.get("DIPCATCHER_LOG_DIR", "/tmp/dipcatcher-ops7-logs"))
        report = drill(log_dir)
    elif len(argv) >= 2 and argv[1] == "scan":
        paths = [Path(item) for item in argv[2:]]
        report = {"findings": scan_files(paths, needles_from_env()), "ok": True}
        report["ok"] = report["findings"] == []
    else:
        sys.stderr.write("redaction-audit: drill | scan PATH...\n")
        return 1
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
