#!/usr/bin/env python3
"""Prove DipcatcherFilesystemReadOnly pages and DiskPressure stays quiet.

``promtool test rules`` evaluates ``deploy/observability/fs_readonly_test.yml``
against ``alerts.yml``. A read-only ext4 mount pages the new alert. The same
series does not page DipcatcherDiskPressure, because that rule requires
``node_filesystem_readonly == 0``. Stdout is one JSON object.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RULES = _ROOT / "deploy" / "observability"


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def execute() -> dict[str, object]:
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
            f"{_RULES}:/rules:ro",
            "-w",
            "/rules",
            "prom/prometheus:v2.54.1",
            "test",
            "rules",
            "fs_readonly_test.yml",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    detail = ((proc.stdout or "") + (proc.stderr or ""))[-1500:]
    return {
        "error": "" if proc.returncode == 0 else detail,
        "ok": proc.returncode == 0,
        "promtool_exit": proc.returncode,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-fs-readonly takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
