#!/usr/bin/env python3
"""Publish the operator gauges and prove the three alerts fire.

The writers are the same functions the backup, canary, and refresh jobs
use. ``promtool test rules`` evaluates ``deploy/observability/alerts.yml``
against that text. Stdout is one JSON object. No lab stack is started.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_ALERTS = _ROOT / "deploy" / "observability" / "alerts.yml"
_NAMES = (
    "DipcatcherReconcileDrift",
    "DipcatcherBackupAge",
    "DipcatcherCanaryAbort",
)
_GAUGE = {
    "DipcatcherReconcileDrift": "dipcatcher_reconcile_drift",
    "DipcatcherBackupAge": "dipcatcher_backup_last_success_timestamp",
    "DipcatcherCanaryAbort": "dipcatcher_canary_abort",
}


def _signals():
    path = _ROOT / "scripts" / "ops_signals.py"
    name = "dipcatcher_ops_signals_drill"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ops_signals")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def publish_faults(directory: Path) -> dict[str, int]:
    """Write the fault values through the job publisher and read them back."""
    module = _signals()
    module.publish(
        str(directory),
        {
            "dipcatcher_reconcile_drift": 1,
            "dipcatcher_backup_last_success_timestamp": 0,
            "dipcatcher_canary_abort": 1,
        },
    )
    return module.read_gauges(directory / "dipcatcher.prom")


def _annotations() -> dict[str, dict[str, str]]:
    document = yaml.safe_load(_ALERTS.read_text(encoding="utf-8"))
    found: dict[str, dict[str, str]] = {}
    for group in document["groups"]:
        for rule in group["rules"]:
            name = rule.get("alert")
            if name in _NAMES:
                notes = rule.get("annotations") or {}
                found[str(name)] = {str(key): str(value) for key, value in notes.items()}
    return found


def promtool_case(gauges: dict[str, int], notes: dict[str, dict[str, str]]) -> dict[str, object]:
    """A promtool unit test. Backup age needs ``time()`` past 86400 seconds."""
    series = []
    for name in _NAMES:
        gauge = _GAUGE[name]
        value = int(gauges[gauge])
        series.append({"series": gauge, "values": f"{value}+0x1700"})
    checks = []
    for name in _NAMES:
        checks.append(
            {
                "eval_time": "26h",
                "alertname": name,
                "exp_alerts": [
                    {
                        "exp_labels": {"severity": "page"},
                        "exp_annotations": notes[name],
                    }
                ],
            }
        )
    return {
        "rule_files": ["alerts.yml"],
        "evaluation_interval": "1m",
        "tests": [{"interval": "1m", "input_series": series, "alert_rule_test": checks}],
    }


def _docker() -> list[str]:
    sudo = os.environ.get("SUDO")
    if sudo is None:
        sudo = "sudo"
    sudo = sudo.strip()
    if not sudo:
        return ["docker"]
    return [*shlex.split(sudo), "docker"]


def run_promtool(work: Path) -> tuple[int, str]:
    """Evaluate the generated test with the pinned Prometheus image."""
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
            "signal_drill.yml",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    detail = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, detail[-2000:]


def execute() -> dict[str, object]:
    work = Path(os.environ.get("DIPCATCHER_SIGNAL_DRILL_DIR", "/tmp/dipcatcher-ops14-signals"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    gauges = publish_faults(work)
    notes = _annotations()
    missing = [name for name in _NAMES if name not in notes]
    if missing or any(_GAUGE[name] not in gauges for name in _NAMES):
        return {"error": "publish", "gauges": gauges, "missing": missing, "ok": False}
    case = promtool_case(gauges, notes)
    (work / "signal_drill.yml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    shutil.copyfile(_ALERTS, work / "alerts.yml")
    code, detail = run_promtool(work)
    return {
        "alerts": list(_NAMES),
        "error": "" if code == 0 else detail,
        "gauges": gauges,
        "ok": code == 0 and not missing,
        "promtool_exit": code,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-signal-drill takes no arguments\n")
        return 1
    report = execute()
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
