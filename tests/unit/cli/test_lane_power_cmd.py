"""CLI contract for ``lane-power`` at the smallest legal overlay bounds."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(_ROOT / "src"),
        "NO_COLOR": "1",
        "TERM": "dumb",
        "FORCE_COLOR": "0",
        "FX1_CONFIG_DIR": str(tmp_path / "fx1"),
        "FX1_SESSION_DIR": str(tmp_path / "sessions"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_lane_power_smallest_run_seals_a_synthetic_receipt(tmp_path: Path) -> None:
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "lane-power",
            "--defects",
            "0.0",
            "--n-steps",
            "16",
            "--n-seeds",
            "1",
            "--lanes",
            "serial_watch",
            "--out-dir",
            str(out),
            "--receipt-version",
            "2",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DATA_LABEL=SYNTHETIC" in proc.stdout
    assert "SYNTHETIC" in proc.stdout
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("receipt="))
    receipt = Path(receipt_line.split("=", 1)[1])
    assert receipt.is_file()
    checked = subprocess.run(
        [sys.executable, "-m", "dipcatcher_cli", "verify-receipt", str(receipt)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=60,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert json.loads(checked.stdout)["valid"] is True
