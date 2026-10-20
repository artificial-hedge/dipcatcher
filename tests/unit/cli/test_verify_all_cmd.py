"""CLI contract for ``verify-all``: a temp receipts dir, not the library helper."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_RECEIPTS = (
    _ROOT / "receipts" / "alert_budget.json",
    _ROOT / "receipts" / "grammar_map.json",
)


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
    }


def _verify_all(tmp_path: Path, receipts: Path, out: Path) -> subprocess.CompletedProcess[str]:
    # ``--no-check-index`` keeps the audit on this directory. The committed
    # evidence index describes ``receipts/``, not a two-file temp corpus.
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "dipcatcher_cli",
            "verify-all",
            "--receipts-dir",
            str(receipts),
            "--out-dir",
            str(out),
            "--no-check-index",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        env=_env(tmp_path),
        timeout=60,
        check=False,
    )


def _copy_pair(directory: Path) -> None:
    directory.mkdir()
    for source in _RECEIPTS:
        shutil.copy(source, directory / source.name)


def test_verify_all_seals_an_audit_of_two_committed_receipts(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    out = tmp_path / "out"
    _copy_pair(receipts)
    proc = _verify_all(tmp_path, receipts, out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("receipt="))
    receipt = Path(receipt_line.split("=", 1)[1])
    assert receipt.is_file()
    assert receipt.parent == out
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


def test_verify_all_rejects_a_tampered_receipt(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    _copy_pair(receipts)
    target = receipts / "alert_budget.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["claim"]["verdict"] = "tampered"
    target.write_text(json.dumps(payload), encoding="utf-8")
    proc = _verify_all(tmp_path, receipts, tmp_path / "out")
    assert proc.returncode != 0
    assert "sealed_receipt_invalid" in proc.stdout or "sealed_receipt_invalid" in proc.stderr
