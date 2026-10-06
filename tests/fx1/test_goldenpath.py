"""Regression for scripts/fx1_goldenpath.py — the end-to-end golden-path lane.

Runs the real driver subprocess (it itself boots a real ``fx1 harness
serve``), then asserts the sealed ``goldenpath.v1`` verifies and every leg
that ran passed. Legs gated on optional-but-absent pieces (no anthropic
package, no real weight artifacts in-repo) stay honest ``ran: false`` gaps.
"""

from __future__ import annotations

import json
import os
import subprocess  # noqa: S404 — the driver is itself a subprocess harness
import sys
from pathlib import Path

from quant_fund.research.receipt_v2 import verify_receipt_file

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "fx1_goldenpath.py"


def test_goldenpath_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "goldenpath.v1.json"
    proc = subprocess.run(  # noqa: S603 — argv is a fixed local list
        [
            sys.executable,
            str(_SCRIPT),
            "--out",
            str(out),
            "--work-dir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(_ROOT / "src")),
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    verdict = verify_receipt_file(out)
    assert verdict["valid"], verdict["errors"]
    receipt = json.loads(out.read_text())
    claim = receipt["claim"]
    results = claim["results"]

    # every leg that really ran succeeded — honest gaps are excluded by design
    assert claim["runnable_ok"], claim["gaps"]
    for name, entry in results.items():
        if entry["ran"]:
            assert entry["ok"], f"{name}: {entry['detail']}"
        else:
            assert entry["detail"], f"{name}: honest gaps must carry a detail"

    # the documented gaps stay pinned honest
    assert results["weights_direct_ran"]["ran"] is False
    assert "weights_direct_ran" in claim["gaps"]

    # the non-negotiable legs all ran for real
    for must in (
        "serve_subprocess_boots",
        "harness_client_surface",
        "inprocess_sdk_parity",
        "local_fx1_link_ran",
        "eval_suite_terminal",
        "eval_diff_ran",
        "webhook_signed_delivery",
        "finetune_submitted",
        "finetune_terminal",
        "sigkill_recovery",
        "wire_receipts_verify",
        "goldenpath_receipt_verifies",
    ):
        assert results[must]["ran"], f"{must} did not run: {results[must]}"
        assert results[must]["ok"], f"{must}: {results[must]['detail']}"
