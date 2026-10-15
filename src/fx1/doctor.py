"""fx-1 environment doctor — readiness signals for the model lane.

Mirrors the harness-side ``dipcatcher doctor``: presence flags only,
never secret values. Informational — nothing here hard-fails; missing
pieces are reported so an operator can fix the environment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import fx1


def _count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_receipts(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for _ in path.rglob("*.json"))


def _distribution_lineage_digest(root: Path) -> dict[str, object] | None:
    """Read distribution lineage in a harness subprocess (fx1 must not import research)."""
    script = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from quant_fund.research.crown_jewels import distribution_lineage_digest\n"
        "print(json.dumps(distribution_lineage_digest(Path(sys.argv[1]))))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def collect_status(root: Path | None = None) -> dict[str, object]:
    """Return fx-1 readiness signals rooted at *root* (default: cwd)."""
    root = root or Path.cwd()
    status: dict[str, object] = {
        "fx1_version": fx1.__version__,
        "base_model": fx1.BASE_MODEL,
        "harness_package": "ok" if importlib.util.find_spec("quant_fund") else "missing",
        # Presence flags only — values never leave the process.
        "moonshot_key": "set" if os.environ.get("MOONSHOT_API_KEY") else "unset",
        "signing_key": "set" if os.environ.get("FX1_SIGNING_KEY") else "unset",
    }
    receipts = _count_receipts(root / "receipts")
    status["receipts"] = receipts if receipts is not None else "missing"
    for name in ("corpus", "corpus_runs", "eval", "manifest"):
        path = root / "data" / "fx1" / f"{name}.{'json' if name == 'manifest' else 'jsonl'}"
        n = _count_jsonl(path)
        status[f"data_fx1_{name}"] = n if n is not None else "missing"
    ledger_path = root / "data" / "fx1" / "corpus_ledger.jsonl"
    if ledger_path.is_file():
        try:
            from fx1.data.ledger import CorpusLedger

            status["corpus_ledger_chain"] = (
                "ok" if CorpusLedger(ledger_path).verify_chain() else "BROKEN"
            )
        except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
            status["corpus_ledger_chain"] = "unverifiable"
    else:
        status["corpus_ledger_chain"] = "missing"
    try:
        from fx1.eval import DEFAULT_BANK

        status["eval_bank_tasks"] = len(DEFAULT_BANK)
    except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
        status["eval_bank_tasks"] = "unavailable"
    try:
        from fx1.harness import HARNESS_REGISTRY
        from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

        lineage = _distribution_lineage_digest(root)
        if lineage is not None:
            status["distribution_lineage"] = lineage.get("digest", "unavailable")
            status["distribution_jewels_intact"] = lineage.get("jewels_intact")
        harness_names = sorted(command.name for command in HARNESS_REGISTRY)
        status["harness_registry_digest"] = hash_bytes(canonical_json_bytes(harness_names))
    except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
        status["distribution_lineage"] = "unavailable"
    return status
