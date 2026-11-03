"""fx-1 environment doctor — readiness signals for the model lane.

Mirrors the harness-side ``dipcatcher doctor``: presence flags only,
never secret values. Informational — nothing here hard-fails; missing
pieces are reported so an operator can fix the environment.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import fx1


def _count_jsonl(path: Path) -> int | str | None:
    """Count non-blank lines.

    ``None`` means the path is absent. ``"unreadable"`` means it exists but
    cannot be decoded. The doctor reports that instead of crashing.
    """
    try:
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    except (OSError, UnicodeError):
        return "unreadable"


def _count_receipts(path: Path) -> int | str | None:
    """Count receipt JSON files.

    ``None`` means the directory is absent. ``"unreadable"`` means listing it
    failed.
    """
    try:
        if not path.is_dir():
            return None
        return sum(1 for _ in path.rglob("*.json"))
    except OSError:
        return "unreadable"


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
    receipts_path = root / "receipts"
    receipts = _count_receipts(receipts_path)
    if receipts is not None:
        status["receipts"] = receipts
    else:
        # Present-but-unreadable is a different defect from absent.
        status["receipts"] = "unreadable" if receipts_path.is_dir() else "missing"
    for name in ("corpus", "corpus_runs", "eval", "manifest"):
        path = root / "data" / "fx1" / f"{name}.{'json' if name == 'manifest' else 'jsonl'}"
        n = _count_jsonl(path)
        if n is not None:
            status[f"data_fx1_{name}"] = n
        else:
            # Present-but-unreadable is a different defect from absent.
            status[f"data_fx1_{name}"] = "unreadable" if path.is_file() else "missing"
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
    return status
