"""fx-1 environment doctor — readiness signals for the model lane.

Mirrors the harness-side ``dipcatcher doctor``: presence flags only,
never secret values. Informational — nothing here hard-fails; missing
pieces are reported so an operator can fix the environment.

Distribution lineage is computed here with ``quant_fund.utils.hashing``
only — ``quant_fund.research.crown_jewels`` is outside the fx1 harness
surface allowlist (``configs/arch_boundaries.toml``), so doctor must not
import it. The local check still fingerprints the pin bytes and verifies
every pinned jewel file against its digest.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import fx1

_CROWN_JEWELS_PIN = Path("quality/crown_jewels.json")
_EPOCH_HEADS_PIN = Path("quality/epoch_heads.json")
_REPO_INTEGRITY_PIN = Path("quality/repo_integrity.json")
_DISTRIBUTION_LINEAGE_SCHEMA = "distribution_lineage.v1"


def _count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_receipts(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for _ in path.rglob("*.json"))


def _extra_presence() -> dict[str, str]:
    """Report optional runtime extras without importing heavy stacks."""
    return {
        "torch": "present" if importlib.util.find_spec("torch") else "missing",
        "safetensors": "present" if importlib.util.find_spec("safetensors") else "missing",
    }


def _jewels_intact(root: Path, pin: Path) -> bool:
    """True when every file listed in the crown-jewels pin matches its digest."""
    if not pin.is_file():
        return False
    try:
        raw: Any = json.loads(pin.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict) or raw.get("schema") != "crown_jewels.v1":
        return False
    files = raw.get("files")
    if not isinstance(files, dict) or not files:
        return False
    from quant_fund.utils.hashing import hash_bytes

    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            return False
        path = root / name
        if path.is_symlink() or not path.is_file():
            return False
        try:
            if hash_bytes(path.read_bytes()) != expected:
                return False
        except OSError:
            return False
    return True


def _distribution_lineage(root: Path) -> dict[str, object]:
    """Fingerprint the integrity pins without importing research.crown_jewels."""
    from quant_fund.utils.hashing import hash_bytes

    pin = root / _CROWN_JEWELS_PIN
    payload: dict[str, object] = {
        "schema": _DISTRIBUTION_LINEAGE_SCHEMA,
        "crown_jewels_pin_present": pin.is_file(),
        "jewels_intact": _jewels_intact(root, pin),
    }
    for label, rel in (
        ("crown_jewels_pin_sha256", _CROWN_JEWELS_PIN),
        ("epoch_heads_sha256", _EPOCH_HEADS_PIN),
        ("repo_integrity_sha256", _REPO_INTEGRITY_PIN),
    ):
        path = root / rel
        payload[label] = hash_bytes(path.read_bytes()) if path.is_file() else None
    bind_parts = [
        f"{key}={payload[key]}"
        for key in sorted(payload)
        if str(key).endswith("_sha256") and payload[key] is not None
    ]
    bind_parts.append(f"jewels_intact={payload['jewels_intact']}")
    payload["digest"] = hash_bytes("\n".join(bind_parts).encode("utf-8"))
    return payload


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
        "optional_extras": _extra_presence(),
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

        lineage = _distribution_lineage(root)
        status["distribution_lineage"] = lineage["digest"]
        status["distribution_jewels_intact"] = lineage["jewels_intact"]
        harness_names = sorted(command.name for command in HARNESS_REGISTRY)
        status["harness_registry_digest"] = hash_bytes(canonical_json_bytes(harness_names))
    except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
        status["distribution_lineage"] = "unavailable"
    try:
        from fx1.interactive.superpower import interactive_surface_digest

        status["interactive_surface_seal"] = interactive_surface_digest()
    except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
        status["interactive_surface_seal"] = "unavailable"
    return status
