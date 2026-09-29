"""Repo integrity capstone: one sealed attestation over the whole evidence
integrity substrate.

The pieces each gate one surface — ``crown_jewels`` byte-pins the gate-defining
files, ``corpus_epoch`` chains the evidence corpora, ``check_epoch_chain``
verifies them. ``verify_repo`` composes them into a single verdict and seals it
as a ``repo_integrity.v1`` receipt: a point-in-time proof that the repo's
evidence corpus and the config that gates it were all intact *at the same
instant*.

The committed attestation is itself a ``quality/`` chain member, so a stale
attestation can't be silently edited — the next epoch stamp records the drift.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import (
    check_epoch_chain,
    epoch_heads_key,
    load_heads_pin,
)
from quant_fund.research.crown_jewels import (
    DEFAULT_PIN_PATH as JEWELS_PIN,
)
from quant_fund.research.crown_jewels import (
    crown_jewels_errors,
)
from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import hash_bytes

REPO_INTEGRITY_SCHEMA = "repo_integrity.v1"

# (corpus_dir, member glob, require_stamped, allow_member_updates) — the same
# policy the evidence-audit Makefile target enforces; keep in sync.
CORPORA: tuple[tuple[str, str, bool, bool], ...] = (
    ("receipts", "*.json", False, False),
    ("verifier", "*.md", False, False),
    ("quality", "*.json", False, True),
    (".github/workflows", "*.yml", True, True),
)


def verify_repo(
    root: Path | str = ".", *, heads_pin: Path | str = "quality/epoch_heads.json"
) -> dict[str, Any]:
    """Run every integrity gate against the live tree.

    Returns ``{"gates": {name: {"ok": bool, "errors": [...]}}, "ok": bool}``.
    ``ok`` is true iff every gate reports zero errors. ``heads_pin`` is the
    committed epoch-heads pin required by every corpus chain.
    """
    root = Path(root)
    pin_path = root / heads_pin if not Path(heads_pin).is_absolute() else Path(heads_pin)
    gates: dict[str, dict[str, Any]] = {}

    jewel_errs = crown_jewels_errors(root, pin_path=root / JEWELS_PIN)
    gates["crown_jewels"] = {"ok": not jewel_errs, "errors": jewel_errs}

    pin_present = pin_path.is_file()
    heads = load_heads_pin(pin_path) if pin_present else {}
    for corpus_dir, pattern, require_stamped, allow_updates in CORPORA:
        cdir = root / corpus_dir
        if not cdir.is_dir():
            gates[f"epoch:{corpus_dir}"] = {
                "ok": False,
                "errors": [f"corpus_missing:{corpus_dir}"],
            }
            continue
        if not pin_present:
            expected = None
            pin_errors = [f"heads_pin_missing:{heads_pin}"]
        else:
            expected = heads.get(epoch_heads_key(corpus_dir, pattern))
            pin_errors = (
                [] if expected is not None else [f"heads_pin_entry_missing:{corpus_dir}/{pattern}"]
            )
        res = check_epoch_chain(
            cdir,
            pattern=pattern,
            expected_head=expected,
            require_stamped=require_stamped,
            allow_member_updates=allow_updates,
        )
        errs = pin_errors + list(res["errors"])
        gates[f"epoch:{corpus_dir}"] = {"ok": not errs, "errors": errs}

    return {"gates": gates, "ok": all(g["ok"] for g in gates.values())}


def repo_integrity_receipt(
    root: Path | str = ".", *, heads_pin: Path | str = "quality/epoch_heads.json"
) -> dict[str, Any]:
    """Sealed ``repo_integrity.v1`` attestation over the live gate verdicts."""
    from quant_fund.research.receipt_v2 import seal_receipt

    root = Path(root)
    verdict = verify_repo(root, heads_pin=heads_pin)
    pin_path = root / heads_pin
    body: dict[str, Any] = {
        "kind": "repo_integrity",
        "schema": REPO_INTEGRITY_SCHEMA,
        "data_label": "SYNTHETIC",
        "ok": verdict["ok"],
        "gates": {
            name: {"ok": g["ok"], "errors": sorted(g["errors"])}
            for name, g in verdict["gates"].items()
        },
        "pins": {
            "epoch_heads_sha256": hash_bytes(pin_path.read_bytes()) if pin_path.is_file() else None,
            "crown_jewels_sha256": (
                hash_bytes((root / JEWELS_PIN).read_bytes())
                if (root / JEWELS_PIN).is_file()
                else None
            ),
        },
        "params": {"corpora": [c[0] for c in CORPORA]},
    }
    return seal_receipt(body)


def write_repo_integrity_receipt(
    out_path: Path | str,
    root: Path | str = ".",
    *,
    heads_pin: Path | str = "quality/epoch_heads.json",
) -> Path:
    """Write the sealed attestation to ``out_path`` (atomic)."""
    receipt = repo_integrity_receipt(root, heads_pin=heads_pin)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return out
