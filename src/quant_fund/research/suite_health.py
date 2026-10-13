"""Suite health — one sealed summary over the whole evidence trail.

Re-verifies every committed receipt in a directory (structure + seal +
kind contracts), harvests its p-values/e-values when ``corpus_inference``
is merged, and pools the evidence with ``emerge`` mergers — valid under
the arbitrary dependence between receipts that the corpus imposes.

The receipt answers "does the inference suite have anything to say right
now, and is every artifact it cites intact?" — a single command a CI job
or a human can run to audit the entire evidence trail at once. Fails
closed: an unverifiable or corrupt receipt shows up as ``n_failed`` and
the pooled claim is withheld rather than asserted over partial evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.research.emerge import emerge_mean
from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.research.legacy_unsealed import (
    EMPTY_QUARANTINE,
    Quarantine,
    ReceiptStatus,
    classify_receipt,
    corpus_receipt_files,
    quarantine_summary,
    stale_quarantine_entries,
)
from quant_fund.research.receipt_v2 import (
    seal_receipt,
    verify_receipt_bytes,
    wrap_receipt_v2,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SUITE_HEALTH_SCHEMA = "suite_health.v1"


def _lazy_corpus() -> Any:
    try:
        import importlib

        return importlib.import_module("quant_fund.research.corpus_inference")
    except ImportError:
        return None


def _lazy_epoch_check() -> Any:
    try:
        import importlib

        return importlib.import_module("quant_fund.research.corpus_epoch").check_epoch_chain
    except ImportError:
        return None


def _encodable(value: object) -> str:
    """Attacker-controlled strings (lone surrogates such as ``\\ud800``) must
    not crash the frame or the canonical hash — escape what UTF-8 can't hold."""
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


def _epoch_state(
    root: Path,
    expected_head: Mapping[str, Any] | None,
    epoch_acks: list[dict[str, Any]] | None,
    require_stamped: bool,
) -> dict[str, Any]:
    """Epoch-chain attestation: the health receipt pins the membership-chain
    state it ran under (same pin + ledger as ``corpus-epoch --check``), so a
    stamped-then-tampered corpus is visible in the audit record itself."""
    epoch_check = _lazy_epoch_check()
    if epoch_check is None:
        return {"available": False}
    chain = epoch_check(
        root, acks=epoch_acks, expected_head=expected_head, require_stamped=require_stamped
    )
    # "no_epoch_receipts" = unstamped corpus — benign only when no head pin
    # claims a chain exists; a broken stamped chain is the violation.
    hard = [e for e in chain["errors"] if e != "no_epoch_receipts" or expected_head is not None]
    return {
        "available": True,
        "head": chain.get("head"),
        "head_epoch_root": chain.get("head_epoch_root"),
        "errors": list(chain["errors"]),
        "acknowledged": list(chain.get("acknowledged", [])),
        "pinned": expected_head is not None,
        "n_unstamped": len(chain["unstamped"]),
        "chain_ok": not hard,
    }


def suite_health(
    receipts_dir: Path | str = "receipts",
    *,
    alpha: float = 0.05,
    quarantine: Quarantine = EMPTY_QUARANTINE,
    expected_head: Mapping[str, Any] | None = None,
    epoch_acks: list[dict[str, Any]] | None = None,
    require_stamped: bool = False,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Re-verify and summarize every receipt under ``receipts_dir``.

    Returns (per-receipt frame, ``suite_health.v1`` receipt). The frame has
    one row per file: path, schema/kind, seal verdict, ``status``
    (``verified`` / ``quarantined`` / ``invalid`` per the ``quarantine``
    registry), error count, and any harvested evidence. Receipt validity and
    corpus validity are reported separately: per-file statuses, then the
    epoch-chain state (checked with the same heads pin, acknowledgment
    ledger and stamping policy as ``corpus-epoch --check``). Evidence is
    harvested from verified receipts only, and the pooled claim is withheld
    when any receipt failed verification, quarantined or not.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    root = Path(receipts_dir)
    if not root.is_dir():
        raise ValueError(f"receipts dir not found: {root}")
    # Recursive to match the epoch chain's member semantics — a receipt in a
    # subdirectory is still corpus evidence, and the quarantine subtree is
    # verified against its registry rather than skipped unseen.
    files = corpus_receipt_files(root)
    if not files:
        raise ValueError(f"no receipts under {root}")

    corpus_mod = _lazy_corpus()
    rows: list[dict[str, Any]] = []
    evalues: list[float] = []
    labels: dict[str, str] = {}
    digests: dict[str, str] = {}
    findings_total = 0
    statuses: dict[str, ReceiptStatus] = {}
    problems: dict[str, list[str]] = {}

    for path in files:
        # One read: the verdict and the digest classified against the
        # registry must describe the same bytes.
        raw = path.read_bytes()
        result = verify_receipt_bytes(raw, path)
        rel = path.relative_to(root).as_posix()
        digests[rel] = hash_bytes(raw)
        status, why = classify_receipt(
            rel,
            digests[rel],
            valid=bool(result["valid"]),
            errors=list(result["errors"]),
            quarantine=quarantine,
        )
        statuses[rel] = status
        if why:
            problems[rel] = why
        try:
            payload: Any = json.loads(raw)
        except (ValueError, RecursionError):
            # Hostile input (bad JSON/encoding, >4300-digit ints, pathological
            # nesting) degrades to "no payload"; its verdict above already
            # records why — one file must never crash the corpus sweep.
            payload = None
        kind = (payload.get("kind") or payload.get("schema")) if isinstance(payload, dict) else None
        if isinstance(payload, dict):
            body = payload.get("payload")
            inner = body if isinstance(body, dict) else payload
            labels[rel] = _encodable(inner.get("data_label") or "UNKNOWN")
        else:
            labels[rel] = "UNKNOWN"
        n_findings = 0
        # Only verified receipts contribute evidence — a quarantined or
        # invalid receipt's claims are retained record, not measurements.
        if corpus_mod is not None and isinstance(payload, dict) and status == "verified":
            for f in corpus_mod.harvest_findings(payload, rel):
                n_findings += 1
                e = f.get("evalue")
                if isinstance(e, (int, float)) and np.isfinite(e) and e > 0:
                    evalues.append(float(e))
        findings_total += n_findings
        rows.append(
            {
                "file": rel,
                "kind": _encodable(kind) if kind else None,
                "valid": result["valid"],
                "status": status,
                "n_errors": len(result["errors"]),
                "n_findings": n_findings,
            }
        )

    frame = pl.DataFrame(rows)
    distinct_labels = set(labels.values())
    if len(distinct_labels) == 1:
        data_label = distinct_labels.pop()
    elif distinct_labels:
        data_label = "MIXED"
    else:
        data_label = "UNKNOWN"
    n_ok = sum(1 for st in statuses.values() if st == "verified")
    n_failed = frame.height - n_ok
    n_quarantined = sum(1 for st in statuses.values() if st == "quarantined")
    stale_entries = stale_quarantine_entries(root, quarantine)
    # pooled evidence: arithmetic mean of harvested e-values — valid under
    # arbitrary dependence (Vovk & Wang). Withheld if any receipt fails:
    # a corrupt artifact poisons the corpus, so the suite stays silent.
    pooled = float(emerge_mean(evalues)) if evalues and n_failed == 0 else None
    pooled_alarmed = bool(pooled is not None and pooled >= 1.0 / alpha)

    # Epoch-chain attestation: the health receipt pins the membership-chain
    # state it ran under, so a stamped-then-tampered corpus is visible in the
    # audit record itself, not only via `corpus-epoch --check`.
    epoch_state = _epoch_state(root, expected_head, epoch_acks, require_stamped)

    receipt: dict[str, Any] = {
        "schema": SUITE_HEALTH_SCHEMA,
        "kind": "suite_health",
        "level": "research",
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": hash_bytes(frame.write_csv().encode("utf-8")),
        # Corpus-level fingerprint: digest over the audited receipt file
        # contents only — corpus lanes over the same directory agree on it,
        # which is what the cross-receipt lattice edges on.
        "dataset_sha256": hash_bytes(
            canonical_json_bytes(
                {"shards": {name: {"file_sha256": d} for name, d in digests.items()}}
            )
        ),
        "code_revision": git_revision(),
        "params": {
            "receipts_dir": str(root),
            "alpha": alpha,
            "n_files": len(files),
            "input_labels": labels,
        },
        "n_receipts": frame.height,
        "n_ok": n_ok,
        "n_failed": n_failed,
        # n_failed = quarantined (registered, byte-pinned, retained record)
        # + invalid (anything else — the strict gate fails on these).
        "n_quarantined": n_quarantined,
        "n_invalid": n_failed - n_quarantined,
        "invalid": {rel: why for rel, why in sorted(problems.items())},
        "quarantine": {
            **quarantine_summary(statuses, quarantine),
            "stale_entries": stale_entries,
        },
        "corpus_epoch": epoch_state,
        "corpus_lane_available": corpus_mod is not None,
        "n_findings_harvested": findings_total,
        "n_evalues_pooled": len(evalues),
        "pooled_evalue": pooled,
        "pooled_alarmed": pooled_alarmed,
        "evidence": [
            "per_receipt_reverification",
            "arbitrary_dependence_pooling",
            "fail_closed_on_corruption",
            "lazy_corpus_harvest",
        ],
        "claims": [
            {
                "text": (
                    "all committed receipts re-verify and harvested evidence "
                    "is pooled under arbitrary dependence"
                ),
                "kind": "empirical_synthetic",
            }
        ],
    }
    return frame, receipt


def write_suite_health_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal a suite_health receipt and write ``suite_health_<hash>.json``.

    Filename digest = canonical ``receipt_sha256``. Atomic, fail-closed on
    a malformed receipt. ``receipt_version=2`` wraps the same body in the
    unified ``receipt.v2`` envelope instead.
    """
    if (
        receipt.get("schema") != SUITE_HEALTH_SCHEMA
        or receipt.get("kind") != "suite_health"
        or not isinstance(receipt.get("inputs_sha256"), str)
        or not isinstance(receipt.get("params"), Mapping)
    ):
        raise ValueError("suite_health receipt violates its contract")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="fail" if receipt.get("n_failed") else "pass",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"suite_health_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["SUITE_HEALTH_SCHEMA", "suite_health", "write_suite_health_receipt"]
