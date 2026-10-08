#!/usr/bin/env python
"""Fast in-process integrity scan over committed receipts.

Why this exists
---------------
``make receipts-reverify`` shells out to ``quant_fund.cli.main
verify-receipt`` **once per receipt**. On this tree that is ~3.1 s of
interpreter startup and heavy transitive imports per file — roughly
**20 minutes for ~390 receipts, with no output until the end**. A
verification surface that cannot return a verdict in a useful time is not
a verification surface, and a slow gate buries its own findings: the
20-minute run reported 10 undifferentiated failures, which mixed one real
gate bug, three genuinely broken self-seals, and six schema-convention
mismatches that are not corruption at all.

This scanner performs the *digest* half of that verification **in-process**
and reports in seconds, split by cause so a human can triage it:

- ``ok``         — ``receipt_sha256`` recomputes correctly under the canonical
  convention (``canonical_json_bytes``, ``ensure_ascii=False``).
- ``ok_alt_convention`` — sealed under ``json.dumps``'s default
  ``ensure_ascii=True``, which escapes non-ASCII. These receipts hash correctly
  over their own content; they are **not** corrupt and **must not** be reported
  as such. Two committed receipts are in this state.
- ``mismatch``   — ``receipt_sha256`` does **not** match the content. The
  receipt's integrity claim is false and it is **not usable as evidence**
  until honestly re-sealed.
- ``no_selfseal``— the schema carries its own seal mechanism and has no
  ``receipt_sha256`` field at all. These are **not failures**; they are
  handed to their schema-specific verifier (``quant_fund.data.prereg_seal``
  for ``forward_record_preregistration.v1``, and the lane-receipt verifier
  for ``lane_receipt.v1``). This scanner reports them so they are visible
  rather than silently skipped.
- ``unparseable``— not valid JSON.

Scope and honesty
-----------------
This is a **digest scanner, not a full verifier.** It does not check
tape manifests, params, research-only flags, or live-PnL claims — that is
``verify-receipt``'s job. It exists so the *integrity* half is cheap and
fast enough to run on every change, leaving the expensive full gate for CI.

It fails closed: any ``mismatch`` exits non-zero. ``no_selfseal`` does not
fail the scan, because those receipts are governed by a different seal and
counting them as failures would be exactly the false-positive this tool
exists to eliminate.

Usage::

    uv run python scripts/receipt_integrity_scan.py receipts
    uv run python scripts/receipt_integrity_scan.py receipts --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_fund.proofcore.ci import receipt_paths  # noqa: E402
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes  # noqa: E402

_SELF_FIELD = "receipt_sha256"


def classify(path: Path) -> tuple[str, str]:
    """Return ``(status, detail)`` for one receipt file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "unparseable", f"{type(exc).__name__}"
    if not isinstance(payload, dict):
        return "unparseable", "not_a_mapping"
    schema = str(payload.get("schema", "?"))

    if _SELF_FIELD not in payload:
        return "no_selfseal", schema

    stored = payload[_SELF_FIELD]
    body = {k: v for k, v in payload.items() if k != _SELF_FIELD}
    try:
        canonical = hash_bytes(canonical_json_bytes(body))
    except Exception as exc:  # fail closed: cannot recompute == cannot vouch
        return "mismatch", f"{schema} (recompute failed: {type(exc).__name__})"

    if stored == canonical:
        return "ok", schema

    # A second serialization convention exists in the committed corpus:
    # ``json.dumps`` with the default ``ensure_ascii=True``, which escapes
    # non-ASCII (em dash, arrow) instead of emitting UTF-8 as
    # ``canonical_json_bytes`` does. Receipts written by those producers hash
    # correctly over their own content — they are NOT corrupt, and reporting
    # them as corrupt would be exactly the false positive this tool exists to
    # remove. They get their own status so the distinction stays visible.
    try:
        alt = hash_bytes(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    except Exception:
        alt = None
    if alt is not None and stored == alt:
        return "ok_alt_convention", f"{schema} (sealed with ensure_ascii=True)"

    return "mismatch", f"{schema} stored={str(stored)[:16]}… actual={canonical[:16]}…"


def scan(receipts_dir: Path) -> dict[str, Any]:
    paths = receipt_paths(receipts_dir)
    results: dict[str, list[tuple[str, str]]] = {
        "ok": [],
        "ok_alt_convention": [],
        "mismatch": [],
        "no_selfseal": [],
        "unparseable": [],
    }
    for path in paths:
        status, detail = classify(path)
        results[status].append((path.name, detail))
    return {
        "schema": "receipt_integrity_scan.v1",
        "scanned": len(paths),
        "counts": {k: len(v) for k, v in results.items()},
        "mismatched": [{"file": n, "detail": d} for n, d in results["mismatch"]],
        "no_selfseal": [{"file": n, "schema": d} for n, d in results["no_selfseal"]],
        "unparseable": [{"file": n, "detail": d} for n, d in results["unparseable"]],
        "verdict": "PASS" if not results["mismatch"] and not results["unparseable"] else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="receipt_integrity_scan.py",
        description="Fast in-process self-seal digest scan over committed receipts.",
    )
    parser.add_argument(
        "receipts_dir",
        nargs="?",
        type=Path,
        default=Path("receipts"),
        help="Directory of committed receipts (default: receipts)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    report = scan(args.receipts_dir)
    counts = report["counts"]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"receipt integrity scan — {args.receipts_dir}  ({report['verdict']})")
        print(f"  scanned        : {report['scanned']}")
        print(f"  self-seal ok   : {counts['ok']}")
        print(
            f"  ok (alt conv.) : {counts['ok_alt_convention']}  (sealed with ensure_ascii=True; intact)"
        )
        print(f"  MISMATCH       : {counts['mismatch']}")
        print(f"  no self-seal   : {counts['no_selfseal']}  (different schema seal; not a failure)")
        print(f"  unparseable    : {counts['unparseable']}")
        for entry in report["mismatched"]:
            print(f"    FAIL {entry['file']}  [{entry['detail']}]")
        for entry in report["unparseable"]:
            print(f"    FAIL {entry['file']}  [{entry['detail']}]")

    # Fail closed on a false integrity claim or an unreadable artifact.
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
