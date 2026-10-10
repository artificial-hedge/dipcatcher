"""receiptstore_audit — the content-addressed receipt index's contract.

``receipt_store.py`` backs ``GET /receipts*`` on both the HTTP and SDK
surfaces: a lazy sha256→file index over a receipts dir. This battery pins:

- *admission* — only ``*.json`` regular files whose document carries a
  full 64-hex ``receipt_sha256`` are indexed; malformed JSON, non-dict
  documents, missing/wrong-length/uppercase digests, symlinks,
  directories, and non-regular files never index (and never raise).
- *lookup* — sha→path resolution, duplicate digests resolving to the
  lexicographically first filename, item listing in name order.
- *staleness* — additions, removals, renames, and same-name content
  replacements are picked up on the next call; unchanged snapshots
  short-circuit; a missing or unreadable root clears the cache rather
  than serving stale lookups; availability returns honestly.

Probes run against real temp directories; the sealed receipt names
every defect found.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from fx1.serve.receipt_store import ReceiptIndex

__all__ = ["receiptstore_audit", "receiptstore_audit_bench"]

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_ALPHA_JSON = "alpha.json"
_TWO_JSON = "two.json"
_RENAMED_JSON = "renamed.json"


def _write(root: Path, name: str, sha: str | None = "x") -> Path:
    doc: dict[str, Any] = {"kind": "test_receipt"}
    if sha is not None:
        doc["receipt_sha256"] = sha
    path = root / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _probe_admission(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = tmp / "admit"
    root.mkdir()
    _write(root, "good.json", _SHA_A)
    _write(root, "nodigest.json", None)
    (root / "badjson.json").write_text("{broken", encoding="utf-8")
    _write(root, "short.json", "abc123")
    _write(root, "upper.json", "A" * 64)
    _write(root, "arr.json", None)
    (root / "arr.json").write_text("[1, 2]", encoding="utf-8")
    (root / "notjson.txt").write_text(json.dumps({"receipt_sha256": _SHA_B}))
    (root / "subdir.json").mkdir()
    (root / "bin.json").write_bytes(b"\x00\xff\xfe")
    outside = tmp / "outside.json"
    _write(tmp, "outside.json", _SHA_C)
    (root / "link.json").symlink_to(outside)
    idx = ReceiptIndex(root)
    items = idx.items()
    out["ad_only_valid"] = [sha for sha, _ in items] == [_SHA_A]
    out["ad_lookup_valid"] = idx.lookup(_SHA_A) == root / "good.json"
    out["ad_missing_none"] = idx.lookup(_SHA_B) is None
    out["ad_upper_refused"] = idx.lookup("A" * 64) is None
    out["ad_short_refused"] = idx.lookup("abc123") is None
    out["ad_symlink_skipped"] = idx.lookup(_SHA_C) is None
    out["ad_nonjson_ignored"] = all(p.name != "notjson.txt" for _, p in items)
    out["ad_dir_skipped"] = all(p.name != "subdir.json" for _, p in items)
    out["ad_binary_skipped"] = all(p.name != "bin.json" for _, p in items)
    # deeply nested document — RecursionError tolerated, not indexed
    deep = "[" * 2000 + "]" * 2000
    (root / "deep.json").write_text(deep, encoding="utf-8")
    out["ad_recursion_safe"] = idx.lookup(_SHA_B) is None and idx.available() is True
    return out


def _probe_duplicates(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = tmp / "dupes"
    root.mkdir()
    _write(root, "zeta.json", _SHA_A)
    _write(root, _ALPHA_JSON, _SHA_A)
    _write(root, "mid.json", _SHA_B)
    idx = ReceiptIndex(root)
    out["du_first_filename_wins"] = idx.lookup(_SHA_A) == root / _ALPHA_JSON
    out["du_items_name_order"] = [p.name for _, p in idx.items()] == [
        _ALPHA_JSON,
        "mid.json",
    ]
    out["du_count_distinct"] = len(idx.items()) == 2
    return out


def _probe_staleness(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    root = tmp / "stale"
    root.mkdir()
    first = _write(root, "one.json", _SHA_A)
    idx = ReceiptIndex(root)
    out["st_initial"] = idx.lookup(_SHA_A) == first
    # addition picked up
    _write(root, _TWO_JSON, _SHA_B)
    out["st_addition"] = idx.lookup(_SHA_B) == root / _TWO_JSON
    # removal clears the entry
    first.unlink()
    out["st_removal"] = idx.lookup(_SHA_A) is None
    # rename moves the mapping
    (root / _TWO_JSON).rename(root / _RENAMED_JSON)
    out["st_rename"] = idx.lookup(_SHA_B) == root / _RENAMED_JSON
    # same-name replacement with new digest
    _write(root, _RENAMED_JSON, _SHA_C)
    out["st_replace"] = idx.lookup(_SHA_B) is None and idx.lookup(_SHA_C) == root / _RENAMED_JSON
    # unchanged snapshot short-circuits without losing state
    out["st_stable"] = idx.lookup(_SHA_C) == root / _RENAMED_JSON
    # a corrupt file appearing then being fixed updates
    (root / _RENAMED_JSON).write_text("{garbage", encoding="utf-8")
    out["st_corrupt_drops"] = idx.lookup(_SHA_C) is None
    _write(root, _RENAMED_JSON, _SHA_C)
    out["st_repaired_returns"] = idx.lookup(_SHA_C) is not None
    return out


def _probe_availability(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    missing = tmp / "no-such-dir"
    idx = ReceiptIndex(missing)
    out["av_missing_unavailable"] = idx.available() is False
    out["av_missing_lookup_none"] = idx.lookup(_SHA_A) is None
    # root appears later → availability flips honestly
    missing.mkdir()
    _write(missing, "found.json", _SHA_A)
    out["av_appears"] = idx.available() is True and idx.lookup(_SHA_A) is not None
    # unreadable root clears cache rather than serving stale
    root2 = tmp / "goesaway"
    root2.mkdir()
    _write(root2, "r.json", _SHA_B)
    idx2 = ReceiptIndex(root2)
    out["av_was_available"] = idx2.available() is True
    os.chmod(root2, 0o000)
    try:
        blocked = idx2.available() is False and idx2.lookup(_SHA_B) is None
    finally:
        os.chmod(root2, stat.S_IRWXU)
    out["av_unreadable_clears"] = blocked
    # a file-not-dir root fails the same way
    froot = tmp / "file.json"
    _write(tmp, "file.json", _SHA_A)
    idx3 = ReceiptIndex(froot)
    out["av_file_not_dir"] = idx3.available() is False
    return out


def receiptstore_audit() -> dict[str, bool]:
    """Every receipt-index contract as booleans."""
    out: dict[str, bool] = {}
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        out.update(_probe_admission(base))
        out.update(_probe_duplicates(base))
        out.update(_probe_staleness(base))
        out.update(_probe_availability(base))
    return out


def receiptstore_audit_bench() -> dict[str, Any]:
    """Sealed receipt for the receipt-index battery."""
    r = receiptstore_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "receiptstore_audit",
        "schema": "receiptstore_audit.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "real temp receipt dirs (metadata index only)",
            "not_verified": [
                "cryptographic receipt verification (see verify-research lanes)",
                "concurrent-writer integrity (documented non-goal)",
            ],
        },
        "interpretation": (
            "Receipt index contract holds: admission is strict 64-hex "
            "documents on regular .json files, duplicates resolve "
            "lexicographically, staleness tracks every filesystem "
            "mutation class, and unavailable roots clear rather than "
            "serve stale lookups."
            if ok
            else f"RECEIPT-STORE AUDIT DEFECTS: {defects}"
        ),
    }
    from quant_fund.utils.reproducibility import git_revision

    out["git_revision"] = git_revision()
    from quant_fund.research.receipt_v2 import canonical_json_bytes
    from quant_fund.utils.hashing import hash_bytes

    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(receiptstore_audit_bench(), indent=2, sort_keys=True))
