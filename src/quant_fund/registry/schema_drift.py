"""schema_drift — same-tag receipts must carry the same claim shape + real revisions.

Two silent-erosion classes on the evidence corpus:

1. **Claim-key drift** — receipts sharing a ``schema``/``kind`` tag are
   supposed to be one contract. If a later emission adds or drops claim keys
   while keeping the tag, the contract weakened (or drifted) without a schema
   bump. Drift = a receipt whose claim-key set differs from the group's modal
   key set.
2. **Phantom revisions** — ``git_revision`` is self-attested; a receipt
   naming a commit that is not an ancestor of HEAD (or not a commit at all)
   claims provenance that never existed. Ancestry is checked with
   ``git merge-base --is-ancestor``.

Receipts missing a schema/kind tag are flagged ``missing_schema_tag``.
Verdict ``ok`` iff zero drifts and zero unacknowledged phantom revisions.
A ``not_a_commit`` revision is *acknowledged* — reported, not flagged — only
when ``quality/legacy_quarantine.json`` ``provenance_gaps`` pins that exact
file (bytes) and revision with a reason: code provenance the repository
cannot resolve is recorded as unresolvable, never rewritten to a different
commit. A gap entry whose file or bytes or revision no longer match is
flagged ``provenance_gap_stale``. Sealed ``schema_drift.v1``.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["schema_drift", "schema_drift_bench"]

_HEX = frozenset("0123456789abcdef")


def _claim_keys(payload: dict[str, Any]) -> tuple[str, ...]:
    claim = payload.get("claim")
    if isinstance(claim, dict):
        return tuple(sorted(claim))
    return ()


def _is_ancestor(rev: str, repo: Path) -> bool | None:
    """True/False via merge-base; None when the object does not exist."""
    if not rev or not set(rev) <= _HEX or not (7 <= len(rev) <= 64):
        return None
    kind = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-t", rev],
        capture_output=True,
        text=True,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        return None
    probe = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", rev, "HEAD"],
        capture_output=True,
    )
    return probe.returncode == 0


def _registry_gaps(receipts_dir: Path, repo: Path) -> Mapping[str, Any]:
    """Provenance gaps from ``quality/legacy_quarantine.json``. The registry
    describes the repository's own ``receipts/`` corpus — it must never
    excuse (or flag) files in some other directory being scanned."""
    from quant_fund.research.legacy_unsealed import (
        DEFAULT_QUARANTINE_PATH,
        load_quarantine_or_empty,
    )

    if receipts_dir.resolve() != (repo / "receipts").resolve():
        return {}
    return load_quarantine_or_empty(repo / DEFAULT_QUARANTINE_PATH).provenance_gaps


def _stale_gaps(gaps: Mapping[str, Any], hits: set[str]) -> list[dict[str, str]]:
    """Gap entries that matched nothing: file gone, bytes rewritten, revision
    edited, or the commit now resolves — flagged so the registry cannot keep
    excusing a receipt that changed under it."""
    return [
        {"file": rel, "rev": gap.git_revision, "why": "provenance_gap_stale"}
        for rel, gap in sorted(gaps.items())
        if rel not in hits
    ]


def _gap_pins(gap: Any, rev: str, path: Path) -> bool:
    """True iff a registry provenance gap pins exactly this file and revision."""
    return (
        gap is not None and gap.git_revision == rev and gap.sha256 == hash_bytes(path.read_bytes())
    )


def _revision_verdict(
    path: Path, rev: str, repo: Path, gaps: Mapping[str, Any]
) -> tuple[str, dict[str, str] | None]:
    """Row verdict + record for one receipt's self-attested ``git_revision``."""
    anc = _is_ancestor(rev, repo)
    if anc is None and _gap_pins(gaps.get(path.name), rev, path):
        record = {"file": path.name, "rev": rev, "why": "provenance_gap"}
        return "note:provenance_gap_acknowledged", record
    if anc is None:
        return "phantom_revision", {"file": path.name, "rev": rev, "why": "not_a_commit"}
    if anc is False:
        # Post-rebase provenance: the commit exists but the receipt's
        # branch tip was rewritten — noted, not flagged.
        return "note:revision_not_ancestor", {"file": path.name, "rev": rev, "why": "not_ancestor"}
    return "", None


def schema_drift(receipts_dir: Path | str = "receipts", repo: Path | str = ".") -> dict[str, Any]:
    """Group receipts by schema tag; flag key-set drift + phantom revisions."""
    receipts_dir = Path(receipts_dir)
    repo = Path(repo)
    gaps = _registry_gaps(receipts_dir, repo)
    acknowledged: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    untagged: list[str] = []
    phantom: list[dict[str, str]] = []
    for path in sorted(receipts_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            rows.append({"file": path.name, "verdict": "unparseable"})
            continue
        if not isinstance(payload, dict):
            rows.append({"file": path.name, "verdict": "not_an_object"})
            continue
        tag = payload.get("schema") or payload.get("kind")
        if not isinstance(tag, str) or not tag:
            untagged.append(path.name)
            rows.append({"file": path.name, "verdict": "missing_schema_tag"})
            continue
        groups.setdefault(tag, []).append((path.name, _claim_keys(payload)))
        rev = payload.get("git_revision")
        if isinstance(rev, str) and rev:
            verdict, record = _revision_verdict(path, rev, repo, gaps)
            if record is not None:
                acked = record["why"] == "provenance_gap"
                (acknowledged if acked else phantom).append(record)
                rows.append({"file": path.name, "verdict": verdict})
    drifted: list[dict[str, Any]] = []
    group_stats: dict[str, Any] = {}
    for tag, members in sorted(groups.items()):
        keysets = Counter(keys for _, keys in members)
        modal, modal_n = keysets.most_common(1)[0]
        group_stats[tag] = {
            "n": len(members),
            "n_keysets": len(keysets),
            "modal_keys": len(modal),
            "modal_share": round(modal_n / len(members), 4),
        }
        if len(keysets) > 1:
            for fname, keys in members:
                if keys != modal:
                    missing = sorted(set(modal) - set(keys))
                    extra = sorted(set(keys) - set(modal))
                    drifted.append({"file": fname, "tag": tag, "missing": missing, "extra": extra})
    hard_phantom = [p for p in phantom if p["why"] == "not_a_commit"]
    # A gap entry that matched nothing is stale (file gone, bytes rewritten,
    # revision edited, or the commit now resolves) — fail closed so the
    # registry cannot keep excusing a receipt that changed under it.
    stale_gaps = _stale_gaps(gaps, {a["file"] for a in acknowledged})
    return {
        "n_receipts": sum(g["n"] for g in group_stats.values()),
        "n_groups": len(group_stats),
        "groups_with_drift": sum(1 for g in group_stats.values() if g["n_keysets"] > 1),
        "group_stats": group_stats,
        "drifted": drifted,
        "phantom_revisions": phantom,
        "untagged": untagged,
        "acknowledged_provenance_gaps": acknowledged,
        "stale_provenance_gaps": stale_gaps,
        "n_flagged": len(drifted) + len(hard_phantom) + len(stale_gaps),
        "n_notes": len(phantom) - len(hard_phantom) + len(untagged),
    }


def schema_drift_bench(
    receipts_dir: Path | str = "receipts", repo: Path | str = "."
) -> dict[str, Any]:
    audit = schema_drift(receipts_dir, repo)
    ok = audit["n_flagged"] == 0
    payload: dict[str, Any] = {
        "kind": "schema_drift",
        "schema": "schema_drift.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "invariant": "same-tag receipts share one claim schema; git_revision is a real ancestor",
            "n_receipts": audit["n_receipts"],
            "n_groups": audit["n_groups"],
            "groups_with_drift": audit["groups_with_drift"],
            "n_drifted": len(audit["drifted"]),
            "n_phantom_not_commit": sum(
                1 for p in audit["phantom_revisions"] if p["why"] == "not_a_commit"
            ),
            "n_not_ancestor": sum(
                1 for p in audit["phantom_revisions"] if p["why"] == "not_ancestor"
            ),
            "n_untagged": len(audit["untagged"]),
            "n_provenance_gaps_acknowledged": len(audit["acknowledged_provenance_gaps"]),
            "n_provenance_gaps_stale": len(audit["stale_provenance_gaps"]),
            "acknowledged_provenance_gaps": audit["acknowledged_provenance_gaps"][:20],
            "drifted": audit["drifted"][:20],
            "phantom_revisions": audit["phantom_revisions"][:20],
            "untagged": audit["untagged"][:20],
            "ok": ok,
        },
        "interpretation": (
            f"{audit['n_receipts']} receipts in {audit['n_groups']} schema groups: "
            f"{len(audit['drifted'])} drifted, {len(audit['phantom_revisions'])} phantom "
            f"revisions, {len(audit['acknowledged_provenance_gaps'])} acknowledged "
            f"provenance gaps, {len(audit['untagged'])} untagged."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
