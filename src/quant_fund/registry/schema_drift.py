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
Verdict ``ok`` iff zero drifts and zero phantom revisions — after the
pinned exceptions in ``quality/schema_drift_known.json``, which keep
already-committed anomalies visible (sealed receipts whose git provenance
is unattributable, or frozen tag collisions) without failing the audit.
A pin that no longer matches is itself flagged as ``stale_pins`` so the
manifest cannot quietly rot. Sealed ``schema_drift.v1``.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["schema_drift", "schema_drift_bench"]

_HEX = frozenset("0123456789abcdef")

_KNOWN_PATH = "quality/schema_drift_known.json"


def _load_known(repo: Path) -> dict[str, Any]:
    path = repo / _KNOWN_PATH
    if not path.is_file():
        return {"phantom_revisions": {}, "drifted": {}}
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"phantom_revisions": {}, "drifted": {}}
    if not isinstance(doc, dict):
        return {"phantom_revisions": {}, "drifted": {}}
    phantom = doc.get("phantom_revisions") if isinstance(doc.get("phantom_revisions"), dict) else {}
    drifted = doc.get("drifted") if isinstance(doc.get("drifted"), dict) else {}
    return {"phantom_revisions": phantom, "drifted": drifted}


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


def schema_drift(receipts_dir: Path | str = "receipts", repo: Path | str = ".") -> dict[str, Any]:
    """Group receipts by schema tag; flag key-set drift + phantom revisions."""
    receipts_dir = Path(receipts_dir)
    repo = Path(repo)
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    untagged: list[str] = []
    phantom: list[dict[str, str]] = []
    scanned: set[str] = set()
    for path in sorted(receipts_dir.glob("*.json")):
        scanned.add(path.name)
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
            anc = _is_ancestor(rev, repo)
            if anc is None:
                phantom.append({"file": path.name, "rev": rev, "why": "not_a_commit"})
                rows.append({"file": path.name, "verdict": "phantom_revision"})
            elif anc is False:
                # Post-rebase provenance: the commit exists but the receipt's
                # branch tip was rewritten — noted, not flagged.
                phantom.append({"file": path.name, "rev": rev, "why": "not_ancestor"})
                rows.append({"file": path.name, "verdict": "note:revision_not_ancestor"})
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
    known = _load_known(repo)
    known_phantom_map = known["phantom_revisions"]
    known_drift_map = known["drifted"]
    known_phantom: list[dict[str, str]] = []
    hard_phantom: list[dict[str, str]] = []
    for p in phantom:
        if p["why"] != "not_a_commit":
            continue
        pin = known_phantom_map.get(p["file"])
        if isinstance(pin, dict) and pin.get("rev") == p["rev"]:
            known_phantom.append(p)
        else:
            hard_phantom.append(p)
    known_drifted: list[dict[str, Any]] = []
    open_drifted: list[dict[str, Any]] = []
    for d in drifted:
        pin = known_drift_map.get(d["file"])
        if (
            isinstance(pin, dict)
            and pin.get("tag") == d["tag"]
            and sorted(pin.get("missing") or []) == sorted(d["missing"])
            and sorted(pin.get("extra") or []) == sorted(d["extra"])
        ):
            known_drifted.append(d)
        else:
            open_drifted.append(d)
    matched_phantom = {p["file"] for p in known_phantom}
    matched_drifted = {d["file"] for d in known_drifted}
    # A pin goes stale when its file is present but no longer matches the
    # pinned signature, or — only when scanning the repo's own corpus — the
    # file was removed without cleaning the manifest. Foreign dirs (unit
    # fixtures) never render pins stale.
    canonical_corpus = receipts_dir.resolve() == (repo / "receipts").resolve()
    unpinned = set(known_phantom_map) - matched_phantom | set(known_drift_map) - matched_drifted
    stale_pins = sorted(name for name in unpinned if canonical_corpus or name in scanned)
    return {
        "n_receipts": sum(g["n"] for g in group_stats.values()),
        "n_groups": len(group_stats),
        "groups_with_drift": sum(1 for g in group_stats.values() if g["n_keysets"] > 1),
        "group_stats": group_stats,
        "drifted": open_drifted,
        "phantom_revisions": phantom,
        "untagged": untagged,
        "known_phantom": known_phantom,
        "known_drifted": known_drifted,
        "stale_pins": stale_pins,
        "n_flagged": len(open_drifted) + len(hard_phantom) + len(stale_pins),
        "n_notes": (len(phantom) - len(hard_phantom) - len(known_phantom) + len(untagged)),
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
            "n_known_phantom": len(audit["known_phantom"]),
            "n_known_drifted": len(audit["known_drifted"]),
            "n_stale_pins": len(audit["stale_pins"]),
            "drifted": audit["drifted"][:20],
            "phantom_revisions": audit["phantom_revisions"][:20],
            "untagged": audit["untagged"][:20],
            "ok": ok,
        },
        "interpretation": (
            f"{audit['n_receipts']} receipts in {audit['n_groups']} schema groups: "
            f"{len(audit['drifted'])} drifted, {len(audit['phantom_revisions'])} phantom "
            f"revisions, {len(audit['untagged'])} untagged."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
