"""receipts_audit — adversarial probes on receipt eligibility loading.

Pinned contract for ``load_receipts``/``_eligibility``:

- Fail-closed default: a payload with no ``research_only`` and no
  ``claim`` is treated as not-research + live-claim → ineligible.
- The explicit ``live_pnl_claim`` key always wins over the
  ``claim: "research_only"`` implication.
- ``claim == "research_only"`` with no live key → eligible.
- Every loaded record is content-bound (``sha256`` of raw bytes).
- Unparseable files and non-dict payloads are skipped silently.
- ``rglob("*.json")`` — nested dirs are scanned.

**Type contract**: ``research_only`` must be the literal ``true`` — a
truthy non-boolean (``"yes"``, ``1``) is not a research-scope
declaration; ``live_pnl_claim`` present is read with ``bool(...)``.
Together ``{"research_only": "yes", "live_pnl_claim": null}`` is
**ineligible** — the coercion escape is closed.

Sealed ``receipts_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["receipts_audit", "receipts_audit_bench"]


def _w(root: Path, name: str, payload: Any) -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        p.write_text(payload)
    else:
        p.write_text(json.dumps(payload))
    return p


def receipts_audit() -> dict[str, Any]:
    from fx1.data.receipts import load_receipts

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _w(
            root,
            "good.json",
            {
                "schema": "x.v1",
                "research_only": True,
                "live_pnl_claim": False,
                "synthetic": True,
            },
        )
        _w(root, "implicit.json", {"schema": "y.v2"})
        _w(root, "claim_contract.json", {"claim": "research_only"})
        _w(
            root,
            "explicit_wins.json",
            {
                "claim": "research_only",
                "live_pnl_claim": True,
            },
        )
        _w(
            root,
            "weird_coerce.json",
            {
                "research_only": "yes",
                "live_pnl_claim": None,
            },
        )
        _w(
            root,
            "nested/deep.json",
            {
                "research_only": True,
                "live_pnl_claim": False,
            },
        )
        _w(root, "broken.json", "{not json")
        _w(root, "listy.json", ["not", "a", "dict"])

        recs = {Path(r.path).name: r for r in load_receipts(root)}
        out["all_parseable_loaded"] = {
            "good.json",
            "implicit.json",
            "claim_contract.json",
            "explicit_wins.json",
            "weird_coerce.json",
            "deep.json",
        } == set(recs)
        out["fails_closed_default"] = (
            recs["implicit.json"].eligible is False and recs["implicit.json"].live_pnl_claim is True
        )
        out["claim_contract_eligible"] = recs["claim_contract.json"].eligible is True
        out["explicit_live_wins"] = (
            recs["explicit_wins.json"].eligible is False
            and recs["explicit_wins.json"].live_pnl_claim is True
        )
        out["coercion_closed"] = recs["weird_coerce.json"].eligible is False
        out["synthetic_class"] = (
            recs["good.json"].evidence_class == "synthetic"
            and recs["implicit.json"].evidence_class == "research"
        )
        out["nested_scanned"] = "deep.json" in recs
        out["digest_binds_bytes"] = all(len(r.sha256) == 64 for r in recs.values())

        # multi-dir iterable
        root2 = Path(tmp) / "second"
        _w(root2, "other.json", {"research_only": True, "live_pnl_claim": False})
        multi = load_receipts([root, root2])
        out["multi_dir"] = any(r.path.endswith("other.json") for r in multi)
    return out


def receipts_audit_bench() -> dict[str, Any]:
    r = receipts_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "receipts_audit",
        "schema": "receipts_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {
                "truthiness_coercion_eligible": not r["coercion_closed"],
            },
            "ok": ok,
        },
        "interpretation": (
            "Receipt eligibility contract holds: absent markers fail "
            "closed, explicit live claim always wins, claim contract "
            "resolves, digests bind bytes, unparseable skipped. Flag "
            "closed: {'research_only': 'yes', 'live_pnl_claim': "
            "null} is ineligible — research_only requires the literal "
            "true, and truthy non-booleans no longer qualify."
            if ok
            else f"RECEIPTS AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
