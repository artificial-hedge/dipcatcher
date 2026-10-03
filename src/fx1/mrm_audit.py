"""mrm_audit — adversarial probes on the MRM dossier compiler.

One real laundering hole fixed: a contamination report filed under a
non-``contamination_report`` activity key with an innocuous filename was
never parsed — ``overall_flagged: true`` escaped the ship gate. Every
JSON artifact is now inspected; any document declaring the flag counts,
and a *declared* report that is malformed or missing the flag counts as
flagged (fail closed both directions).

Pinned edges: missing artifact raises ``FileNotFoundError``; incomplete
five-activity evidence vetoes ship eligibility even when the model card
passed; artifact hashes are real sha256s; contamination, incomplete, and
failed-card dossiers all report ``ship_eligible=False``.
Sealed ``mrm_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["mrm_audit", "mrm_audit_bench"]


def _card(path: Path, *, ship: bool = True) -> Path:
    from fx1.modelcard import EvalDelta, ModelCard

    card = ModelCard(
        version="fx-1.v0.1",
        base_model="moonshotai/Kimi-K3",
        corpus_sha256="a" * 64,
        corpus_receipt_range="r0..r9",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.6 if ship else 0.4,
            general_pass_rate_base=0.5,
            general_pass_rate_candidate=0.5,
            honesty_gate_candidate=ship,
        ),
    )
    card.save(path)
    return path


def _artifacts(
    root: Path, *, contamination: dict[str, Any] | None = None, skip: str = ""
) -> dict[str, str | Path]:
    art: dict[str, str | Path] = {}
    for activity in ("development", "implementation", "validation", "monitoring"):
        if activity == skip:
            continue
        p = root / f"{activity}.json"
        p.write_text(json.dumps({"activity": activity}))
        art[activity] = str(p)
    if contamination is not None:
        p = root / "contamination_report.json"
        p.write_text(json.dumps(contamination))
        art["contamination_report"] = str(p)
    return art


def _compile(card: Path, art: dict[str, str | Path], root: Path) -> Any:
    from fx1.mrm import compile_dossier

    return compile_dossier(modelcard_path=card, artifacts=art, out_path=root / "dossier.json")


def mrm_audit() -> dict[str, Any]:
    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        card = _card(root / "card.json")

        full = _compile(card, _artifacts(root, contamination={"overall_flagged": False}), root)
        out["full_dossier"] = {
            "complete": full.complete,
            "ship_eligible": full.ship_eligible,
            "n_sections": len(full.sections),
        }

        flagged = _compile(card, _artifacts(root, contamination={"overall_flagged": True}), root)
        out["flagged_ship_vetoed"] = not flagged.ship_eligible
        out["flagged_reported"] = flagged.contamination_flagged

        # laundering: contamination content under "validation" key + plain name
        art = _artifacts(root)
        p = root / "val.json"
        p.write_text(json.dumps({"overall_flagged": True}))
        art["validation"] = str(p)
        laundered = _compile(card, art, root)
        out["laundered_flagged"] = laundered.contamination_flagged
        out["laundered_ship_vetoed"] = not laundered.ship_eligible

        # malformed declared contamination report → flagged
        art = _artifacts(root)
        bad = root / "contamination_report.json"
        bad.write_text("{not json")
        art["contamination_report"] = str(bad)
        malformed = _compile(card, art, root)
        out["malformed_declared_flagged"] = malformed.contamination_flagged

        # declared report missing the flag key → flagged (fail-closed default)
        art = _artifacts(root)
        empty = root / "contamination_report.json"
        empty.write_text(json.dumps({"sections": 3}))
        art["contamination_report"] = str(empty)
        missing_flag = _compile(card, art, root)
        out["missing_flag_declared_flagged"] = missing_flag.contamination_flagged

        # incomplete dossier vetoes ship even with a clean card
        incomplete = _compile(card, _artifacts(root, skip="monitoring"), root)
        out["incomplete_vetoes_ship"] = not incomplete.ship_eligible
        out["incomplete_not_complete"] = not incomplete.complete

        # missing artifact fails closed
        art = _artifacts(root)
        art["development"] = str(root / "ghost.json")
        try:
            _compile(card, art, root)
            out["missing_artifact_raises"] = "no-raise"
        except FileNotFoundError:
            out["missing_artifact_raises"] = "raise:FileNotFoundError"

        # hashes are real
        dev = root / "development.json"
        expected = hashlib.sha256(dev.read_bytes()).hexdigest()
        sec = next(s for s in full.sections if s.activity == "development")
        out["artifact_hashes_real"] = expected in sec.artifact_hashes.values()

        # failed card vetoes
        badcard = _card(root / "bad.json", ship=False)
        vetoed = _compile(badcard, _artifacts(root, contamination={"overall_flagged": False}), root)
        out["failed_card_vetoed"] = not vetoed.ship_eligible
    return out


def mrm_audit_bench() -> dict[str, Any]:
    r = mrm_audit()
    ok = (
        r["full_dossier"]["complete"] is True
        and r["full_dossier"]["ship_eligible"] is True
        and r["full_dossier"]["n_sections"] == 5
        and r["flagged_ship_vetoed"] is True
        and r["laundered_flagged"] is True
        and r["laundered_ship_vetoed"] is True
        and r["malformed_declared_flagged"] is True
        and r["missing_flag_declared_flagged"] is True
        and r["incomplete_vetoes_ship"] is True
        and r["incomplete_not_complete"] is True
        and r["missing_artifact_raises"] == "raise:FileNotFoundError"
        and r["artifact_hashes_real"] is True
        and r["failed_card_vetoed"] is True
    )
    payload: dict[str, Any] = {
        "kind": "mrm_audit",
        "schema": "mrm_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "MRM dossier contract holds: contamination (declared, malformed, "
            "missing-flag, or laundered under another activity) vetoes ship; "
            "incomplete evidence vetoes ship; missing artifacts fail closed."
            if ok
            else f"MRM DOSSIER DEFECT: {r}"
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
