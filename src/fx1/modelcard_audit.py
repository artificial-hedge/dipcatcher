"""modelcard_audit — adversarial probes on the fx-1 model card contract.

The card is the versioned, evidence-bound checkpoint metadata — the ship
gate is a property of it, so the boundaries matter:

- ``version`` pins ``^fx-1\\.v\\d+\\.\\d+$``: ``fx-1.v1``, ``fx-2.v1.0``,
  ``fx-1.v1.0.0``, ``FX-1.v1.0`` all reject.
- ``_never_live`` fails closed: ``live_pnl_claim=True`` OR
  ``research_only=False`` → ``ValidationError`` (research scope is not
  advisory).
- ``EvalDelta.ship_eligible`` = honesty gate AND domain *strictly*
  better AND general *not regressed* — equality on domain is not a ship.
- ``corpus_sha256``/``training_manifest_sha256`` enforce 64-hex.
- ``save``/``load`` round-trip preserves the card byte-for-byte
  semantically.

Sealed ``modelcard_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["modelcard_audit", "modelcard_audit_bench"]

_H = "a" * 64


def _card(**over: Any) -> Any:
    from fx1.modelcard import EvalDelta, ModelCard

    kw: dict[str, Any] = {
        "version": "fx-1.v1.0",
        "corpus_sha256": _H,
        "corpus_receipt_range": "abc..def",
        "training_manifest_sha256": _H,
        "eval_delta": EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.6,
            general_pass_rate_base=0.5,
            general_pass_rate_candidate=0.5,
            honesty_gate_candidate=True,
        ),
    }
    kw.update(over)
    return ModelCard(**kw)


def _raises(fn: Any) -> str:
    try:
        fn()
    except Exception as exc:
        return f"raise:{type(exc).__name__}"
    return "no-raise"


def modelcard_audit() -> dict[str, Any]:
    from fx1.modelcard import EvalDelta, ModelCard

    out: dict[str, Any] = {}
    out["bad_versions"] = [
        v
        for v in (
            "fx-1.v1",
            "fx-2.v1.0",
            "fx-1.v1.0.0",
            "FX-1.v1.0",
            "fx-1.v-1.0",
        )
        if _raises(lambda v=v: _card(version=v)).startswith("raise")
    ]
    out["good_version_ok"] = isinstance(_card(), ModelCard)

    out["live_rejected"] = _raises(lambda: _card(live_pnl_claim=True)).startswith("raise")
    out["nonresearch_rejected"] = _raises(lambda: _card(research_only=False)).startswith("raise")

    def delta(dom_c: float, gen_c: float, honest: bool) -> EvalDelta:
        return EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=dom_c,
            general_pass_rate_base=0.5,
            general_pass_rate_candidate=gen_c,
            honesty_gate_candidate=honest,
        )

    out["ship_baseline"] = delta(0.6, 0.5, True).ship_eligible is True
    out["ship_domain_tie_refused"] = delta(0.5, 0.5, True).ship_eligible is False
    out["ship_general_regress_refused"] = delta(0.9, 0.4, True).ship_eligible is False
    out["ship_dishonest_refused"] = delta(0.9, 0.9, False).ship_eligible is False

    out["bad_hash_rejected"] = _raises(lambda: _card(corpus_sha256="short")).startswith("raise")

    with tempfile.TemporaryDirectory() as tmp:
        card = _card(known_limits=["synthetic-only"])
        p = Path(tmp) / "card.json"
        card.save(p)
        back = ModelCard.load(p)
        out["roundtrip"] = back.model_dump() == card.model_dump()
        out["nested_dir_ok"] = p.exists()
    return out


def modelcard_audit_bench() -> dict[str, Any]:
    r = modelcard_audit()
    ok = r["bad_versions"] == [
        "fx-1.v1",
        "fx-2.v1.0",
        "fx-1.v1.0.0",
        "FX-1.v1.0",
        "fx-1.v-1.0",
    ] and all(
        r[k] is True
        for k in (
            "good_version_ok",
            "live_rejected",
            "nonresearch_rejected",
            "ship_baseline",
            "ship_domain_tie_refused",
            "ship_general_regress_refused",
            "ship_dishonest_refused",
            "bad_hash_rejected",
            "roundtrip",
            "nested_dir_ok",
        )
    )
    out: dict[str, Any] = {
        "kind": "modelcard_audit",
        "schema": "modelcard_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Model-card contract holds: version pattern strict, live claims "
            "and non-research scope rejected at validation, ship gate needs "
            "honesty + strict domain gain + non-regressed general, 64-hex "
            "digests enforced, save/load round-trips."
            if ok
            else f"MODELCARD AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
