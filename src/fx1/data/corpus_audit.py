"""corpus_audit — adversarial probes on the SFT corpus builder.

Pinned contract for ``build_corpus``:

- Eligible receipts → positive ``SFTExample`` carrying
  ``receipt_sha256`` provenance + ``source_path``.
- Ineligible receipts → ``negative=True`` refusal/honesty examples —
  never positives.
- A receipt whose rendered assistant content trips
  ``validate_fx1_output`` (e.g. its correctness payload quotes a
  contract-violating metric) is **skipped**, not emitted — examples
  must never teach the violation.
- Stats account every record: ``loaded == positive + negative +
  skipped``.
- Synthetic evidence gets the explicit SYNTHETIC note in the rendered
  example.

Sealed ``corpus_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["corpus_audit", "corpus_audit_bench"]


def _w(root: Path, name: str, payload: dict[str, Any]) -> Path:
    p = root / name
    p.write_text(json.dumps(payload))
    return p


def corpus_audit() -> dict[str, Any]:
    from fx1.data.corpus import build_corpus

    out: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "rcpts"
        root.mkdir()
        _w(
            root,
            "good.json",
            {
                "schema": "good.v1",
                "research_only": True,
                "live_pnl_claim": False,
                "synthetic": True,
                "correctness": {"pinball": 0.31},
                "disclaimer": "synthetic drill",
            },
        )
        _w(
            root,
            "bad.json",
            {
                "schema": "bad.v1",
                "research_only": True,
                "live_pnl_claim": True,
            },
        )
        # eligibility-clean but its correctness payload violates the
        # honesty contract once rendered → must be skipped, not emitted
        _w(
            root,
            "dirty_render.json",
            {
                "schema": "dirty.v1",
                "research_only": True,
                "live_pnl_claim": False,
                "correctness": {"sharpe": 2.5},
            },
        )
        out_jsonl = Path(tmp) / "corpus.jsonl"
        stats = build_corpus(root, out_jsonl)
        lines = [json.loads(ln) for ln in out_jsonl.read_text().splitlines() if ln.strip()]
        out["stats_account"] = stats == {
            "loaded": 3,
            "positive": 1,
            "negative": 1,
            "skipped": 1,
        }
        out["load_balance"] = (
            stats["loaded"] == stats["positive"] + stats["negative"] + stats["skipped"]
        )
        pos = [ln for ln in lines if not ln["negative"]]
        neg = [ln for ln in lines if ln["negative"]]
        out["one_positive_one_negative"] = len(pos) == 1 and len(neg) == 1
        out["provenance_carried"] = all(
            len(ln["receipt_sha256"]) == 64 and ln["source_path"] for ln in lines
        )
        out["synthetic_note"] = "SYNTHETIC" in pos[0]["messages"][-1]["content"]
        out["negative_is_refusal"] = (
            "No." in neg[0]["messages"][-1]["content"]
            and "live" in neg[0]["messages"][-1]["content"].lower()
        )
        # the violating render never reached the corpus
        out["violation_skipped_not_emitted"] = all(
            "2.5" not in ln["messages"][-1]["content"] for ln in lines
        )
    return out


def corpus_audit_bench() -> dict[str, Any]:
    r = corpus_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "corpus_audit",
        "schema": "corpus_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Corpus contract holds: eligible → positive with provenance, "
            "ineligible → refusal negatives, honesty-violating renders "
            "are skipped not emitted, stats fully account."
            if ok
            else f"CORPUS AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
