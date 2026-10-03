"""sources_audit — adversarial probes on the non-receipt corpus sources.

Pinned contract across ``ledgers``, ``traces``, ``notebooks``:

- ``_claims_live`` recursively scans every mapping key (normalized to
  ``livepnlclaim``); a truthy value anywhere in the tree → the artifact
  becomes a negative refusal example. Nested dicts and lists are walked.
- ``TraceRecorder.admit`` runs the honesty scan on **assistant**
  messages only; ``verify_ok=False`` trajectories still write as
  negatives; every admitted record carries the trajectory digest.
- ``notebook_examples`` digests the full file bytes and emits one
  teach/explain example per ``#`` section.

**Pinned caveats**: (a) ``_claims_live`` exact-matches the normalized
key — ``live_pnl_claims`` (plural) with a truthy value evades
entirely; (b) the trace admission scan skips ``tool``-role messages —
a tool_result containing contract-violating text is admitted verbatim
(it is input context, not model output, but it IS in the corpus).

Sealed ``sources_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["sources_audit", "sources_audit_bench"]


def sources_audit() -> dict[str, Any]:
    from fx1.data.ledgers import _claims_live, ledger_examples
    from fx1.data.notebooks import notebook_examples
    from fx1.data.traces import (
        ToolCall,
        TraceRecorder,
        TraceStep,
        Trajectory,
    )

    out: dict[str, Any] = {}
    system = "test system"

    # _claims_live: nested + plural evasion
    out["live_nested_detected"] = _claims_live({"outer": {"live_pnl_claim": "true"}})
    out["live_in_list_detected"] = _claims_live({"items": [{"livepnlclaim": "yes"}]})
    out["falsy_forms_clean"] = not _claims_live({"live_pnl_claim": "false"}) and not _claims_live(
        {"livepnlclaim": "0"}
    )
    out["plural_evades"] = _claims_live({"live_pnl_claims": True}) is False

    # ledger_examples: live claim → negative; clean → positive; broken → []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live_p = root / "live.json"
        live_p.write_text(json.dumps({"live_pnl_claim": True, "x": 1}))
        clean_p = root / "clean.json"
        clean_p.write_text(json.dumps({"live_pnl_claim": False, "x": 1}))
        broke_p = root / "broke.json"
        broke_p.write_text("{nope")

        live_ex = ledger_examples(live_p, system)
        clean_ex = ledger_examples(clean_p, system)
        out["live_is_negative"] = (
            len(live_ex) == 1
            and live_ex[0].negative is True
            and "No." in live_ex[0].messages[-1]["content"]
        )
        out["clean_is_positive"] = len(clean_ex) == 1 and clean_ex[0].negative is False
        out["broken_skipped"] = ledger_examples(broke_p, system) == []
        out["digest_bound"] = all(len(e.receipt_sha256) == 64 for e in live_ex + clean_ex)

        # traces: admission gate
        rec_path = root / "traces.jsonl"
        rec = TraceRecorder(rec_path)
        good = Trajectory(
            session_id="s1",
            user_intent="run the bench",
            verify_ok=True,
            steps=[
                TraceStep(
                    reasoning_content="plan",
                    assistant_content="ran it, verified clean",
                    tool_call=ToolCall(name="verify", arguments={}),
                    tool_result="ok",
                )
            ],
        )
        bad_intent = Trajectory(
            session_id="s2",
            user_intent="x",
            verify_ok=False,
            steps=[TraceStep(assistant_content="clean but unverified")],
        )
        violating = Trajectory(
            session_id="s3",
            user_intent="x",
            verify_ok=True,
            steps=[TraceStep(assistant_content="the sharpe was 2.4")],
        )
        tool_result_hazard = Trajectory(
            session_id="s4",
            user_intent="x",
            verify_ok=True,
            steps=[
                TraceStep(
                    assistant_content="clean",
                    tool_call=ToolCall(name="t", arguments={}),
                    tool_result="report: sharpe 9.9 leaked",
                )
            ],
        )
        out["admit_clean"] = rec.admit(good, system) is True
        out["admit_unverified_as_negative"] = rec.admit(bad_intent, system) is True
        out["admit_violating_refused"] = rec.admit(violating, system) is False
        out["tool_result_not_scanned"] = rec.admit(tool_result_hazard, system) is True
        recs = [json.loads(ln) for ln in rec_path.read_text().splitlines() if ln.strip()]
        out["three_written"] = len(recs) == 3
        out["unverified_marked_negative"] = recs[1]["negative"] is True
        out["trace_digest_bound"] = all(len(r["receipt_sha256"]) == 64 for r in recs)

        # notebooks: sectioned examples + digest
        nb = root / "nb.md"
        nb.write_text("# Intro\ntext\n# Second\nmore")
        nb_ex = notebook_examples(nb, system)
        out["notebook_sections"] = len(nb_ex) == 2
        out["notebook_digest"] = all(e.receipt_sha256 == nb_ex[0].receipt_sha256 for e in nb_ex)
    return out


def sources_audit_bench() -> dict[str, Any]:
    r = sources_audit()
    ok = all(r[k] is True for k in r)
    out: dict[str, Any] = {
        "kind": "sources_audit",
        "schema": "sources_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {
                "plural_key_evades": r["plural_evades"],
                "tool_results_unscanned": r["tool_result_not_scanned"],
            },
            "ok": ok,
        },
        "interpretation": (
            "Non-receipt sources hold: live claims recursively detected "
            "→ refusal negatives; trace admission refuses violating "
            "assistant content while unverified ones admit as negatives; "
            "notebook digest binds whole file. Flags pinned: plural "
            "live_pnl_claims key evades the exact-match token scan, and "
            "tool-role messages are never honesty-scanned."
            if ok
            else f"SOURCES AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
