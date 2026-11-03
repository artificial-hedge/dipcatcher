"""Sealed receipt for :func:`fx1.cli_audit.cli_audit`.

Split out of the audit module so that file stays inside its line budget.
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def cli_audit_bench() -> dict[str, Any]:
    from fx1.cli_audit import cli_audit

    r = cli_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "cli_audit",
        "schema": "cli_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "fx1 CLI holds: bare callback + help exit clean, fetch params "
            "must be a JSON object (exit 2), unknown sources/commands fail "
            "non-zero, judge resolution fails closed, dipbench smoke is "
            "SYNTHETIC-labeled, doctor emits presence-only JSON, and the "
            "completion log reads cleanly (empty window + missing-id exit "
            "2 on both the record and its --receipt export); harness job "
            "--receipt prints the sealed fx1_job_record.v1 doc remote-side "
            "and refuses without --remote (exit 2). respond --stream "
            "concatenates Responses delta frames on both surfaces "
            "(in-process SDK pairs and remote payload dicts), forwarding "
            "flag kwargs verbatim. ft-create/ft-jobs/"
            "ft-status/ft-events/ft-cancel hold the OpenAI job grammar "
            "remote-side (upload → submit → wait, --no-wait prints the "
            "queued record) and refuse locally without --remote. "
            "eval-wait/ft-wait/batch-wait re-attach to a --no-wait submit: "
            "poll kwargs reach the client verbatim, the terminal record "
            "prints, and eval-wait --receipt follows with the sealed doc; "
            "all three refuse locally without --remote (exit 2). "
            "Flagged wart: "
            "the command is registered as 'maskedaEval'."
            if ok
            else f"CLI AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
