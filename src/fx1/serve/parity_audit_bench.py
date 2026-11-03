"""Sealed receipt for :func:`fx1.serve.parity_audit.parity_audit`.

Split out of the audit module so that file stays inside its line budget.
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision


def parity_audit_bench() -> dict[str, Any]:
    from fx1.serve.parity_audit import parity_audit

    r = parity_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "parity_audit",
        "schema": "parity_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "three-surface parity holds: SDK, HTTP API, and HarnessClient "
            "return byte-identical content, envelopes, chunk lists, error "
            "classes, verifier verdicts, registry, and health over the same "
            "injected backend; the in-flight cap fails saturated work with "
            "503+Retry-After while cheap routes respond, and releases "
            "cleanly. Both surfaces keep a per-call completion log whose "
            "records share prompt/output sha256s (the completion_id itself "
            "is a per-surface mint — not part of the parity claim). Each "
            "surface also exports its own logged call as a sealed "
            "fx1_completion_record.v1 receipt — same record hashes, each "
            "doc verifiable through either surface's verifier. A terminal "
            "wire job's fx1_job_record.v1 embeds the same digested run "
            "result that Fx1Harness.run_receipt seals in-process as "
            "fx1_run_result.v1 — identical on the shared fields. "
            "Fine-tuning parity holds: the SDK's synchronous "
            "create_finetune_job produces the same terminal job fields "
            "(status, ft:-name grammar, result_files, event feed) as the "
            "wire route under the same runner, and guards map "
            "ValueError↔HarnessTransportError / KeyError↔KeyError. The "
            "model registry holds on both surfaces: a succeeded job's ft: "
            "name lists and retrieves, completions naming it resolve to "
            "local_fx1 at the job's checkpoint, and unregistered ft: names "
            "fail closed (OpenAICompatError in-process, 404 on the wire). "
            "Flags: "
            "unknown backend names are KeyError in-process "
            "vs 422 literal rejection over the wire (request validation "
            "runs before resolution); empty batches are [] in-process vs "
            "422 over the wire."
            if ok
            else f"PARITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
