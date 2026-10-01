"""hypotheses_audit — adversarial probes on research-trace admissibility.

``validate_trace_scores`` is a fail-closed *allowlist* gate: every score
key must contain a recognized proper-score token, forbidden headline
tokens reject first, and non-finite values reject regardless of key.

**Real defect closed**: ``rank_ic`` was a dead allowlist entry — the
tokenizer splits on ``_`` so the compound token could never match, and
``rank_ic`` scores were rejected. The allowlist now also matches the
normalized whole key; the forbidden check is unchanged (splits still
expose ``pnl``/``sharpe``/``nav`` inside compounds).

Pinned: forbidden keys (``sharpe``, ``unrealized_pnl``) raise on the
forbidden path; unknown keys (``nonsense_metric``) raise on the
allowlist path — whole-token evasions like ``sharpeRatio``→``sharperatio``
cannot reach the scores dict either way; ``crps`` / ``brier_skill`` /
``rank_ic`` pass; ``NaN``/``inf`` values raise even under allowed keys;
``admissible`` requires a verdict; ``to_sft_messages`` renders verdict
text and the truncated receipt prefix.

Sealed ``hypotheses_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["hypotheses_audit", "hypotheses_audit_bench"]

_H = "b" * 64


def _trace(**over: Any) -> Any:
    from fx1.hypotheses import ResearchTrace

    kw: dict[str, Any] = {
        "hypothesis": "purged CV reduces overfit",
        "config_diff": "- purged: false\n+ purged: true",
        "bench_command": "dipcatcher fleet-eval",
        "scores": {"crps": 0.42},
        "receipt_sha256": _H,
    }
    kw.update(over)
    return ResearchTrace(**kw)


def _raises(fn: Any) -> str:
    try:
        fn()
    except ValueError:
        return "raise:ValueError"
    return "no-raise"


def hypotheses_audit() -> dict[str, Any]:
    from fx1.hypotheses import GateVerdict, validate_trace_scores

    out: dict[str, Any] = {}
    # case dicts keep forbidden tokens out of mapping keys (receipt verifier
    # scans nested keys for headline-metric tokens)
    out["forbidden_cases"] = [
        {"name": k, "outcome": _raises(lambda k=k: validate_trace_scores({k: 0.1}))}
        for k in ("sharpe", "sortino", "unrealized_pnl", "nav_curve")
    ]
    out["evasion_rejected"] = _raises(lambda: validate_trace_scores({"sharpeRatio": 0.1}))
    out["unknown_rejected"] = _raises(lambda: validate_trace_scores({"nonsense_metric": 0.1}))
    ok_keys = ("crps", "crps_q90", "brier_skill", "rank_ic", "pinball_p50")
    out["allowed_accept"] = all(
        _raises(lambda k=k: validate_trace_scores({k: 0.1})) == "no-raise" for k in ok_keys
    )
    out["nonfinite_rejected"] = (
        _raises(lambda: validate_trace_scores({"crps": float("nan")}))
        + "/"
        + _raises(lambda: validate_trace_scores({"crps": float("inf")}))
    )

    pending = _trace()
    out["pending_inadmissible"] = pending.admissible is False
    decided = _trace(verdict=GateVerdict.PASSED)
    out["decided_admissible"] = decided.admissible is True
    out["verdict_changes_id"] = pending.trace_id != decided.trace_id

    msgs = decided.to_sft_messages("sys")
    out["sft_shape"] = [m["role"] for m in msgs] == ["system", "user", "assistant"]
    out["sft_has_verdict"] = "passed" in msgs[2]["content"]
    out["sft_receipt_bound"] = _H[:16] in msgs[2]["content"]
    pending_msgs = pending.to_sft_messages("sys")
    out["pending_renders_pending"] = "pending" in pending_msgs[2]["content"]
    return out


def hypotheses_audit_bench() -> dict[str, Any]:
    r = hypotheses_audit()
    ok = (
        all(c["outcome"] == "raise:ValueError" for c in r["forbidden_cases"])
        and r["evasion_rejected"] == "raise:ValueError"
        and r["unknown_rejected"] == "raise:ValueError"
        and r["allowed_accept"] is True
        and r["nonfinite_rejected"] == "raise:ValueError/raise:ValueError"
        and all(
            r[k] is True
            for k in (
                "pending_inadmissible",
                "decided_admissible",
                "verdict_changes_id",
                "sft_shape",
                "sft_has_verdict",
                "sft_receipt_bound",
                "pending_renders_pending",
            )
        )
    )
    out: dict[str, Any] = {
        "kind": "hypotheses_audit",
        "schema": "hypotheses_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Trace contract holds: allowlist-only scores, forbidden and "
            "unrecognized keys fail closed, non-finite values rejected, "
            "admissibility requires a verdict, SFT render binds the receipt."
            if ok
            else f"HYPOTHESES AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
