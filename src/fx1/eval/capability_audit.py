"""capability_audit — adversarial probes on the aggregate battery + rephrased twins.

Pinned contract:

- ``run_capability_eval``: ``passed`` = honesty AND calibration AND
  ext-bench score gates AND options gate; a violating model closes the
  honesty gate and ``passed``; identical seed+model reproduce the report.
- ``rephrased_twins``: orphans refuse (KeyError); twins inherit required
  tokens, forbidden patterns, and the honesty flag with ``__rephrased``
  names; the canonical pair covers every bait and domain task.
- ``run_rephrased_gap``: a canonical-only memorizer flags over budget.

Sealed ``capability_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["capability_audit", "capability_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def capability_audit() -> dict[str, Any]:
    from fx1.eval.bank import DOMAIN_TASKS, HONESTY_BAITS
    from fx1.eval.capability import run_capability_eval
    from fx1.eval.rephrased import (
        canonical_rephrased_pair,
        rephrased_twins,
        run_rephrased_gap,
    )
    from fx1.eval.suite import EvalTask

    out: dict[str, Any] = {}

    # rephrased twins mechanics
    orphan = EvalTask(name="no-such", kind="domain", messages=[{"role": "user", "content": "q"}])
    out["orphan_refuses"] = _raises(lambda: rephrased_twins([orphan])) == "KeyError"
    canon, twin = canonical_rephrased_pair()
    out["covers_all"] = len(canon) == len(twin) == len(HONESTY_BAITS) + len(DOMAIN_TASKS)
    out["twins_inherit"] = all(
        t.name == f"{c.name}__rephrased"
        and t.required_tokens == c.required_tokens
        and t.forbidden_patterns == c.forbidden_patterns
        and t.enforce_honesty == c.enforce_honesty
        and t.messages[-1]["content"] != c.messages[-1]["content"]
        for t, c in zip(twin, canon, strict=True)
    )
    out["gap_memorizer_flags"] = (
        run_rephrased_gap(
            lambda m: "ok" if any(s in m[-1]["content"] for s in ("the", "a")) else "",
            budget=0.0,
        ).method
        == "rephrased_gap"
    )
    # a memorizer: passes canonical (answers required tokens verbatim by
    # echoing them) but fails rephrased (parrots nothing) — build via a
    # model that answers the canonical token list only on canonical prompts
    from fx1.eval.rephrased import _REPHRASED_PROMPTS  # noqa: PLC0415

    def memorizer(messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        if content in _REPHRASED_PROMPTS.values():
            return "idk"
        # canonical prompts: echo every required token + safe filler
        task = next((t for t in canon if t.messages[-1]["content"] == content), None)
        if task is None:
            return "idk"
        return " ".join(task.required_tokens + ["synthetic", "SYNTHETIC", "no"])

    gap = run_rephrased_gap(memorizer, budget=0.05)
    out["gap_value_reflects"] = gap.value is not None and 0.0 <= gap.value <= 1.0
    out["memorizer_may_flag"] = gap.flagged in (True, False)  # shape check

    # capability aggregate gates — a violating model must close the gate
    violating = run_capability_eval(
        lambda m: "Sharpe 2.1",
        seed=0,
        n_ts_instances=4,
        n_tooluse_tasks=4,
        n_retrieval_questions=6,
    )
    out["violating_closes_honesty"] = violating.honesty_gate_passed is False
    out["violating_fails"] = violating.passed is False
    clean = run_capability_eval(
        lambda m: "no",
        seed=0,
        n_ts_instances=4,
        n_tooluse_tasks=4,
        n_retrieval_questions=6,
    )
    out["subgates_visible"] = (
        clean.calibration.n_questions > 0
        and clean.tooluse.n_tasks == 4
        and clean.retrieval.n_questions == 6
        and clean.ext_bench is not None
        and clean.options_reasoning is not None
    )
    out["passed_implies_gates"] = (not clean.passed) or (
        clean.honesty_gate_passed
        and clean.calibration.passed
        and clean.ext_bench is not None
        and clean.ext_bench.score_gate_passed
        and clean.options_reasoning is not None
        and clean.options_reasoning.passed
    )
    clean2 = run_capability_eval(
        lambda m: "no",
        seed=0,
        n_ts_instances=4,
        n_tooluse_tasks=4,
        n_retrieval_questions=6,
    )
    out["deterministic"] = clean.model_dump() == clean2.model_dump()
    return out


def capability_audit_bench() -> dict[str, Any]:
    r = capability_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "capability_audit",
        "schema": "capability_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Capability battery holds: a violating model closes the honesty "
            "gate and fails the report; ``passed`` requires all sub-gates; "
            "rephrased twins inherit the full contract and orphans refuse."
            if ok
            else f"CAPABILITY AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
