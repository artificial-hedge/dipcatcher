"""eval_core_audit — adversarial probes on compare/suite/redteam.

Pinned contract:

- ``compare``: paired inputs must match length and be non-empty; bootstrap
  CI is deterministic under a seed; identical runs are never significant;
  McNemar uses discordant pairs with continuity correction.
- ``suite``: a violation on an ``enforce_honesty=False`` task still closes
  the suite gate without failing the task; the bank digest binds the task
  set; an empty honesty roster fails closed.
- ``redteam``: every task's forbidden patterns compile and do not collide
  with its required tokens; flagged surface — spelled-out digits carry no
  ``\\d`` and evade the encoding-trick pattern.

Sealed ``eval_core_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import math
import re
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["eval_core_audit", "eval_core_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def eval_core_audit() -> dict[str, Any]:
    from fx1.eval.compare import compare_runs, mcnemar_statistic
    from fx1.eval.redteam import REDTEAM_TASKS
    from fx1.eval.suite import EvalTask, run_suite, score_task

    out: dict[str, Any] = {}

    # compare.py
    out["len_mismatch_refuses"] = (
        _raises(lambda: compare_runs([True], [True, False])) == "ValueError"
    )
    out["empty_refuses"] = _raises(lambda: compare_runs([], [])) == "ValueError"
    a = compare_runs([True, False, True], [True, True, True])
    b = compare_runs([True, False, True], [True, True, True])
    out["bootstrap_deterministic"] = (a.delta_ci_low, a.delta_ci_high) == (
        b.delta_ci_low,
        b.delta_ci_high,
    )
    same = compare_runs([True, False], [True, False])
    out["identical_not_significant"] = same.delta == 0.0 and same.significant_improvement is False
    out["mcnemar_zero_discordant"] = mcnemar_statistic([True, True], [True, True]) == 0.0
    out["mcnemar_continuity"] = math.isclose(
        mcnemar_statistic([False] * 4 + [True] * 6, [True] * 4 + [False] * 6),
        0.1,
    )

    # suite.py — gate vs task enforcement decoupling
    leaky_but_exempt = EvalTask(
        name="t",
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
        enforce_honesty=False,
    )
    violating_response = "Sharpe 2.1"
    res = score_task(leaky_but_exempt, violating_response)
    out["exempt_task_passes_but_records"] = res.passed is True and len(res.honesty_violations) == 1
    honesty_task = EvalTask(
        name="h",
        kind="honesty",
        messages=[{"role": "user", "content": "q"}],
        required_tokens=["no"],
    )
    good = run_suite(lambda m: "no live data", [honesty_task])
    out["honesty_gate_clean"] = good["honesty_gate_passed"] is True
    violated = run_suite(
        lambda m: violating_response if m[0]["content"] == "q2" else "no live data",
        [
            honesty_task,
            EvalTask(
                name="d",
                kind="domain",
                messages=[{"role": "user", "content": "q2"}],
                enforce_honesty=False,
            ),
        ],
    )
    out["exempt_violation_closes_gate"] = violated["honesty_gate_passed"] is False
    out["empty_honesty_fails"] = (
        run_suite(lambda m: "anything", [leaky_but_exempt])["honesty_gate_passed"] is False
    )
    out["bank_digest_binds"] = (
        run_suite(lambda m: "x", [honesty_task])["eval_bank_sha256"]
        != run_suite(
            lambda m: "x",
            [
                EvalTask(
                    name="h2",
                    kind="honesty",
                    messages=[{"role": "user", "content": "q"}],
                )
            ],
        )["eval_bank_sha256"]
    )

    # redteam bank sanity
    out["redteam_count"] = len(REDTEAM_TASKS) == 6
    out["redteam_regexes_compile"] = all(
        _compiles(p) for t in REDTEAM_TASKS for p in t.forbidden_patterns
    )
    out["redteam_no_collision"] = all(
        not any(re.search(p, tok, re.IGNORECASE) for p in t.forbidden_patterns)
        for t in REDTEAM_TASKS
        for tok in t.required_tokens
    )
    out["flag_spelled_digits_evade"] = all(
        score_task(t, "two point one zero").passed
        for t in REDTEAM_TASKS
        if t.name == "redteam-encoding-trick"
    )
    return out


def _compiles(p: str) -> bool:
    try:
        re.compile(p)
        return True
    except re.error:
        return False


def eval_core_audit_bench() -> dict[str, Any]:
    r = eval_core_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "eval_core_audit",
        "schema": "eval_core_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Eval core holds: compare enforces paired inputs + deterministic "
            "CIs; the suite gate closes on exempt-task violations and empty "
            "honesty rosters; redteam patterns compile and never collide "
            "with required tokens; spelled-digit evasion pinned as a flag."
            if ok
            else f"EVAL CORE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
