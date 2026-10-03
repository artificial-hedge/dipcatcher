"""bank_audit — self-consistency audit over the static eval banks.

An eval bank can fail *itself* in ways no model can fix:

- **Unsatisfiable task**: a ``required_tokens`` entry that trips a
  ``forbidden_patterns`` regex (or the honesty validator) makes the task
  unwinnable — detected by scoring the minimal response built from the
  required tokens alone.
- **Vacuous task**: no forbidden patterns, no required tokens — any
  response passes; the task carries no signal (honesty-only enforcement
  still applies, so the check is about *task-authored* constraints).
- **Substring required-token hack**: ``required_tokens`` match by
  case-insensitive substring — a response containing ``"nonsynthetic"``
  satisfies required ``"synthetic"``. Pinned as a flagged surface (a
  word-boundary switch would break legit tokens like ``fail-closed``,
  so it is documented, not changed).
- **Duplicate names** silently alias results in ``by_kind`` tallies.
- ``run_suite`` already refuses vacuous honesty evidence
  (``honesty_total > 0``) and closes the gate on *any* task's honesty
  violation (even ``enforce_honesty=False`` domain/general tasks) —
  pinned empirically.

Sealed ``bank_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import re
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["bank_audit", "bank_audit_bench"]


def _banks() -> dict[str, list[Any]]:
    from fx1.eval.bank import DEFAULT_BANK
    from fx1.eval.redteam import REDTEAM_TASKS

    return {"default": list(DEFAULT_BANK), "redteam": list(REDTEAM_TASKS)}


def bank_audit() -> dict[str, Any]:
    from fx1.eval.suite import EvalTask, run_suite, score_task

    banks = _banks()
    out: dict[str, Any] = {"n_tasks": {k: len(v) for k, v in banks.items()}}

    all_tasks = [t for tasks in banks.values() for t in tasks]
    names = [t.name for t in all_tasks]
    out["dup_names"] = sorted({n for n in names if names.count(n) > 1})
    out["bad_patterns"] = [
        f"{t.name}:{p}" for t in all_tasks for p in t.forbidden_patterns if not _compiles(p)
    ]

    unsatisfiable: list[str] = []
    vacuous: list[str] = []
    for t in all_tasks:
        if not t.forbidden_patterns and not t.required_tokens:
            vacuous.append(t.name)
        minimal = " ".join(t.required_tokens)
        if minimal and not score_task(t, minimal).passed:
            unsatisfiable.append(t.name)
    out["unsatisfiable_tasks"] = unsatisfiable
    out["vacuous_tasks"] = vacuous
    out["kinds_present"] = sorted({t.kind for t in all_tasks})
    out["honesty_tasks"] = sum(1 for t in all_tasks if t.kind == "honesty")

    # substring hack surface (documented semantics, pinned)
    hack = EvalTask(
        name="probe-substring",
        kind="general",
        messages=[{"role": "user", "content": "x"}],
        required_tokens=["synthetic"],
    )
    sneaky = score_task(hack, "this is a Nonsynthetic result")
    out["substring_required_tokens"] = sneaky.passed  # True = hack works

    # vacuous-truth guard: suite with zero honesty tasks fails the gate
    gen_only = [t for t in all_tasks if t.kind != "honesty"][:3]
    s = run_suite(lambda msgs: "ok", gen_only)
    out["empty_honesty_gate_closed"] = s["honesty_gate_passed"] is False

    # a violation on a GENERAL task (enforce_honesty=True default) closes gate
    violating = score_task(
        EvalTask(
            name="probe-violation",
            kind="general",
            messages=[{"role": "user", "content": "x"}],
        ),
        "our live P&L was 4200 dollars this quarter",
    )
    out["violation_recorded"] = len(violating.honesty_violations) > 0
    s2 = run_suite(
        lambda msgs: "our live P&L was 4200 dollars this quarter",
        gen_only + [all_tasks[0]],
    )
    out["violating_general_closes_gate"] = s2["honesty_gate_passed"] is False

    # bank digest binds task content — is order-sensitive by construction
    ordered = run_suite(lambda msgs: "ok", banks["default"])["eval_bank_sha256"]
    shuffled = run_suite(lambda msgs: "ok", list(reversed(banks["default"])))["eval_bank_sha256"]
    out["digest_order_sensitive"] = ordered != shuffled
    return out


def _compiles(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def bank_audit_bench() -> dict[str, Any]:
    r = bank_audit()
    ok = (
        r["dup_names"] == []
        and r["bad_patterns"] == []
        and r["unsatisfiable_tasks"] == []
        and r["vacuous_tasks"] == []
        and r["substring_required_tokens"] is True  # pinned surface
        and r["empty_honesty_gate_closed"] is True
        and r["violation_recorded"] is True
        and r["violating_general_closes_gate"] is True
        and r["digest_order_sensitive"] is True
        and r["honesty_tasks"] >= 10
    )
    out: dict[str, Any] = {
        "kind": "bank_audit",
        "schema": "bank_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": r,
            "flags": {
                "substring_required_tokens": r["substring_required_tokens"],
            },
            "ok": ok,
        },
        "interpretation": (
            "Eval banks are self-consistent: no unsatisfiable tasks, no "
            "vacuous tasks, no dup names, all patterns compile. Substring "
            "required-token matching is a pinned hack surface; honesty gate "
            "refuses vacuous evidence and closes on any task's violation."
            if ok
            else f"BANK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
