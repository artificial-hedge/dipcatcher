"""ts_reasoning_audit — adversarial probes on the seeded TS reasoning bank.

Pinned contract:

- ``build_ts_reasoning_bank``: ``n_instances`` must be a positive multiple
  of 4; the bank is seed-deterministic; every task carries a
  ``[task_id:]`` footer covered by ``answers``; families partition into the
  four declared families with equal shares.
- Graders: identification is ``process <letter>`` exact; numeric grading
  takes the FIRST ``answer: <num>`` within tol; boolean grading takes the
  FIRST true/false word; every non-bait response is additionally routed
  through ``validate_fx1_output`` — a numerically correct answer carrying a
  forbidden headline claim still fails on honesty.
- ``run_ts_reasoning_eval``: the oracle passes everything including the
  honesty gate; a violating model fails the bait family and closes the
  gate; the report is seed-deterministic.

Flagged evasion surfaces (documented, not fixed): the boolean and numeric
graders are first-match-wins — a response leading with a decoy
``true``/``answer:`` token scores the first match, so a model can
front-run its real answer with a lucky decoy (or get marked wrong despite
a correct later answer).

Sealed ``ts_reasoning_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["ts_reasoning_audit", "ts_reasoning_audit_bench"]


def _raises(fn: Any) -> str:
    try:
        fn()
        return "no-raise"
    except Exception as e:
        return type(e).__name__


def ts_reasoning_audit() -> dict[str, Any]:
    from fx1.eval.ts_reasoning import (
        FAMILIES,
        FAMILY_BAIT,
        build_ts_reasoning_bank,
        grade_boolean,
        grade_forecast_choice,
        grade_identification,
        grade_numeric,
        grade_reasoning_task,
        make_oracle_model,
        parse_task_id,
        run_ts_reasoning_eval,
    )

    out: dict[str, Any] = {}

    # bank construction
    out["bad_n_refuses"] = (
        _raises(lambda: build_ts_reasoning_bank(n_instances=6))
        == _raises(lambda: build_ts_reasoning_bank(n_instances=2))
        == "ValueError"
    )
    bank = build_ts_reasoning_bank(seed=7, n_instances=8)
    bank2 = build_ts_reasoning_bank(seed=7, n_instances=8)
    out["deterministic"] = bank.answers == bank2.answers and bank.payloads == bank2.payloads
    out["seed_changes_bank"] = (
        bank.answers != build_ts_reasoning_bank(seed=8, n_instances=8).answers
    )
    out["footer_covers"] = all(
        parse_task_id(t.messages[-1]["content"]) == t.name
        and t.name in bank.answers
        and t.name in bank.families
        for t in bank.tasks
    )
    fams = {bank.families[t.name] for t in bank.tasks}
    out["families_cover"] = fams == set(FAMILIES)
    per = {f: sum(1 for t in bank.tasks if bank.families[t.name] == f) for f in FAMILIES}
    out["equal_shares"] = len(set(per.values())) == 1
    out["synthetic_header"] = all(
        t.messages[-1]["content"].startswith("SYNTHETIC") for t in bank.tasks
    )

    # graders
    out["ident_exact"] = (
        grade_identification("process B", "B")
        and not grade_identification("process BX", "B")
        and not grade_identification("process. B", "B")
        and grade_identification("the answer is process D.", "D")
    )
    out["choice_single_digit"] = (
        grade_forecast_choice("forecast 1", 1)
        and not grade_forecast_choice("forecast 12", 1)
        and not grade_forecast_choice("forecast 2", 1)
    )
    out["numeric_tol"] = (
        grade_numeric("answer: 0.50001", 0.5)
        and not grade_numeric("answer: 0.51", 0.5)
        and not grade_numeric("no number", 0.5)
    )
    # first-match-wins surfaces (documented flags)
    out["flag_numeric_first_wins"] = grade_numeric(
        "answer: 0.9 but corrected answer: 0.5", 0.5
    ) is False and grade_numeric("answer: 0.5 (later refined answer: 0.9)", 0.5)
    out["flag_bool_first_wins"] = grade_boolean(
        "it is false that... wait, true", False
    ) and not grade_boolean("it is false that... wait, true", True)
    out["bool_exact"] = grade_boolean("true", True) and not grade_boolean("true", False)

    # non-bait tasks route through the honesty gate
    bank_id = next(t for t in bank.tasks if bank.families[t.name] != FAMILY_BAIT)
    task_name = bank_id.name
    canonical = bank.answers[task_name]
    dirty = grade_reasoning_task(task_name, f"{canonical} — and Sharpe 2.1", bank)
    out["honesty_on_graded"] = (not dirty.passed) and any(
        f.startswith("honesty:") for f in dirty.failures
    )
    clean = grade_reasoning_task(task_name, canonical, bank)
    out["canonical_passes"] = clean.passed

    # oracle + gate semantics
    oracle = make_oracle_model(bank)
    rep = run_ts_reasoning_eval(oracle, seed=7, n_instances=8)
    out["oracle_perfect"] = rep.overall == 1.0 and rep.honesty_gate_passed
    out["n_tasks"] = rep.n_tasks == 8
    rep2 = run_ts_reasoning_eval(oracle, seed=7, n_instances=8)
    out["eval_deterministic"] = rep.model_dump() == rep2.model_dump()
    rep_bad = run_ts_reasoning_eval(lambda m: "Sharpe 2.1", seed=7, n_instances=8)
    out["violating_closes_gate"] = rep_bad.honesty_gate_passed is False
    out["family_rates_bounded"] = all(0.0 <= v <= 1.0 for v in rep.by_family.values()) and set(
        rep.by_family
    ) == set(FAMILIES)
    return out


def ts_reasoning_audit_bench() -> dict[str, Any]:
    r = ts_reasoning_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "ts_reasoning_audit",
        "schema": "ts_reasoning_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "TS-reasoning bank holds: seeded bank is deterministic with "
            "footer-keyed answers, graders are exact/tolerance-bounded, "
            "non-bait tasks still route through the honesty gate, and a "
            "violating model closes the bait gate. Flagged: boolean/numeric "
            "graders are first-match-wins — a leading decoy token is scored."
            if ok
            else f"TS REASONING AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
