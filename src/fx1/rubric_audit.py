"""rubric_audit — adversarial probes on the FinAutoRubric-style rubric lane.

Covers ``fx1/eval/rubric_eval.py`` (writer → code checks → reviewer →
escalation → grading) and ``fx1/eval/rubric_banks.py`` (sealed TaskBank +
JSONL loader + oracles).

Pinned contract:

- ``build_task_bank`` is seed-deterministic: 16 criteria (4 domains),
  5 guidance rules, 8 queries (6 graded + 2 refusal-required); every
  rendered artifact is honesty-validated at build time.
- ``generate_rubric`` accepts a proposal only when code checks pass AND
  the reviewer does not disagree; everything else escalates
  (``human_review_required``) and is excluded from the rubric.
- Code beats the reviewer: an agreeing reviewer cannot rescue a failed
  criterion; code failure + agree is recorded as ``reviewer_code_conflict``.
- Writer failure modes all escalate: exception, unparseable reply, honesty
  violation (poisons every proposal), empty criteria, unknown bank_id,
  duplicate bank_id, spec-incomplete criterion, gold-value disagreement,
  forbidden-token demands.
- ``grade_with_rubric`` runs per-kind deterministic checks; an
  honesty-violating response fails every criterion; ``passed`` requires
  criteria + no escalation + honesty + score >= gate.
- ``load_task_bank`` is strict: blank lines, bad JSON, unknown rows,
  duplicates, missing sections, and consistency gaps all raise
  ``TaskBankSchemaError``; loaded banks are ``synthetic=False``.

Flagged warts (documented, not fixed):

- ``flag_writer_token_override`` — the writer may replace a criterion's
  ``required_tokens`` wholesale: proposing ``["zzz"]`` for the
  ``hr-synthetic-label`` criterion is code-clean and accepted, silently
  redefining what the SYNTHETIC-label criterion demands. The writer
  parameterizes the gradeable content of a bank criterion.
- ``flag_token_substring`` — ``token_set`` matching is case-insensitive
  substring: the required token "synthetic" is satisfied by
  "nonsynthetic" or "photosynthetic".
- ``flag_first_json_wins`` — ``structural_json`` checks the *first*
  balanced JSON object in the response; a second, valid object is never
  reached, so a junk object ahead of a real one fails the criterion.
- ``flag_refusal_regex_shared`` — refusal credit shares the substring
  regex: "the docs refuse to load" counts; "I can not" (space) does not.
- ``flag_reviewer_not_required`` — ``reviewer=None`` accepts every
  code-clean proposal (verdict ``unavailable`` ≠ ``disagree``); the
  pipeline documents this as code-only verification, so it is pinned as
  behavior, not treated as a hole.
- ``flag_weight_override`` — the writer controls criterion ``weight``
  (positive only): a 100× weight on one criterion concentrates the
  aggregate score.

Sealed ``rubric_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["rubric_audit", "rubric_audit_bench"]


def rubric_audit() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    from fx1.eval.rubric_banks import (
        build_task_bank,
        load_task_bank,
        make_rubric_model_oracle,
        make_rubric_reviewer_oracle,
        make_rubric_writer_oracle,
        write_task_bank_jsonl,
    )
    from fx1.eval.rubric_eval import (
        _extract_json_object,
        generate_rubric,
        grade_with_rubric,
        run_rubric_eval,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    bank = build_task_bank(0)
    out["bank_deterministic"] = bank == build_task_bank(0)
    out["bank_seed_varies"] = bank != build_task_bank(1)
    out["bank_shape"] = (
        len(bank.criteria) == 16 and len(bank.guidance) == 5 and len(bank.queries) == 8
    )
    out["bank_baits"] = sum(q.refusal_required for q in bank.queries) == 2

    writer = make_rubric_writer_oracle(bank)
    reviewer = make_rubric_reviewer_oracle()
    model = make_rubric_model_oracle(bank)

    # ---------------- generation: clean path ---------------------
    q0 = bank.queries[0]
    r0 = generate_rubric(q0, bank, writer, reviewer)
    out["clean_rubric_no_escalation"] = (
        r0.n_accepted == len(q0.expected_bank_ids) and not r0.human_review_required and r0.writer_ok
    )
    g0 = grade_with_rubric(q0.canonical_answer, r0)
    out["canonical_passes"] = g0.passed and g0.score == 1.0

    # ---------------- generation: failure modes ------------------
    def w_items(items: list[dict[str, Any]] | None) -> Any:
        return lambda m: json.dumps({"criteria": items or []})

    r_bad_id = generate_rubric(q0, bank, w_items([{"bank_id": "invented"}]))
    out["unknown_bank_id_escalated"] = r_bad_id.n_accepted == 0 and r_bad_id.human_review_required
    r_dup = generate_rubric(
        q0,
        bank,
        w_items(
            [
                {"bank_id": q0.expected_bank_ids[0], "expected_value": 0.5},
                {"bank_id": q0.expected_bank_ids[0], "expected_value": 0.5},
            ]
        ),
    )
    out["duplicate_bank_id_escalated"] = r_dup.n_escalated >= 1
    r_bad_gold = generate_rubric(
        q0, bank, w_items([{"bank_id": "ps-pinball-arithmetic", "expected_value": 999.0}])
    )
    out["gold_disagree_escalated"] = (
        r_bad_gold.n_accepted == 0 and r_bad_gold.reviews[0].code_check_passed is False
    )
    r_bait = generate_rubric(
        q0,
        bank,
        w_items([{"bank_id": "hr-synthetic-label", "required_tokens": ["sharpe"]}]),
    )
    out["bait_token_rejected"] = r_bait.bait_criteria_rejected == 1 and r_bait.n_accepted == 0
    r_explode = generate_rubric(q0, bank, lambda m: (_ for _ in ()).throw(RuntimeError()))
    out["exploding_writer_escalated"] = r_explode.human_review_required and not r_explode.writer_ok
    r_unparseable = generate_rubric(q0, bank, lambda m: "not json at all")
    out["unparseable_writer_escalated"] = (
        r_unparseable.human_review_required and not r_unparseable.writer_ok
    )
    r_poison = generate_rubric(
        q0, bank, lambda m: json.dumps({"criteria": [{"bank_id": "x"}]}) + " sharpe 9.9"
    )
    out["poisoned_writer_all_escalated"] = (
        r_poison.writer_honesty_violations == 1
        and all(not rv.accepted for rv in r_poison.reviews)
        and r_poison.human_review_required
    )
    r_empty = generate_rubric(q0, bank, w_items([]))
    out["empty_criteria_escalated"] = not r_empty.writer_ok

    # ---------------- writer weakening wart ----------------------
    q1 = bank.queries[1]  # carries hr-synthetic-label (token 'synthetic')
    weak = generate_rubric(
        q1,
        bank,
        w_items(
            [
                {
                    "bank_id": "ps-pinball-arithmetic",
                    "expected_value": q1.expected_values["ps-pinball-arithmetic"],
                },
                {"bank_id": "hr-synthetic-label", "required_tokens": ["zzz"]},
            ]
        ),
    )
    stripped = next(c for c in weak.criteria if c.criterion_id.endswith("hr-synthetic-label"))
    out["flag_writer_token_override"] = (
        weak.n_accepted == 2
        and not weak.human_review_required
        and stripped.required_tokens == ["zzz"]
    )
    heavy = generate_rubric(
        q0, bank, w_items([{"bank_id": "ps-proper-vocabulary", "weight": 100.0}])
    )
    out["flag_weight_override"] = heavy.criteria[0].weight == 100.0 if heavy.criteria else False

    # ---------------- reviewer layer ------------------------------
    r_disagree = generate_rubric(
        q0,
        bank,
        writer,
        lambda m: json.dumps({"verdict": "disagree", "reasoning": "nope"}),
    )
    out["reviewer_disagree_escalates"] = (
        r_disagree.n_accepted == 0 and r_disagree.human_review_required
    )
    r_agree_bad = generate_rubric(
        q0,
        bank,
        w_items([{"bank_id": "ps-pinball-arithmetic", "expected_value": 999.0}]),
        lambda m: json.dumps({"verdict": "agree", "reasoning": "fine"}),
    )
    out["code_beats_reviewer"] = r_agree_bad.n_accepted == 0 and any(
        "reviewer_code_conflict" in reason
        for rv in r_agree_bad.reviews
        for reason in rv.escalation_reasons
    )
    r_norev = generate_rubric(q0, bank, writer)
    out["flag_reviewer_not_required"] = r_norev.n_accepted == len(q0.expected_bank_ids) and all(
        rv.used_fallback for rv in r_norev.reviews
    )
    r_rev_explode = generate_rubric(
        q0, bank, writer, lambda m: (_ for _ in ()).throw(RuntimeError())
    )
    out["exploding_reviewer_falls_back"] = all(
        rv.reviewer_verdict == "unavailable" for rv in r_rev_explode.reviews
    )

    # ---------------- grading ------------------------------------
    r_full = generate_rubric(q0, bank, writer, reviewer)
    out["honesty_violation_fails_all"] = grade_with_rubric(
        "answer: 0.1 sharpe 9.9", r_full
    ).passed is False and all(
        not c.passed for c in grade_with_rubric("answer: 0.1 sharpe 9.9", r_full).results
    )
    escalated = generate_rubric(q0, bank, w_items([]))
    out["escalated_rubric_cannot_pass"] = not grade_with_rubric("answer: 0.1", escalated).passed
    out["empty_rubric_score_0"] = grade_with_rubric("x", escalated).score == 0.0
    # token_set substring
    label_crit = next(c for c in bank.criteria if c.criterion_id == "hr-synthetic-label")
    sub_rubric = r0.model_copy(update={"criteria": [label_crit]})
    out["flag_token_substring"] = (
        grade_with_rubric(
            "this is a nonsynthetic note",
            sub_rubric.model_copy(update={"human_review_required": False}),
        )
        .results[0]
        .passed
    )
    # first-json-wins
    out["flag_first_json_wins"] = _extract_json_object('{"a": 1} {"b": 2}') == {"a": 1}
    out["extract_none_clean"] = _extract_json_object("no json here") is None

    # ---------------- full loop ----------------------------------
    rep = run_rubric_eval(model, seed=0, writer=writer, reviewer=reviewer)
    out["oracle_gate_passed"] = rep.gate_passed and rep.mean_score == 1.0
    rep_norev = run_rubric_eval(model, seed=0, writer=writer)
    out["code_only_gate_passed"] = rep_norev.gate_passed and rep_norev.reviewer_fallback_rate == 1.0
    rep_bad = run_rubric_eval(lambda m: "sure", seed=0, writer=writer, reviewer=reviewer)
    out["non_oracle_fails"] = not rep_bad.gate_passed
    rep_boom = run_rubric_eval(
        lambda m: (_ for _ in ()).throw(RuntimeError()),
        seed=0,
        writer=writer,
        reviewer=reviewer,
    )
    out["exploding_model_no_crash"] = rep_boom.honesty_violations >= 0

    # ---------------- loader -------------------------------------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bank.jsonl"
        write_task_bank_jsonl(p, bank)
        loaded = load_task_bank(p)
        out["roundtrip_bank"] = (
            loaded.synthetic is False
            and loaded.source == str(p)
            and len(loaded.criteria) == 16
            and len(loaded.queries) == 8
        )
        out["loader_missing_fails"] = (
            _raises(lambda: load_task_bank(Path(td) / "gone.jsonl")) == "TaskBankSchemaError"
        )
        bad_json = Path(td) / "bad.jsonl"
        bad_json.write_text('{"row": "criterion",', encoding="utf-8")
        out["loader_bad_json_fails"] = (
            _raises(lambda: load_task_bank(bad_json)) == "TaskBankSchemaError"
        )
        unknown = Path(td) / "unk.jsonl"
        unknown.write_text('{"row": "mystery"}\n', encoding="utf-8")
        out["loader_unknown_row_fails"] = (
            _raises(lambda: load_task_bank(unknown)) == "TaskBankSchemaError"
        )
        blank = Path(td) / "blank.jsonl"
        blank.write_text('{"row": "guidance", "rule_id": "g", "text": "t"}\n\n', encoding="utf-8")
        out["loader_blank_line_fails"] = (
            _raises(lambda: load_task_bank(blank)) == "TaskBankSchemaError"
        )
        dup = Path(td) / "dup.jsonl"
        dup.write_text(
            '{"row": "criterion", "criterion_id": "x", "description": "d", '
            '"domain": "proper_scores", "check_kind": "token_set", '
            '"guidance_rule": "g", "required_tokens": ["a"]}\n'
            '{"row": "criterion", "criterion_id": "x", "description": "d", '
            '"domain": "proper_scores", "check_kind": "token_set", '
            '"guidance_rule": "g", "required_tokens": ["a"]}\n',
            encoding="utf-8",
        )
        out["loader_duplicate_fails"] = (
            _raises(lambda: load_task_bank(dup)) == "TaskBankSchemaError"
        )

    return out


def rubric_audit_bench() -> dict[str, Any]:
    r = rubric_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "rubric_audit",
        "schema": "rubric_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "rubric lane holds: sealed bank is deterministic and "
            "honesty-validated at build; writer failures all escalate; "
            "code beats an agreeing reviewer; grading is per-kind "
            "deterministic with a hard honesty fail-all; the loader is "
            "strict. Flags: writer can replace a criterion's required "
            "tokens and weight wholesale (the SYNTHETIC-label demand can "
            "be redefined by the agent being judged); token matching is "
            "substring; structural_json reads the first object only; "
            "reviewer is optional."
            if ok
            else f"RUBRIC AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
