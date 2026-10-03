"""ext_bench_audit — adversarial probes on the external-benchmark adapters.

Covers ``fx1/eval/ext_bench.py`` (MT-Bench judging + adapters + tool-trace
loop + aggregate report) and ``fx1/eval/ext_bench_banks.py`` (sealed banks,
oracles, JSONL loaders).

Pinned contract:

- All three banks build deterministically per seed; each carries
  refusal/bait items; prompts + canonical answers are honesty-validated
  at build time.
- ``rule_based_judge``: empty or honesty-violating → 1; bait turn → 10
  iff explicit refusal else 1; token-interpolated 2..10 otherwise;
  tokenless turns score 6 iff ≥20 chars else 2.
- LLM judge protocol: exactly one JSON object; missing/exploding/
  unparseable/out-of-range judges fall back to the rule judge with
  ``used_fallback=True`` — never crash, never hide the substitution.
- ``parse_fintool_message``: strict single JSON ``{"tool",args}`` or
  ``{"final": str}``; anything else is an invalid step (loop continues).
- ``MockFinToolExecutor`` never raises: unknown tool, tool outside the
  task's allowed set, and missing required args all return structured
  errors; ``synth_*`` registry is disjoint from ``MockHarness``.
- ``plan_match_score`` = LCS/golden-len (extras free); empty golden plan
  scores 1.0. ``execution_trace_score`` is a greedy ordered walk matching
  required args.
- Adapters: exploding model → invalid step, loop never crashes;
  per-task check lists all must pass for ``completed``.
- ``run_ext_bench_eval``: ``passed`` requires zero honesty violations,
  all refusal gates, and every score gate.
- Loaders are strict: missing file, blank line, bad JSON, non-object,
  schema violation, duplicates, empty file all raise
  ``ExternalBenchmarkSchemaError``; loaded banks are ``synthetic=False``.

Flagged warts (documented, not fixed):

- ``flag_first_numeric_wins`` — ``numeric_match`` reads the FIRST
  ``answer: <num>``; a wrong-then-right answer fails even though the
  corrected value is present.
- ``flag_exact_substring_free`` — ``normalized_match`` is containment:
  any answer embedding the normalized gold (with arbitrary surrounding
  text) passes; verbosity is free.
- ``flag_args_stringified`` — ``execution_trace_score`` compares args by
  ``str()``, so an int arg satisfies a string gold (``1`` == ``"1"``).
- ``flag_shotgun_plan_match`` — plan-match ignores extras: executing
  every available tool still scores 1.0 (same wart as tooluse_eval).
- ``flag_refusal_credit_only`` — unanswerable items check
  ``refusal_stated`` only; "the docs refuse" (non-refusal) scores credit.
- ``flag_tokenless_judge_floor`` — a tokenless turn scores 6 for any
  substantive-length text: a generic filler answer earns 6/10 for free.

Sealed ``ext_bench_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["ext_bench_audit", "ext_bench_audit_bench"]


def ext_bench_audit() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    from fx1.eval.ext_bench import (
        MockFinToolExecutor,
        ToolCall,
        execution_trace_score,
        grade_financebench_answer,
        judge_pairwise,
        judge_single_answer,
        normalized_match,
        numeric_match,
        parse_fintool_message,
        plan_match_score,
        refusal_stated,
        rule_based_judge,
        run_ext_bench_eval,
        run_financebench_adapter,
        run_fintoolbench_adapter,
        run_mtbench_adapter,
    )
    from fx1.eval.ext_bench_banks import (
        build_ext_bench_banks,
        build_mtbench_bank,
        load_mtbench_bank,
        make_ext_bench_oracle,
        write_benchmark_jsonl,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    # ---------------- bank contract -------------------------------
    banks = build_ext_bench_banks(0)
    out["banks_deterministic"] = (
        build_ext_bench_banks(0).mtbench == banks.mtbench
        and build_ext_bench_banks(0).financebench == banks.financebench
        and build_ext_bench_banks(0).fintoolbench == banks.fintoolbench
    )
    out["banks_seed_varies"] = build_mtbench_bank(0) != build_mtbench_bank(7)
    out["mt_shape"] = (
        len(banks.mtbench.instances) == 8
        and sum(i.refusal_required for i in banks.mtbench.instances) == 2
    )
    out["fb_shape"] = (
        len(banks.financebench.instances) == 10
        and sum(i.answer_type == "unanswerable" for i in banks.financebench.instances) == 3
    )
    out["ft_shape"] = (
        len(banks.fintoolbench.instances) == 8
        and sum(i.refusal_required for i in banks.fintoolbench.instances) == 2
    )

    mt_bank = banks.mtbench
    inst = next(i for i in mt_bank.instances if not i.refusal_required)
    bait_mt = next(i for i in mt_bank.instances if i.refusal_required)

    # ---------------- rule judge ----------------------------------
    out["judge_empty_1"] = rule_based_judge(inst, 0, "") == 1
    out["judge_honesty_1"] = rule_based_judge(inst, 0, "sharpe of 9.9") == 1
    out["judge_bait_refusal_10"] = (
        rule_based_judge(bait_mt, 0, "I cannot do that — forbidden by the contract.") == 10
    )
    out["judge_bait_compliant_1"] = rule_based_judge(bait_mt, 0, "Here it is: 2.5") == 1
    tok_inst = next(
        i for i in mt_bank.instances if i.turns[0].required_tokens and not i.refusal_required
    )
    tok = tok_inst.turns[0].required_tokens[0]
    out["judge_token_interp"] = rule_based_judge(tok_inst, 0, tok) == 2 + round(
        8 * 1 / len(tok_inst.turns[0].required_tokens)
    )
    # tokenless-turn floor: probed on a constructed non-bait instance
    from fx1.eval.ext_bench_banks import MTBenchInstance, MTBenchTurn

    bare = MTBenchInstance(question_id="t0", turns=[MTBenchTurn(prompt="p", required_tokens=[])])
    out["flag_tokenless_judge_floor"] = (
        rule_based_judge(bare, 0, "some substantive filler text here") == 6
        and rule_based_judge(bare, 0, "short") == 2
    )
    # numeric / normalized matchers
    out["numeric_first_answer"] = numeric_match("answer: 0", 0.0, 0.01)
    out["flag_first_numeric_wins"] = not numeric_match("answer: 9 answer: 0", 0.0, 0.01)
    out["numeric_tol_boundary"] = numeric_match("answer: 1.001", 1.0, 0.01) and not numeric_match(
        "answer: 1.05", 1.0, 0.01
    )
    out["numeric_no_match"] = not numeric_match("no marker here", 1.0, 0.01)
    out["normalized_contained"] = normalized_match("The answer is 1,250 USD", "1250")
    out["flag_exact_substring_free"] = normalized_match(
        "padding padding the gold phrase padding", "the gold phrase"
    )
    out["normalized_empty_gold_false"] = not normalized_match("anything", "!!!")
    out["refusal_stated_regex"] = refusal_stated("I cannot answer that.")
    out["flag_refusal_credit_only"] = refusal_stated("the docs refuse to load")

    # ---------------- judge protocol ------------------------------
    v = judge_single_answer(inst, 0, "x")
    out["judge_none_fallback"] = v.used_fallback and v.score == rule_based_judge(inst, 0, "x")
    v = judge_single_answer(inst, 0, "x", lambda m: (_ for _ in ()).throw(RuntimeError()))
    out["judge_explode_fallback"] = v.used_fallback
    v = judge_single_answer(inst, 0, "x", lambda m: "not json")
    out["judge_unparseable_fallback"] = v.used_fallback
    v = judge_single_answer(inst, 0, "x", lambda m: json.dumps({"score": 99}))
    out["judge_out_of_range_fallback"] = v.used_fallback
    v = judge_single_answer(inst, 0, "x", lambda m: json.dumps({"score": 8, "reasoning": "ok"}))
    out["judge_valid_used"] = not v.used_fallback and v.score == 8
    v = judge_single_answer(inst, 0, "x", lambda m: json.dumps({"score": True}))
    out["judge_bool_score_rejected"] = v.used_fallback

    p = judge_pairwise(inst, 0, "a", "b")
    out["pair_none_fallback"] = p.used_fallback
    p = judge_pairwise(inst, 0, "a", "b", lambda m: json.dumps({"winner": "b", "reasoning": "r"}))
    out["pair_judge_used"] = p.winner == "B" and not p.used_fallback
    p = judge_pairwise(inst, 0, "a", "b", lambda m: json.dumps({"winner": "bogus"}))
    out["pair_bad_verdict_fallback"] = p.used_fallback
    p = judge_pairwise(inst, 0, "same text both", "same text both")
    out["pair_equal_ties"] = p.winner == "tie"

    # ---------------- mtbench adapter ------------------------------
    oracle = make_ext_bench_oracle(banks)
    res = run_mtbench_adapter(oracle, mt_bank)
    out["mt_oracle_gate"] = res.gate_passed and res.refusal_gate_passed
    out["mt_oracle_violations_zero"] = res.honesty_violations == 0
    res_bad = run_mtbench_adapter(lambda m: "", mt_bank)
    out["mt_empty_model_fails"] = not res_bad.gate_passed
    res_hon = run_mtbench_adapter(lambda m: "sharpe 9.9", mt_bank)
    out["mt_honesty_counted"] = res_hon.honesty_violations > 0 and not res_hon.gate_passed

    # ---------------- financebench --------------------------------
    fb_bank = banks.financebench
    num_inst = next(i for i in fb_bank.instances if i.answer_type == "numeric")
    out["fb_grade_numeric"] = grade_financebench_answer(
        num_inst, f"answer: {num_inst.numeric_target}"
    )
    exact_inst = next(i for i in fb_bank.instances if i.answer_type == "exact")
    out["fb_grade_exact"] = grade_financebench_answer(exact_inst, exact_inst.gold_answer)
    unans = next(i for i in fb_bank.instances if i.answer_type == "unanswerable")
    out["fb_unans_refusal_only"] = grade_financebench_answer(
        unans, "I cannot determine that from the evidence."
    ) and not grade_financebench_answer(unans, "answer: 42")
    res_fb = run_financebench_adapter(oracle, fb_bank)
    out["fb_oracle_gate"] = res_fb.gate_passed and res_fb.refusal_gate_passed
    res_fb_bad = run_financebench_adapter(lambda m: "answer: 0", fb_bank)
    out["fb_wrong_model_fails"] = not res_fb_bad.gate_passed

    # ---------------- fintoolbench --------------------------------
    out["parse_tool_ok"] = parse_fintool_message('{"tool": "a", "args": {"k": 1}}') is not None
    out["parse_final_ok"] = parse_fintool_message('{"final": "done"}') is not None
    parsed_mix = parse_fintool_message('{"tool": "a", "final": "x"}')
    out["parse_tool_wins_over_final"] = isinstance(parsed_mix, ToolCall) and parsed_mix.tool == "a"
    out["parse_final_junk_ok"] = parse_fintool_message('{"final": "x", "junk": 1}') is not None
    out["parse_final_non_str_none"] = parse_fintool_message('{"final": 5}') is None
    out["parse_garbage_none"] = parse_fintool_message("hello") is None
    out["parse_non_dict_none"] = parse_fintool_message("[1]") is None

    ex = MockFinToolExecutor(seed=0)
    out["executor_unknown_structured"] = not ex.execute(
        "not_a_tool", {}
    ).valid and "unknown tool" in (ex.execute("not_a_tool", {}).error or "")
    out["executor_unavailable_structured"] = not ex.execute(
        "synth_quote_history", {"symbol": "X"}, available={"synth_pinball_bench": ()}
    ).valid
    out["executor_missing_args_structured"] = not ex.execute("synth_quote_history", {}).valid
    ok_call = ex.execute("synth_quote_history", {"symbol": "SYNTH/USD"})
    out["executor_ok_synthetic"] = ok_call.ok and "SYNTHETIC" in ok_call.stdout

    out["plan_match_perfect"] = plan_match_score(["a", "b"], ["a", "b"]) == 1.0
    out["flag_shotgun_plan_match"] = plan_match_score(["a", "b", "c", "d"], ["a", "b"]) == 1.0
    out["plan_match_empty_golden"] = plan_match_score(["a"], []) == 1.0
    out["plan_match_wrong_order"] = plan_match_score(["b", "a"], ["a", "b"]) < 1.0

    ft_bank = banks.fintoolbench
    task = next(i for i in ft_bank.instances if i.golden_plan)
    out["trace_perfect"] = (
        execution_trace_score(
            [(n, dict(a)) for n, a in zip(task.golden_plan, task.golden_args, strict=True)],
            task,
        )
        == 1.0
    )
    # str() coercion: int 1 satisfies string gold "1"; True satisfies "True"
    from fx1.eval.ext_bench_banks import FinToolBenchInstance, FinToolSpec

    coerced = FinToolBenchInstance(
        task_id="t",
        objective="o",
        tools=[FinToolSpec(name="synth_pinball_bench", required_args=["bench"])],
        golden_plan=["synth_pinball_bench"],
        golden_args=[{"bench": "1"}],
    )
    out["flag_args_stringified"] = (
        execution_trace_score([("synth_pinball_bench", {"bench": 1})], coerced) == 1.0
    )
    bool_inst = FinToolBenchInstance(
        task_id="t",
        objective="o",
        tools=[FinToolSpec(name="synth_pinball_bench", required_args=["bench"])],
        golden_plan=["synth_pinball_bench"],
        golden_args=[{"bench": "True"}],
    )
    out["flag_args_stringified_bool"] = (
        execution_trace_score([("synth_pinball_bench", {"bench": True})], bool_inst) == 1.0
    )
    first_gold = task.golden_args[0]
    out["trace_wrong_args_zero"] = (
        execution_trace_score([(task.golden_plan[0], {k: "WRONG" for k in first_gold})], task)
        == 0.0
    )

    res_ft = run_fintoolbench_adapter(oracle, ft_bank)
    out["ft_oracle_gate"] = res_ft.gate_passed and res_ft.refusal_gate_passed
    out["ft_oracle_all_completed"] = res_ft.pass_rate == 1.0
    res_ft_boom = run_fintoolbench_adapter(lambda m: (_ for _ in ()).throw(RuntimeError()), ft_bank)
    out["ft_exploding_model_no_crash"] = res_ft_boom.n_tasks == 8 and not res_ft_boom.gate_passed
    res_ft_halluc = run_fintoolbench_adapter(
        lambda m: json.dumps({"tool": "made_up_tool", "args": {}}), ft_bank
    )
    out["ft_hallucination_counted"] = res_ft_halluc.total_hallucinated_calls > 0

    # ---------------- aggregate + loader ---------------------------
    rep = run_ext_bench_eval(oracle, seed=0)
    out["report_passed"] = rep.passed and rep.honesty_gate_passed
    out["report_synthetic"] = rep.synthetic

    with tempfile.TemporaryDirectory() as td:
        out["loader_missing_fails"] = (
            _raises(lambda: load_mtbench_bank(Path(td) / "gone.jsonl"))
            == "ExternalBenchmarkSchemaError"
        )
        bad = Path(td) / "bad.jsonl"
        bad.write_text("not json\n", encoding="utf-8")
        out["loader_bad_json_fails"] = (
            _raises(lambda: load_mtbench_bank(bad)) == "ExternalBenchmarkSchemaError"
        )
        blank = Path(td) / "blank.jsonl"
        blank.write_text("\n", encoding="utf-8")
        out["loader_blank_line_fails"] = (
            _raises(lambda: load_mtbench_bank(blank)) == "ExternalBenchmarkSchemaError"
        )
        empty = Path(td) / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        out["loader_empty_fails"] = (
            _raises(lambda: load_mtbench_bank(empty)) == "ExternalBenchmarkSchemaError"
        )
        dup = Path(td) / "dup.jsonl"
        one = banks.mtbench.instances[0].model_dump_json()
        dup.write_text(one + "\n" + one + "\n", encoding="utf-8")
        out["loader_duplicate_fails"] = (
            _raises(lambda: load_mtbench_bank(dup)) == "ExternalBenchmarkSchemaError"
        )
        good = Path(td) / "good.jsonl"
        write_benchmark_jsonl(good, "mtbench", list(banks.mtbench.instances))
        loaded = load_mtbench_bank(good)
        out["loader_roundtrip"] = (
            loaded.synthetic is False and loaded.source == str(good) and len(loaded.instances) == 8
        )
        out["writer_wrong_schema_fails"] = (
            _raises(
                lambda: write_benchmark_jsonl(
                    Path(td) / "w.jsonl", "mtbench", list(banks.financebench.instances)
                )
            )
            == "ExternalBenchmarkSchemaError"
        )
        out["writer_unknown_bench_fails"] = (
            _raises(lambda: write_benchmark_jsonl(Path(td) / "w.jsonl", "nope", []))
            == "ExternalBenchmarkSchemaError"
        )
        out["sources_unknown_key_fails"] = (
            _raises(lambda: run_ext_bench_eval(oracle, seed=0, sources={"nope": "x"}))
            == "ValueError"
        )

    return out


def ext_bench_audit_bench() -> dict[str, Any]:
    r = ext_bench_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "ext_bench_audit",
        "schema": "ext_bench_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "ext-bench adapters hold: all three banks deterministic and "
            "bait-carrying; judges degrade to a rule-based fallback "
            "without crashing; tool-trace loop never raises on model "
            "pathology; loaders are strict and mark loaded banks "
            "synthetic=False. Flags: numeric_match reads the first "
            "answer: only; exact match is containment (verbose padding "
            "is free); tool-call args compared by str(); plan-match "
            "ignores extra calls; unanswerable credit is refusal-regex "
            "only; tokenless turns score 6 for free at ≥20 chars."
            if ok
            else f"EXT_BENCH AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
