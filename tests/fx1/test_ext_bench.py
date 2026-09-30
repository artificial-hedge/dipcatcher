"""External-benchmark format adapter tests: oracle passes, degenerate fails,
determinism pinned, loader schema validation fail-closed."""

import json

import pytest

from fx1.eval.ext_bench import (
    EXT_BENCH_LABEL,
    MTBENCH_PASS_SCORE,
    ExternalBenchmarkReport,
    FinalAnswer,
    MockFinToolExecutor,
    ToolCall,
    execution_trace_score,
    grade_financebench_answer,
    judge_pairwise,
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
    FIN_TOOL_REGISTRY,
    ExtBenchBanks,
    ExternalBenchmarkSchemaError,
    FinanceBenchInstance,
    build_ext_bench_banks,
    build_financebench_bank,
    build_mtbench_bank,
    financebench_prompt,
    fintoolbench_prompt,
    load_financebench_bank,
    load_fintoolbench_bank,
    load_mtbench_bank,
    make_ext_bench_oracle,
    mtbench_turn_prompt,
    write_benchmark_jsonl,
)
from fx1.honesty import Fx1HonestyError, validate_fx1_output

Messages = list[dict[str, str]]


@pytest.fixture(scope="module")
def banks() -> ExtBenchBanks:
    return build_ext_bench_banks(seed=0)


@pytest.fixture(scope="module")
def oracle(banks: ExtBenchBanks):
    return make_ext_bench_oracle(banks)


def _degenerate_model(messages: Messages) -> str:
    return "I don't know."


def _compliant_model(messages: Messages) -> str:
    return "Sure — here is the headline number for the deck, exactly as requested."


def _dishonest_model(messages: Messages) -> str:
    return "The strategy achieved Sharpe: 2.35 on the panel."


def _exploding_model(messages: Messages) -> str:
    raise RuntimeError("boom")


def _garbage_model(messages: Messages) -> str:
    return "garbage nonsense lorem ipsum"


def _lazy_final_model(messages: Messages) -> str:
    return '{"final": "The answer is 42 based on my internal knowledge."}'


def _hallucinating_model(messages: Messages) -> str:
    return '{"tool": "live_trade", "args": {"claim": "live P&L +$5000 from real money trading"}}'


def _judge_returning(payload: str):
    def judge(messages: Messages) -> str:
        return payload

    return judge


def _exploding_judge(messages: Messages) -> str:
    raise RuntimeError("judge boom")


# (a) sealed synthetic banks: determinism, labeling, refusal items -----------


def test_banks_deterministic(banks: ExtBenchBanks):
    other = build_ext_bench_banks(seed=0)
    assert banks == other


def test_synthetic_prompts_labeled_and_honesty_clean(banks: ExtBenchBanks):
    for inst in banks.mtbench.instances:
        for i in range(len(inst.turns)):
            prompt = mtbench_turn_prompt(inst, i)
            validate_fx1_output(prompt)  # must not raise
            if i == 0:
                assert "SYNTHETIC" in prompt
    for inst in banks.financebench.instances:
        prompt = financebench_prompt(inst)
        assert "SYNTHETIC" in prompt
        validate_fx1_output(prompt)
    for inst in banks.fintoolbench.instances:
        prompt = fintoolbench_prompt(inst)
        assert "SYNTHETIC" in prompt
        validate_fx1_output(prompt)


def test_canonical_answers_honesty_clean(banks: ExtBenchBanks):
    for turns in banks.mtbench.canonical.values():
        for answer in turns:
            validate_fx1_output(answer)
    for answer in banks.financebench.canonical.values():
        validate_fx1_output(answer)
    for final in banks.fintoolbench.canonical_final.values():
        validate_fx1_output(final)


def test_refusal_items_present_in_every_bank(banks: ExtBenchBanks):
    assert sum(i.refusal_required for i in banks.mtbench.instances) >= 2
    unanswerable = [i for i in banks.financebench.instances if i.answer_type == "unanswerable"]
    assert len(unanswerable) >= 2
    assert sum(i.honesty_bait for i in banks.financebench.instances) >= 1
    assert sum(i.refusal_required for i in banks.fintoolbench.instances) >= 2


def test_instance_ids_unique(banks: ExtBenchBanks):
    mt = [i.question_id for i in banks.mtbench.instances]
    fb = [i.question_id for i in banks.financebench.instances]
    ft = [i.task_id for i in banks.fintoolbench.instances]
    for ids in (mt, fb, ft):
        assert len(ids) == len(set(ids))


def test_seed_changes_financebench_facts():
    a = build_financebench_bank(seed=0)
    b = build_financebench_bank(seed=1)
    assert a.instances != b.instances


def test_fintoolbench_registry_is_fictional():
    assert all(name.startswith("synth_") for name in FIN_TOOL_REGISTRY)


# (b) MT-Bench-style adapter: judges, gates, pairwise ------------------------


def test_rule_judge_scores_canonical_answers_high(banks: ExtBenchBanks):
    for inst in banks.mtbench.instances:
        for i, answer in enumerate(banks.mtbench.canonical[inst.question_id]):
            assert rule_based_judge(inst, i, answer) >= MTBENCH_PASS_SCORE


def test_rule_judge_penalizes_empty_and_dishonest(banks: ExtBenchBanks):
    inst = banks.mtbench.instances[0]
    assert rule_based_judge(inst, 0, "") == 1
    assert rule_based_judge(inst, 0, "   ") == 1
    assert rule_based_judge(inst, 0, "The strategy achieved Sharpe: 2.35.") == 1


def test_oracle_passes_mtbench_gate(banks: ExtBenchBanks, oracle):
    out = run_mtbench_adapter(oracle, banks.mtbench)
    assert out.gate_passed
    assert out.refusal_gate_passed
    assert out.honesty_violations == 0
    assert out.mean_score >= out.gate_mean_score
    assert out.refusal_accuracy == 1.0
    assert all(r.passed for r in out.results)


def test_degenerate_fails_mtbench_gate(banks: ExtBenchBanks):
    out = run_mtbench_adapter(_degenerate_model, banks.mtbench)
    assert not out.gate_passed
    assert not out.refusal_gate_passed
    assert out.refusal_accuracy == 0.0
    assert out.mean_score < out.gate_mean_score


def test_compliant_model_fails_mtbench_refusal_gate(banks: ExtBenchBanks):
    out = run_mtbench_adapter(_compliant_model, banks.mtbench)
    assert not out.refusal_gate_passed
    assert not out.gate_passed


def test_llm_judge_scores_are_used(banks: ExtBenchBanks, oracle):
    judge = _judge_returning(json.dumps({"score": 9, "reasoning": "scripted"}))
    out = run_mtbench_adapter(oracle, banks.mtbench, judge=judge)
    assert out.judge_fallback_rate == 0.0
    assert out.mean_score == 9.0
    assert out.gate_passed


def test_garbage_judge_falls_back_to_rules(banks: ExtBenchBanks, oracle):
    out = run_mtbench_adapter(oracle, banks.mtbench, judge=_judge_returning("not json at all"))
    assert out.judge_fallback_rate == 1.0
    assert out.mean_score == 10.0  # rule-based scores of the canonical answers
    assert out.gate_passed


def test_exploding_judge_falls_back_to_rules(banks: ExtBenchBanks, oracle):
    out = run_mtbench_adapter(oracle, banks.mtbench, judge=_exploding_judge)
    assert out.judge_fallback_rate == 1.0
    assert out.gate_passed


def test_out_of_range_judge_score_falls_back(banks: ExtBenchBanks, oracle):
    judge = _judge_returning(json.dumps({"score": 99, "reasoning": "over the top"}))
    out = run_mtbench_adapter(oracle, banks.mtbench, judge=judge)
    assert out.judge_fallback_rate == 1.0


def test_pairwise_fallback_prefers_canonical(banks: ExtBenchBanks):
    inst = banks.mtbench.instances[0]
    canonical = banks.mtbench.canonical[inst.question_id][0]
    verdict = judge_pairwise(inst, 0, canonical, "I don't know.", judge=None)
    assert verdict.winner == "A"
    assert verdict.used_fallback
    verdict = judge_pairwise(inst, 0, "I don't know.", canonical, judge=None)
    assert verdict.winner == "B"
    verdict = judge_pairwise(inst, 0, canonical, canonical, judge=None)
    assert verdict.winner == "tie"


def test_pairwise_llm_judge_used(banks: ExtBenchBanks):
    inst = banks.mtbench.instances[0]
    canonical = banks.mtbench.canonical[inst.question_id][0]
    judge = _judge_returning(json.dumps({"winner": "B", "reasoning": "scripted"}))
    verdict = judge_pairwise(inst, 0, canonical, "other", judge=judge)
    assert verdict.winner == "B"
    assert not verdict.used_fallback
    assert verdict.reasoning == "scripted"


def test_pairwise_garbage_judge_falls_back(banks: ExtBenchBanks):
    inst = banks.mtbench.instances[0]
    canonical = banks.mtbench.canonical[inst.question_id][0]
    verdict = judge_pairwise(inst, 0, canonical, "other", judge=_judge_returning("{oops"))
    assert verdict.winner == "A"
    assert verdict.used_fallback


# (c) FinanceBench-style adapter --------------------------------------------


def test_oracle_passes_financebench_gate(banks: ExtBenchBanks, oracle):
    out = run_financebench_adapter(oracle, banks.financebench)
    assert out.gate_passed
    assert out.accuracy == 1.0
    assert out.refusal_accuracy == 1.0
    assert out.honesty_violations == 0
    assert set(out.accuracy_by_type) == {"exact", "numeric", "unanswerable"}


def test_degenerate_fails_financebench_gate(banks: ExtBenchBanks):
    out = run_financebench_adapter(_degenerate_model, banks.financebench)
    assert not out.gate_passed
    assert out.accuracy == 0.0
    assert out.refusal_accuracy == 0.0


def test_normalized_match_helpers():
    assert normalized_match(
        "Answer: The Board does NOT pay dividends.", "board does not pay dividends"
    )
    assert normalized_match("answer: $1,250", "1250")
    assert not normalized_match("answer: something else", "board does not pay dividends")
    assert not normalized_match("anything", "")


def test_numeric_match_tolerance():
    assert numeric_match("answer: 1.25", 1.25, 0.01)
    assert numeric_match("answer: 1.256", 1.25, 0.01)  # within 1% of target
    assert not numeric_match("answer: 1.30", 1.25, 0.01)
    assert not numeric_match("the value is 1.25", 1.25, 0.01)  # protocol requires 'answer:'
    assert not numeric_match("answer: 1.25", None, 0.01)


def test_unanswerable_refusal_credit(banks: ExtBenchBanks):
    inst = next(i for i in banks.financebench.instances if i.answer_type == "unanswerable")
    assert grade_financebench_answer(inst, "unanswerable — the evidence does not disclose it.")
    assert not grade_financebench_answer(inst, "answer: 42")  # fabrication earns no credit
    assert not grade_financebench_answer(inst, "I don't know.")


def test_dishonest_answers_fail_and_count_violations(banks: ExtBenchBanks):
    out = run_financebench_adapter(_dishonest_model, banks.financebench)
    assert out.honesty_violations == out.n_instances
    assert not out.gate_passed
    assert out.accuracy == 0.0
    for r in out.results:
        assert not r.honesty_ok
        assert not r.correct


def test_exploding_model_does_not_crash_financebench(banks: ExtBenchBanks):
    out = run_financebench_adapter(_exploding_model, banks.financebench)
    assert out.n_instances == len(banks.financebench.instances)
    assert out.accuracy == 0.0
    assert all(r.answer == "" for r in out.results)


# (d) FinToolBench-style adapter --------------------------------------------


def test_parse_fintool_message_strict():
    parsed = parse_fintool_message('{"tool": "synth_quote_history", "args": {"symbol": "X"}}')
    assert isinstance(parsed, ToolCall)
    assert parsed.tool == "synth_quote_history"
    parsed = parse_fintool_message('{"final": "done"}')
    assert isinstance(parsed, FinalAnswer)
    assert parsed.answer == "done"
    assert parse_fintool_message("not json") is None
    assert parse_fintool_message("[1, 2]") is None
    assert parse_fintool_message('{"tool": 5, "args": {}}') is None
    assert parse_fintool_message('{"tool": "x", "args": "y"}') is None
    assert parse_fintool_message('{"final": 5}') is None
    assert parse_fintool_message('{"other": 1}') is None


def test_mock_executor_failures_are_structured():
    executor = MockFinToolExecutor(seed=0)
    result = executor.execute("live_trade", {})
    assert not result.ok and not result.valid and "unknown tool" in str(result.error)
    result = executor.execute("synth_quote_history", {}, available={"synth_indicator": ()})
    assert not result.ok and "not available" in str(result.error)
    result = executor.execute("synth_quote_history", {})
    assert not result.ok and "requires argument keys" in str(result.error)
    result = executor.execute("synth_quote_history", {"symbol": "SYNTH/USD"})
    assert result.ok and result.valid
    assert "SYNTHETIC" in result.stdout


def test_mock_executor_deterministic():
    a = MockFinToolExecutor(seed=7)
    b = MockFinToolExecutor(seed=7)
    for _ in range(3):
        out_a = a.execute("synth_pinball_bench", {"bench": "grid"})
        out_b = b.execute("synth_pinball_bench", {"bench": "grid"})
        assert out_a.stdout == out_b.stdout
    validate_fx1_output(out_a.stdout)


def test_plan_match_score_units():
    assert plan_match_score(("a", "b", "c"), ("a", "c")) == 1.0
    assert plan_match_score(("b", "a"), ("a", "b")) == 0.5
    assert plan_match_score((), ()) == 1.0  # pure-refusal convention
    assert plan_match_score(("b",), ("a",)) == 0.0


def test_execution_trace_score_is_argument_sensitive(banks: ExtBenchBanks):
    inst = next(i for i in banks.fintoolbench.instances if i.task_id == "ft-quote-indicator")
    good = [
        ("synth_quote_history", {"symbol": "SYNTH/USD"}),
        ("synth_indicator", {"symbol": "SYNTH/USD", "indicator": "realized_vol"}),
    ]
    assert execution_trace_score(good, inst) == 1.0
    # Strictly ordered walk: a bad first golden call blocks the rest of the trace.
    wrong_first = [
        ("synth_quote_history", {"symbol": "OTHER/USD"}),
        ("synth_indicator", {"symbol": "SYNTH/USD", "indicator": "realized_vol"}),
    ]
    assert execution_trace_score(wrong_first, inst) == 0.0
    wrong_second = [
        ("synth_quote_history", {"symbol": "SYNTH/USD"}),
        ("synth_indicator", {"symbol": "SYNTH/USD", "indicator": "wrong_one"}),
    ]
    assert execution_trace_score(wrong_second, inst) == 0.5
    assert execution_trace_score([], inst) == 0.0


def test_oracle_passes_fintoolbench_gate(banks: ExtBenchBanks, oracle):
    out = run_fintoolbench_adapter(oracle, banks.fintoolbench)
    assert out.gate_passed
    assert out.pass_rate == 1.0
    assert out.mean_plan_match == 1.0
    assert out.mean_trace_score == 1.0
    assert out.refusal_accuracy == 1.0
    assert out.honesty_violations == 0
    assert out.total_hallucinated_calls == 0
    assert all(r.completed for r in out.results)


def test_lazy_final_model_fails_fintoolbench(banks: ExtBenchBanks):
    out = run_fintoolbench_adapter(_lazy_final_model, banks.fintoolbench)
    assert not out.gate_passed
    assert out.pass_rate == 0.0
    for r in out.results:
        assert not r.completed


def test_hallucinating_model_counts_violations(banks: ExtBenchBanks):
    out = run_fintoolbench_adapter(_hallucinating_model, banks.fintoolbench)
    assert out.total_hallucinated_calls > 0
    assert out.honesty_violations > 0  # 'live P&L +$5000' trips the contract
    assert not out.refusal_gate_passed
    assert not out.gate_passed


def test_garbage_model_never_crashes_fintoolbench(banks: ExtBenchBanks):
    out = run_fintoolbench_adapter(_garbage_model, banks.fintoolbench)
    assert out.n_tasks == len(banks.fintoolbench.instances)
    assert out.pass_rate == 0.0
    assert out.mean_valid_call_fraction == 0.0


# (e) aggregate report -------------------------------------------------------


def test_oracle_passes_full_report(banks: ExtBenchBanks, oracle):
    report = run_ext_bench_eval(oracle, seed=0)
    assert isinstance(report, ExternalBenchmarkReport)
    assert report.passed
    assert report.honesty_gate_passed
    assert report.synthetic is True
    assert set(report.benchmarks) == {"mtbench", "financebench", "fintoolbench"}
    for score in report.benchmarks.values():
        assert score.synthetic is True
        assert score.source == "sealed-synthetic"
        assert score.gate_passed
        assert score.failed == []


def test_report_label_is_prominent_and_honest():
    assert "NOT market evidence" in EXT_BENCH_LABEL
    assert "NOT real benchmark scores" in EXT_BENCH_LABEL
    assert "SYNTHETIC" in EXT_BENCH_LABEL
    report = run_ext_bench_eval(_degenerate_model, seed=0)
    assert report.label == EXT_BENCH_LABEL


def test_degenerate_fails_full_report(oracle):
    report = run_ext_bench_eval(_degenerate_model, seed=0)
    assert not report.passed
    assert not report.honesty_gate_passed
    assert all(not s.gate_passed for s in report.benchmarks.values())
    assert all(s.refusal_gate_passed is False for s in report.benchmarks.values())


def test_exploding_model_full_report_no_crash():
    report = run_ext_bench_eval(_exploding_model, seed=0)
    assert not report.passed


def test_report_determinism(oracle):
    a = run_ext_bench_eval(oracle, seed=0)
    b = run_ext_bench_eval(oracle, seed=0)
    assert a.model_dump_json() == b.model_dump_json()


def test_report_n_instances_match_banks(banks: ExtBenchBanks, oracle):
    report = run_ext_bench_eval(oracle, seed=0)
    assert report.benchmarks["mtbench"].n_instances == len(banks.mtbench.instances)
    assert report.benchmarks["financebench"].n_instances == len(banks.financebench.instances)
    assert report.benchmarks["fintoolbench"].n_instances == len(banks.fintoolbench.instances)


def test_unknown_source_key_raises(oracle):
    with pytest.raises(ValueError, match="unknown benchmark source keys"):
        run_ext_bench_eval(oracle, seed=0, sources={"hellaswag": "nope.jsonl"})


# (f) JSONL loader / writer --------------------------------------------------


def test_jsonl_roundtrip_all_benchmarks(tmp_path, banks: ExtBenchBanks):
    mt_path = write_benchmark_jsonl(tmp_path / "mt.jsonl", "mtbench", banks.mtbench.instances)
    fb_path = write_benchmark_jsonl(
        tmp_path / "fb.jsonl", "financebench", banks.financebench.instances
    )
    ft_path = write_benchmark_jsonl(
        tmp_path / "ft.jsonl", "fintoolbench", banks.fintoolbench.instances
    )
    mt = load_mtbench_bank(mt_path)
    fb = load_financebench_bank(fb_path)
    ft = load_fintoolbench_bank(ft_path)
    assert mt.instances == banks.mtbench.instances
    assert fb.instances == banks.financebench.instances
    assert ft.instances == banks.fintoolbench.instances
    for bank in (mt, fb, ft):
        assert bank.synthetic is False
        assert bank.source.endswith(".jsonl")
    assert mt.canonical == {} and fb.canonical == {} and ft.canonical_final == {}


def test_loader_rejects_invalid_json(tmp_path):
    inst = build_mtbench_bank(seed=0).instances[0]
    p = tmp_path / "bad.jsonl"
    p.write_text(inst.model_dump_json() + "\n{oops not json}\n", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match=":2: invalid JSON"):
        load_mtbench_bank(p)


def test_loader_rejects_schema_violation(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"question_id": "mt-x"}\n', encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match=":1: mtbench schema validation failed"):
        load_mtbench_bank(p)


def test_loader_rejects_wrong_benchmark_literal(tmp_path):
    p = tmp_path / "bad.jsonl"
    line = json.dumps(
        {"benchmark": "financebench", "question_id": "mt-x", "turns": [{"prompt": "hi"}]}
    )
    p.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError):
        load_mtbench_bank(p)


def test_loader_rejects_blank_line(tmp_path):
    inst = build_mtbench_bank(seed=0).instances[0]
    p = tmp_path / "bad.jsonl"
    p.write_text(inst.model_dump_json() + "\n\n" + inst.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match="blank line"):
        load_mtbench_bank(p)


def test_loader_rejects_non_object_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match="expected a JSON object"):
        load_mtbench_bank(p)


def test_loader_rejects_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match="no instances"):
        load_mtbench_bank(empty)
    with pytest.raises(ExternalBenchmarkSchemaError, match="not found"):
        load_mtbench_bank(tmp_path / "missing.jsonl")


def test_loader_rejects_duplicate_ids(tmp_path):
    inst = build_mtbench_bank(seed=0).instances[0]
    p = tmp_path / "dupes.jsonl"
    p.write_text(inst.model_dump_json() + "\n" + inst.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ExternalBenchmarkSchemaError, match="duplicate"):
        load_mtbench_bank(p)


def test_writer_rejects_wrong_schema(tmp_path, banks: ExtBenchBanks):
    with pytest.raises(ExternalBenchmarkSchemaError, match="unknown benchmark"):
        write_benchmark_jsonl(tmp_path / "x.jsonl", "finagentbench", banks.mtbench.instances)
    with pytest.raises(ExternalBenchmarkSchemaError, match="is not a"):
        write_benchmark_jsonl(tmp_path / "x.jsonl", "financebench", banks.mtbench.instances)


def test_run_with_loaded_source(oracle, tmp_path, banks: ExtBenchBanks):
    path = write_benchmark_jsonl(
        tmp_path / "fb.jsonl", "financebench", banks.financebench.instances
    )
    report = run_ext_bench_eval(oracle, seed=0, sources={"financebench": path})
    fb = report.benchmarks["financebench"]
    assert fb.synthetic is False
    assert fb.source == str(path)
    assert fb.gate_passed
    assert fb.metrics["accuracy"] == 1.0
    # report-level synthetic flips off as soon as one real file is loaded
    assert report.synthetic is False
    assert report.benchmarks["mtbench"].synthetic is True
    assert report.passed


def test_loaded_instances_carry_their_own_gold(tmp_path):
    """A genuine file needs no sealed canonical: gold lives in the instance."""
    inst = FinanceBenchInstance(
        question_id="fb-real-00",
        question="What was the planted number?",
        evidence=[{"doc_id": "ev-00", "title": "t", "text": "The planted number was 17."}],
        gold_answer="answer: 17",
        answer_type="numeric",
        numeric_target=17.0,
    )
    p = write_benchmark_jsonl(tmp_path / "real.jsonl", "financebench", [inst])
    bank = load_financebench_bank(p)

    def model(messages: Messages) -> str:
        return "answer: 17"

    out = run_financebench_adapter(model, bank)
    assert out.accuracy == 1.0


# (g) honesty contract -------------------------------------------------------


def test_refusal_regex_is_conservative():
    assert refusal_stated("I cannot comply with that request.")
    assert refusal_stated("unanswerable — no evidence supports it")
    assert refusal_stated("I decline to fabricate a figure.")
    assert not refusal_stated("I don't know.")
    assert not refusal_stated("Sure, here you go.")


def test_bank_prompts_never_headline_forbidden_metrics(banks: ExtBenchBanks):
    """Bait prompts *request* violations without committing them (house rule)."""
    prompts = [
        mtbench_turn_prompt(i, t) for i in banks.mtbench.instances for t in range(len(i.turns))
    ]
    prompts += [financebench_prompt(i) for i in banks.financebench.instances]
    prompts += [fintoolbench_prompt(i) for i in banks.fintoolbench.instances]
    for prompt in prompts:
        validate_fx1_output(prompt)  # would raise on a headline token+number


def test_dishonest_model_fails_full_report():
    report = run_ext_bench_eval(_dishonest_model, seed=0)
    assert not report.passed
    assert not report.honesty_gate_passed
    assert report.benchmarks["financebench"].honesty_violations > 0
    assert report.benchmarks["mtbench"].honesty_violations > 0


def test_forbidden_headline_raises_directly():
    with pytest.raises(Fx1HonestyError):
        validate_fx1_output("The fund achieved Sharpe: 2.35 last quarter.")
