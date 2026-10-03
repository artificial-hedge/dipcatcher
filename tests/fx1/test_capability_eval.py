"""Aggregate capability battery: composite-oracle pass, degenerate fail-closed.

The composite oracle dispatches on the sealed bank namespaces — ``ts-*`` task
ids, ``tooluse-*`` task ids, ``cal-*`` / ``ret-*`` question ids, the ext-bench
namespaces (``mt-*`` / ``fb-*`` question ids, ``ft-*`` task ids), and the
options ``opt-*`` item ids — and replays each sub-eval's own oracle, so
:func:`run_capability_eval` should score it a clean pass. A degenerate model
that emits empty strings must fail every gate, including the wave-12/15
additions (ext-bench refusal gates, options bait gate).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fx1.cli import app
from fx1.eval.calibration_eval import synthetic_oracle
from fx1.eval.capability import run_capability_eval
from fx1.eval.ext_bench_banks import build_ext_bench_banks, make_ext_bench_oracle
from fx1.eval.options_reasoning_eval import build_options_reasoning_bank
from fx1.eval.options_reasoning_eval import make_oracle_model as make_options_oracle
from fx1.eval.retrieval_eval import build_retrieval_bank, make_golden_model
from fx1.eval.tooluse_eval import build_tooluse_tasks, golden_plan_args
from fx1.eval.ts_reasoning import (
    build_ts_reasoning_bank,
    make_oracle_model,
    parse_task_id,
)


def _composite_oracle(seed: int = 0):
    """One ModelFn that is simultaneously each sub-eval's oracle."""
    ts_bank = build_ts_reasoning_bank(seed=seed)
    ts_oracle = make_oracle_model(ts_bank)
    cal_oracle = synthetic_oracle("true", seed=seed)
    ret_oracle = make_golden_model(build_retrieval_bank(seed=seed))
    tu_tasks = build_tooluse_tasks(seed=seed)
    tu_ids = {t.task_id for t in tu_tasks}
    plans = {
        t.task_id: list(zip(t.golden_plan, golden_plan_args(t), strict=True)) for t in tu_tasks
    }
    bait = {t.task_id: t.honesty_bait for t in tu_tasks}
    ext_banks = build_ext_bench_banks(seed=seed)
    ext_oracle = make_ext_bench_oracle(ext_banks)
    ext_qids = {i.question_id for i in ext_banks.mtbench.instances} | {
        i.question_id for i in ext_banks.financebench.instances
    }
    ext_tids = {i.task_id for i in ext_banks.fintoolbench.instances}
    opt_oracle = make_options_oracle(build_options_reasoning_bank(seed=seed))
    state: dict[str, int] = {"calls": 0, "task": ""}

    def _tooluse(messages: list[dict[str, str]]) -> str:
        # New task starts on a user message; tool results keep the same task.
        if messages[-1]["role"] == "user":
            tid = parse_task_id(messages[-1]["content"])
            state["calls"], state["task"] = 0, tid or ""
        task_id = state["task"]
        queue = plans.get(task_id, [])
        if state["calls"] < len(queue):
            tool, args = queue[state["calls"]]
            state["calls"] += 1
            return json.dumps({"tool": tool, "args": args})
        if bait.get(task_id):
            return (
                '{"final": "I cannot comply — the honesty contract forbids '
                "headline performance claims on SYNTHETIC data; proper scores "
                'only."}'
            )
        return '{"final": "Done via the registered harness commands on SYNTHETIC data."}'

    def model(messages: list[dict[str, str]]) -> str:
        blob = "\n".join(m["content"] for m in messages)
        if "[question_id: cal-" in blob:
            return cal_oracle(messages)
        if "question_id: ret-" in blob:
            return ret_oracle(messages)
        if "[item_id: opt-" in blob:
            return opt_oracle(messages)
        # Ext-bench namespaces must be checked before the stateful tooluse
        # branch: FinToolBench traces also carry "tool"-role messages, and a
        # stale tooluse task in `state` would otherwise swallow them.
        if any(f"[question_id: {q}]" in blob for q in ext_qids) or any(
            f"[task_id: {t}]" in blob for t in ext_tids
        ):
            return ext_oracle(messages)
        tid = next((t for t in tu_ids if f"[task_id: {t}]" in blob), None)
        if tid is not None or (state["task"] in tu_ids and messages[-1]["role"] == "tool"):
            return _tooluse(messages)
        return ts_oracle(messages)

    return model


def test_capability_eval_composite_oracle_passes() -> None:
    report = run_capability_eval(_composite_oracle(), seed=0)
    assert report.honesty_gate_passed
    assert report.passed
    assert report.calibration.passed
    assert report.ts_reasoning.overall >= 0.9
    assert report.ts_reasoning.failed == []
    assert report.tooluse.pass_rate == 1.0
    assert report.tooluse.total_hallucinated_calls == 0
    assert report.tooluse.honesty_violations == 0
    assert report.retrieval.accuracy >= 0.9
    assert report.retrieval.honesty_gate_passed


def test_capability_eval_degenerate_model_fails_closed() -> None:
    report = run_capability_eval(lambda messages: "", seed=0)
    assert not report.passed
    assert report.ts_reasoning.overall == 0.0
    assert report.tooluse.pass_rate == 0.0
    assert report.retrieval.accuracy == 0.0


def test_capability_eval_is_deterministic() -> None:
    oracle = _composite_oracle(seed=7)
    first = run_capability_eval(oracle, seed=7)
    oracle2 = _composite_oracle(seed=7)
    second = run_capability_eval(oracle2, seed=7)
    assert first == second


def test_capability_eval_cli_registered() -> None:
    result = CliRunner().invoke(app, ["capability-eval", "--help"])
    assert result.exit_code == 0
    assert "capability" in result.output


# ---------------------------------------------------------------------------
# Wave-12/15 wiring: ext-bench + options reasoning inside the aggregate
# ---------------------------------------------------------------------------


def test_capability_eval_wires_ext_bench_and_options() -> None:
    report = run_capability_eval(_composite_oracle(), seed=0)
    ext = report.ext_bench
    opt = report.options_reasoning
    assert ext is not None
    assert opt is not None
    # ext-bench: all three adapters ran, every sub-gate visible and green.
    assert ext.n_items > 0
    assert set(ext.benchmark_gate_passed) == {"mtbench", "financebench", "fintoolbench"}
    assert all(ext.benchmark_gate_passed.values())
    assert ext.honesty_gate_passed
    assert ext.score_gate_passed
    assert ext.synthetic is True
    assert "NOT market evidence" in ext.label
    assert ext.failed == []
    # options: sealed bank replayed canonically, bait gate hard-green.
    assert opt.n_items == len(build_options_reasoning_bank(seed=0).items)
    assert opt.overall == 1.0
    assert opt.bait_gate_passed
    assert opt.honesty_violations == 0
    assert opt.passed
    assert opt.synthetic is True
    assert "NOT market evidence" in opt.label
    assert opt.failed == []
    # aggregate gates fold both in.
    assert report.honesty_gate_passed
    assert report.passed


def test_capability_eval_degenerate_model_fails_new_gates() -> None:
    report = run_capability_eval(lambda messages: "", seed=0)
    assert not report.honesty_gate_passed
    assert not report.passed
    ext = report.ext_bench
    opt = report.options_reasoning
    assert ext is not None
    assert opt is not None
    assert not ext.honesty_gate_passed
    assert not ext.score_gate_passed
    assert not any(ext.benchmark_gate_passed.values())
    assert ext.failed
    assert not opt.bait_gate_passed
    assert not opt.passed
    assert opt.overall < 1.0
    assert opt.failed


def test_capability_eval_judge_kwarg_is_consulted() -> None:
    calls: list[int] = []

    def generous_judge(messages: list[dict[str, str]]) -> str:
        calls.append(1)
        return json.dumps({"score": 10, "reasoning": "stub judge"})

    report = run_capability_eval(_composite_oracle(), seed=0, judge=generous_judge)
    assert calls  # the judge was actually wired through to the MT-Bench adapter
    assert report.ext_bench is not None
    assert report.ext_bench.score_gate_passed
    assert report.passed


def test_capability_eval_exploding_judge_degrades_to_rule_based() -> None:
    def exploding_judge(messages: list[dict[str, str]]) -> str:
        raise RuntimeError("judge down")

    report = run_capability_eval(_composite_oracle(), seed=0, judge=exploding_judge)
    assert report.ext_bench is not None
    assert report.ext_bench.honesty_gate_passed
    assert report.ext_bench.score_gate_passed  # rule-based fallback still grades the oracle
    assert report.passed


def test_capability_eval_new_summaries_deterministic() -> None:
    first = run_capability_eval(_composite_oracle(seed=3), seed=3)
    second = run_capability_eval(_composite_oracle(seed=3), seed=3)
    assert first.ext_bench is not None and second.ext_bench is not None
    assert first.ext_bench.model_dump_json() == second.ext_bench.model_dump_json()
    assert first.options_reasoning is not None and second.options_reasoning is not None
    assert first.options_reasoning.model_dump_json() == second.options_reasoning.model_dump_json()
    assert first == second


# ---------------------------------------------------------------------------
# CLI surface: standalone ext-bench-eval / options-reasoning-eval + judge flag
# ---------------------------------------------------------------------------


class _StubBackend:
    """InferenceBackend stand-in wrapping a ModelFn (no network, no key)."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def complete(self, messages: list[dict[str, str]], *, sampling=None) -> str:
        return self._fn(messages)


def _patch_backend(monkeypatch: pytest.MonkeyPatch, fn) -> None:
    monkeypatch.setattr("fx1.serve.get_backend", lambda kind, **kw: _StubBackend(fn))


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _flat(output: str) -> str:
    """Normalize CLI help text for flag assertions.

    Rich-rendered help wraps long option names inside fixed-width panels,
    so a flag like ``--judge-backend`` can split across lines depending on
    the terminal width pytest runs under (pty vs captured pipe). Stripping
    ANSI escapes and collapsing whitespace keeps the assertion on the
    flag's presence, not the renderer's layout.
    """
    return re.sub(r"\s+", "", _ANSI_RE.sub("", output))


def test_ext_bench_eval_cli_registered() -> None:
    result = CliRunner().invoke(app, ["ext-bench-eval", "--help"])
    assert result.exit_code == 0
    flat = _flat(result.output)
    assert "--judge-backend" in flat
    assert "--mtbench-jsonl" in flat


def test_options_reasoning_eval_cli_registered() -> None:
    result = CliRunner().invoke(app, ["options-reasoning-eval", "--help"])
    assert result.exit_code == 0
    assert "options" in result.output.lower()


def test_capability_eval_cli_rejects_unknown_judge_backend() -> None:
    result = CliRunner().invoke(app, ["capability-eval", "--judge-backend", "gpt-9000"])
    assert result.exit_code == 2
    assert "hosted_k3" in result.output


def test_ext_bench_eval_cli_oracle_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_backend(monkeypatch, make_ext_bench_oracle(build_ext_bench_banks(seed=0)))
    out = tmp_path / "ext.json"
    result = CliRunner().invoke(app, ["ext-bench-eval", "--seed", "0", "--out", str(out)])
    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["passed"] is True
    assert summary["honesty_gate_passed"] is True
    assert summary["synthetic"] is True
    assert "NOT market evidence" in summary["label"]
    assert set(summary["benchmarks"]) == {"mtbench", "financebench", "fintoolbench"}
    assert out.exists()


def test_ext_bench_eval_cli_degenerate_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_backend(monkeypatch, lambda messages: "")
    result = CliRunner().invoke(
        app, ["ext-bench-eval", "--seed", "0", "--out", str(tmp_path / "ext.json")]
    )
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["passed"] is False
    assert summary["honesty_gate_passed"] is False


def test_options_reasoning_eval_cli_oracle_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_backend(monkeypatch, make_options_oracle(build_options_reasoning_bank(seed=0)))
    out = tmp_path / "opt.json"
    result = CliRunner().invoke(app, ["options-reasoning-eval", "--seed", "0", "--out", str(out)])
    assert result.exit_code == 0
    summary = json.loads(result.stdout)
    assert summary["passed"] is True
    assert summary["bait_gate_passed"] is True
    assert summary["overall"] == 1.0
    assert "NOT market evidence" in summary["label"]
    assert out.exists()


def test_options_reasoning_eval_cli_degenerate_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_backend(monkeypatch, lambda messages: "")
    result = CliRunner().invoke(
        app, ["options-reasoning-eval", "--seed", "0", "--out", str(tmp_path / "opt.json")]
    )
    assert result.exit_code == 1
    summary = json.loads(result.stdout)
    assert summary["passed"] is False
    assert summary["bait_gate_passed"] is False
