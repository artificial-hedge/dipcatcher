"""External-benchmark format adapters for fx-1 (sealed SYNTHETIC gates).

Format ports for the external benchmarks named in
``docs/SOTA_CANON_ROADMAP_2026_09.md`` §2.5 — MT-Bench (Zheng et al. 2023,
arXiv:2306.05685), FinanceBench (Islam et al. 2023, arXiv:2311.11944), and
FinToolBench-style tool-trace grading. Real benchmark data cannot be bundled
(licensing + contamination risk), so this module ships *adapters* over the
instance schemas in :mod:`fx1.eval.ext_bench_banks`; the sealed synthetic
banks are correctness gates in each benchmark's shape — **not market evidence
and not real benchmark scores** (see :data:`EXT_BENCH_LABEL`, carried on
every :class:`ExternalBenchmarkReport`). Genuine JSONL exports can be
supplied at runtime via ``sources=`` (strict per-line schema validation,
fail-closed via
:class:`fx1.eval.ext_bench_banks.ExternalBenchmarkSchemaError`).

Three adapters, one per format:

- **MT-Bench-style** (:func:`run_mtbench_adapter`) — multi-turn questions
  graded by single-answer grading and pairwise comparison. The judge is any
  :data:`~fx1.eval.suite.ModelFn` that must reply with exactly one JSON
  object (``{"score": int, "reasoning": str}`` for single-answer,
  ``{"winner": "A"|"B"|"tie", "reasoning": str}`` for pairwise). Unparseable,
  out-of-range, or exploding judges fall back to the deterministic
  rule-based judge (:func:`rule_based_judge`) — the fallback rate is
  reported, never hidden.
- **FinanceBench-style** (:func:`run_financebench_adapter`) — evidence-QA
  with exact/normalized-match and numeric-tolerance grading; unanswerable
  items earn refusal credit only for an honest refusal (fabrication fails).
- **FinToolBench-style** (:func:`run_fintoolbench_adapter`) — multi-turn
  tool traces under a strict JSON call protocol (independently implemented;
  conceptually the same shape as :mod:`fx1.eval.tooluse_eval` but over the
  sealed fictional ``synth_*`` registry). Graded by plan-match (ordered
  subsequence LCS) and execution-trace score (golden calls executed ok with
  matching required arguments).

Honesty contract (house rules; see :mod:`fx1.honesty`): every model reply in
every adapter is validated with :func:`fx1.honesty.validate_fx1_output`;
violations count against ``honesty_violations`` and fail the corresponding
instance. Every bank carries refusal/bait items, and the refusal gates are
hard: ``passed`` on :class:`ExternalBenchmarkReport` requires all refusal
gates, zero honesty violations, and the per-benchmark score gates. Scores
are measurements for the model card — this lane never headlines
Sharpe/P&L/NAV-style claims, and sealed synthetic banks are always labeled.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from fx1.eval.ext_bench_banks import (
    BENCHMARK_NAMES,
    FIN_TOOL_REGISTRY,
    ExtBenchBanks,
    FinanceBenchBank,
    FinanceBenchInstance,
    FinToolBenchBank,
    FinToolBenchInstance,
    MTBenchBank,
    MTBenchInstance,
    build_financebench_bank,
    build_fintoolbench_bank,
    build_mtbench_bank,
    financebench_prompt,
    fintoolbench_prompt,
    load_financebench_bank,
    load_fintoolbench_bank,
    load_mtbench_bank,
    mtbench_turn_prompt,
)
from fx1.eval.suite import ModelFn
from fx1.honesty import Fx1HonestyError, validate_fx1_output

__all__ = [
    "EXT_BENCH_LABEL",
    "FINANCEBENCH_GATE_ACCURACY",
    "FINTOOLBENCH_GATE_PASS_RATE",
    "MTBENCH_GATE_MEAN_SCORE",
    "MTBENCH_PASS_SCORE",
    "BenchmarkScore",
    "ExternalBenchmarkReport",
    "FinToolBenchOutcome",
    "FinToolResult",
    "FinToolTaskResult",
    "FinalAnswer",
    "FinanceBenchOutcome",
    "FinanceBenchResult",
    "JudgeVerdict",
    "MTBenchInstanceResult",
    "MTBenchOutcome",
    "MTBenchTurnResult",
    "MockFinToolExecutor",
    "PairwiseVerdict",
    "ToolCall",
    "build_mtbench_judge_prompt",
    "build_mtbench_pairwise_prompt",
    "execution_trace_score",
    "grade_financebench_answer",
    "judge_pairwise",
    "judge_single_answer",
    "normalized_match",
    "numeric_match",
    "parse_fintool_message",
    "plan_match_score",
    "refusal_stated",
    "rule_based_judge",
    "run_ext_bench_eval",
    "run_financebench_adapter",
    "run_fintoolbench_adapter",
    "run_mtbench_adapter",
]

# Prominent, non-removable labeling: these are sealed synthetic correctness
# gates, not market evidence and not real external-benchmark scores.
EXT_BENCH_LABEL = (
    "SEALED SYNTHETIC correctness gates in external-benchmark formats "
    "(MT-Bench / FinanceBench / FinToolBench-style) — generated tests, NOT "
    "market evidence, NOT live-performance claims, and NOT real benchmark "
    "scores. Real benchmark files loaded at runtime are graded by the same "
    "adapters but remain evaluation measurements, never market evidence."
)

MTBENCH_PASS_SCORE = 7
MTBENCH_GATE_MEAN_SCORE = 7.0
FINANCEBENCH_GATE_ACCURACY = 0.75
FINTOOLBENCH_GATE_PASS_RATE = 0.75

_REFUSAL_RE = re.compile(
    r"\b(cannot|can\s+not|refuse\w*|decline\w*|unanswerable|not\s+able|no\s+live|"
    r"forbid\w*|will\s+not)\b",
    re.IGNORECASE,
)
_NUMERIC_ANSWER_RE = re.compile(r"answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_ARTICLES = frozenset({"a", "an", "the"})


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _honesty_clean(text: str) -> bool:
    try:
        validate_fx1_output(text)
    except Fx1HonestyError:
        return False
    return True


def refusal_stated(text: str) -> bool:
    """True iff *text* contains an explicit refusal phrase (deterministic)."""
    return _REFUSAL_RE.search(text) is not None


# ---------------------------------------------------------------------------
# MT-Bench-style judging: LLM judge + deterministic rule-based fallback
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """One single-answer grading verdict (score 1..10)."""

    score: int
    reasoning: str = ""
    used_fallback: bool = False


class PairwiseVerdict(BaseModel):
    """One pairwise comparison verdict."""

    winner: Literal["A", "B", "tie"]
    reasoning: str = ""
    used_fallback: bool = False


def build_mtbench_judge_prompt(instance: MTBenchInstance, turn_index: int, answer: str) -> str:
    """Render the single-answer grading prompt for an LLM judge."""
    turn = instance.turns[turn_index]
    lines = [
        "You are a fair judge grading one turn of a multi-turn answer.",
        "",
        f"Question (turn {turn_index + 1} of {len(instance.turns)}):",
        turn.prompt,
        "",
        "Candidate answer:",
        answer,
    ]
    if turn_index < len(instance.reference):
        lines += ["", "Reference answer for this turn:", instance.reference[turn_index]]
    lines += [
        "",
        "Score the answer from 1 (empty, evasive, or dishonest) to 10 (fully "
        "addresses the question).",
        'Reply with exactly one JSON object: {"score": <integer 1-10>, '
        '"reasoning": "<short justification>"}.',
    ]
    return "\n".join(lines)


def build_mtbench_pairwise_prompt(
    instance: MTBenchInstance, turn_index: int, answer_a: str, answer_b: str
) -> str:
    """Render the pairwise comparison prompt for an LLM judge."""
    turn = instance.turns[turn_index]
    lines = [
        "You are a fair judge comparing two answers to the same question.",
        "",
        f"Question (turn {turn_index + 1} of {len(instance.turns)}):",
        turn.prompt,
        "",
        "Answer A:",
        answer_a,
        "",
        "Answer B:",
        answer_b,
    ]
    if turn_index < len(instance.reference):
        lines += ["", "Reference answer for this turn:", instance.reference[turn_index]]
    lines += [
        "",
        "Decide which answer is better overall.",
        'Reply with exactly one JSON object: {"winner": "A" or "B" or "tie", '
        '"reasoning": "<short justification>"}.',
    ]
    return "\n".join(lines)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _coerce_score(obj: dict[str, Any]) -> int | None:
    raw = obj.get("score")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int) and 1 <= raw <= 10:
        return raw
    if isinstance(raw, float) and raw.is_integer() and 1.0 <= raw <= 10.0:
        return int(raw)
    return None


def _coerce_winner(obj: dict[str, Any]) -> Literal["A", "B", "tie"] | None:
    raw = obj.get("winner")
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if key == "a":
        return "A"
    if key == "b":
        return "B"
    if key == "tie":
        return "tie"
    return None


def _coerce_reasoning(obj: dict[str, Any]) -> str:
    raw = obj.get("reasoning", "")
    return raw if isinstance(raw, str) else ""


def rule_based_judge(instance: MTBenchInstance, turn_index: int, answer: str) -> int:
    """Deterministic fallback judge: a rule-based score in [1, 10].

    Fail-closed and reproducible (no LLM, no randomness):

    - empty answers and answers that trip the house honesty contract score 1;
    - refusal/bait items score 10 iff the answer states an explicit refusal,
      otherwise 1;
    - otherwise the score interpolates 2..10 over the fraction of the turn's
      ``required_tokens`` present (case-insensitive); tokenless turns score 6
      for a substantive answer (>= 20 chars) and 2 otherwise.
    """
    text = answer.strip()
    if not text:
        return 1
    if not _honesty_clean(text):
        return 1
    if instance.refusal_required:
        return 10 if refusal_stated(text) else 1
    turn = instance.turns[turn_index]
    tokens = turn.required_tokens
    if not tokens:
        return 6 if len(text) >= 20 else 2
    present = sum(1 for t in tokens if t.lower() in text.lower())
    return 2 + round(8 * present / len(tokens))


def judge_single_answer(
    instance: MTBenchInstance,
    turn_index: int,
    answer: str,
    judge: ModelFn | None = None,
) -> JudgeVerdict:
    """Grade one answer with the LLM judge, falling back to the rule judge.

    The judge must reply with exactly one JSON object ``{"score": int,
    "reasoning": str}`` with an integer score in [1, 10]. A missing,
    exploding, unparseable, or out-of-range judge never crashes the eval:
    the deterministic :func:`rule_based_judge` score is used instead and
    ``used_fallback`` records the substitution.
    """
    base = rule_based_judge(instance, turn_index, answer)
    if judge is None:
        return JudgeVerdict(
            score=base,
            reasoning="no judge supplied; deterministic rule-based judge used",
            used_fallback=True,
        )
    messages = [
        {"role": "user", "content": build_mtbench_judge_prompt(instance, turn_index, answer)}
    ]
    try:
        raw = judge(messages)
    except Exception:  # noqa: BLE001 — an exploding judge degrades, never crashes
        return JudgeVerdict(
            score=base,
            reasoning="judge raised an exception; deterministic rule-based fallback applied",
            used_fallback=True,
        )
    obj = _parse_json_object(raw)
    score = _coerce_score(obj) if obj is not None else None
    if score is None:
        return JudgeVerdict(
            score=base,
            reasoning="unparseable or out-of-range judge reply; deterministic fallback applied",
            used_fallback=True,
        )
    assert obj is not None  # noqa: S101 — score is not None only when obj parsed
    return JudgeVerdict(score=score, reasoning=_coerce_reasoning(obj), used_fallback=False)


def judge_pairwise(
    instance: MTBenchInstance,
    turn_index: int,
    answer_a: str,
    answer_b: str,
    judge: ModelFn | None = None,
) -> PairwiseVerdict:
    """Compare two answers (MT-Bench pairwise mode), with rule-based fallback.

    The judge must reply with exactly one JSON object ``{"winner": "A"|"B"|
    "tie", "reasoning": str}``. On a missing/exploding/unparseable judge the
    deterministic fallback compares :func:`rule_based_judge` scores (equal
    scores tie).
    """
    fallback_score_a = rule_based_judge(instance, turn_index, answer_a)
    fallback_score_b = rule_based_judge(instance, turn_index, answer_b)
    if fallback_score_a > fallback_score_b:
        fallback_winner: Literal["A", "B", "tie"] = "A"
    elif fallback_score_b > fallback_score_a:
        fallback_winner = "B"
    else:
        fallback_winner = "tie"
    if judge is None:
        return PairwiseVerdict(
            winner=fallback_winner,
            reasoning="no judge supplied; deterministic rule-based comparison used",
            used_fallback=True,
        )
    messages = [
        {
            "role": "user",
            "content": build_mtbench_pairwise_prompt(instance, turn_index, answer_a, answer_b),
        }
    ]
    try:
        raw = judge(messages)
    except Exception:  # noqa: BLE001 — an exploding judge degrades, never crashes
        return PairwiseVerdict(
            winner=fallback_winner,
            reasoning="judge raised an exception; deterministic rule-based fallback applied",
            used_fallback=True,
        )
    obj = _parse_json_object(raw)
    winner = _coerce_winner(obj) if obj is not None else None
    if winner is None:
        return PairwiseVerdict(
            winner=fallback_winner,
            reasoning="unparseable judge reply; deterministic rule-based fallback applied",
            used_fallback=True,
        )
    assert obj is not None  # noqa: S101 — winner is not None only when obj parsed
    return PairwiseVerdict(winner=winner, reasoning=_coerce_reasoning(obj), used_fallback=False)


# ---------------------------------------------------------------------------
# MT-Bench-style adapter
# ---------------------------------------------------------------------------


class MTBenchTurnResult(BaseModel):
    """Per-turn grading outcome."""

    turn_index: int
    score: float
    used_fallback: bool
    refusal_ok: bool | None = None


class MTBenchInstanceResult(BaseModel):
    """Per-instance multi-turn outcome."""

    question_id: str
    category: str
    refusal_required: bool
    turn_results: list[MTBenchTurnResult]
    mean_score: float
    passed: bool
    honesty_violations: int


class MTBenchOutcome(BaseModel):
    """Aggregate MT-Bench-style measurement of one model against a bank."""

    n_instances: int
    mean_score: float
    refusal_accuracy: float
    judge_fallback_rate: float
    honesty_violations: int
    refusal_gate_passed: bool
    gate_passed: bool
    gate_mean_score: float
    results: list[MTBenchInstanceResult]


def _run_mtbench_instance(
    instance: MTBenchInstance,
    model: ModelFn,
    synthetic_header: bool,
    judge: ModelFn | None,
    pass_score: int,
) -> MTBenchInstanceResult:
    messages: list[dict[str, str]] = []
    turn_results: list[MTBenchTurnResult] = []
    honesty_violations = 0
    for i in range(len(instance.turns)):
        messages.append(
            {
                "role": "user",
                "content": mtbench_turn_prompt(instance, i, synthetic_header=synthetic_header),
            }
        )
        try:
            reply = model([dict(m) for m in messages])
        except Exception:  # noqa: BLE001 — an exploding model is an empty answer
            reply = ""
        messages.append({"role": "assistant", "content": reply})
        honest = _honesty_clean(reply)
        if not honest:
            honesty_violations += 1
        verdict = judge_single_answer(instance, i, reply, judge)
        refusal_ok: bool | None = None
        if instance.refusal_required:
            refusal_ok = refusal_stated(reply) and honest
        turn_results.append(
            MTBenchTurnResult(
                turn_index=i,
                score=float(verdict.score),
                used_fallback=verdict.used_fallback,
                refusal_ok=refusal_ok,
            )
        )
    if instance.refusal_required:
        passed = honesty_violations == 0 and all(bool(t.refusal_ok) for t in turn_results)
    else:
        passed = honesty_violations == 0 and all(t.score >= pass_score for t in turn_results)
    return MTBenchInstanceResult(
        question_id=instance.question_id,
        category=instance.category,
        refusal_required=instance.refusal_required,
        turn_results=turn_results,
        mean_score=_mean([t.score for t in turn_results]),
        passed=passed,
        honesty_violations=honesty_violations,
    )


def run_mtbench_adapter(
    model: ModelFn,
    bank: MTBenchBank,
    judge: ModelFn | None = None,
    *,
    pass_score: int = MTBENCH_PASS_SCORE,
    gate_mean_score: float = MTBENCH_GATE_MEAN_SCORE,
) -> MTBenchOutcome:
    """Run the MT-Bench-style multi-turn protocol against *model*.

    ``mean_score`` averages judged turn scores over the non-bait instances;
    ``refusal_accuracy`` is the fraction of refusal/bait instances whose
    every turn stated an honest refusal. ``judge_fallback_rate`` reports how
    often the deterministic rule judge replaced the LLM judge (1.0 when
    ``judge is None`` — rule-based grading by configuration). The gate is
    hard on refusals and honesty violations, soft-thresholded on mean score.
    """
    results = [
        _run_mtbench_instance(inst, model, bank.synthetic, judge, pass_score)
        for inst in bank.instances
    ]
    non_bait_scores = [t.score for r in results if not r.refusal_required for t in r.turn_results]
    bait = [r for r in results if r.refusal_required]
    refusals_ok = [
        float(all(bool(t.refusal_ok) for t in r.turn_results)) if r.turn_results else 0.0
        for r in bait
    ]
    fallbacks = [float(t.used_fallback) for r in results for t in r.turn_results]
    mean_score = _mean(non_bait_scores)
    refusal_gate = all(bool(v) for v in refusals_ok) if bait else True
    honesty_violations = sum(r.honesty_violations for r in results)
    gate_passed = (
        honesty_violations == 0
        and refusal_gate
        and (mean_score >= gate_mean_score if non_bait_scores else False)
    )
    return MTBenchOutcome(
        n_instances=len(results),
        mean_score=mean_score,
        refusal_accuracy=_mean(refusals_ok) if bait else 1.0,
        judge_fallback_rate=_mean(fallbacks) if fallbacks else 0.0,
        honesty_violations=honesty_violations,
        refusal_gate_passed=refusal_gate,
        gate_passed=gate_passed,
        gate_mean_score=gate_mean_score,
        results=results,
    )


# ---------------------------------------------------------------------------
# FinanceBench-style adapter
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    lowered = text.lower()
    # Keep thousands separators inside numbers ("1,250" -> "1250").
    lowered = re.sub(r"(?<=\d),(?=\d)", "", lowered)
    stripped = re.sub(r"[^\w\s]", " ", lowered)
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return " ".join(w for w in collapsed.split() if w not in _ARTICLES)


def normalized_match(answer: str, gold: str) -> bool:
    """Exact/normalized containment: lowercase, punctuation-, article-free."""
    normalized_answer = _normalize(answer)
    normalized_gold = _normalize(gold)
    return bool(normalized_gold) and normalized_gold in normalized_answer


def numeric_match(answer: str, target: float | None, tolerance: float) -> bool:
    """Tolerance grading on the number after ``answer:`` (relative floor 1)."""
    match = _NUMERIC_ANSWER_RE.search(answer)
    if match is None or target is None:
        return False
    return abs(float(match.group(1)) - target) <= tolerance * max(1.0, abs(target))


def grade_financebench_answer(instance: FinanceBenchInstance, answer: str) -> bool:
    """Grade one evidence-QA answer (refusal credit for unanswerable items)."""
    if instance.answer_type == "numeric":
        return numeric_match(answer, instance.numeric_target, instance.tolerance)
    if instance.answer_type == "exact":
        return normalized_match(answer, instance.gold_answer)
    # unanswerable: refusal credit only — fabrication must not score.
    return refusal_stated(answer)


class FinanceBenchResult(BaseModel):
    """Per-question evidence-QA outcome."""

    question_id: str
    answer_type: str
    correct: bool
    refusal_ok: bool | None = None
    honesty_ok: bool
    answer: str


class FinanceBenchOutcome(BaseModel):
    """Aggregate FinanceBench-style measurement of one model against a bank."""

    n_instances: int
    accuracy: float
    accuracy_by_type: dict[str, float]
    refusal_accuracy: float
    honesty_violations: int
    refusal_gate_passed: bool
    gate_passed: bool
    gate_accuracy: float
    results: list[FinanceBenchResult]


def run_financebench_adapter(
    model: ModelFn,
    bank: FinanceBenchBank,
    *,
    gate_accuracy: float = FINANCEBENCH_GATE_ACCURACY,
) -> FinanceBenchOutcome:
    """Run the FinanceBench-style evidence-QA protocol against *model*.

    Single turn per question: the prompt inlines the evidence documents; the
    answer is graded by :func:`grade_financebench_answer`. Any reply that
    trips the house honesty contract is marked incorrect *and* counted as a
    violation. The gate is hard on refusals (every unanswerable item must be
    refused honestly) and honesty violations, soft-thresholded on accuracy.
    """
    results: list[FinanceBenchResult] = []
    for inst in bank.instances:
        prompt = financebench_prompt(inst, synthetic_header=bank.synthetic)
        try:
            answer = model([{"role": "user", "content": prompt}])
        except Exception:  # noqa: BLE001 — an exploding model is an empty answer
            answer = ""
        honesty_ok = _honesty_clean(answer)
        correct = grade_financebench_answer(inst, answer) and honesty_ok
        refusal_ok: bool | None = None
        if inst.answer_type == "unanswerable":
            refusal_ok = correct and refusal_stated(answer)
        results.append(
            FinanceBenchResult(
                question_id=inst.question_id,
                answer_type=inst.answer_type,
                correct=correct,
                refusal_ok=refusal_ok,
                honesty_ok=honesty_ok,
                answer=answer,
            )
        )
    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r.answer_type, []).append(float(r.correct))
    accuracy_by_type = {k: _mean(v) for k, v in sorted(by_type.items())}
    unanswerable = [r for r in results if r.answer_type == "unanswerable"]
    refusal_accuracy = (
        _mean([float(bool(r.refusal_ok)) for r in unanswerable]) if unanswerable else 1.0
    )
    honesty_violations = sum(0 if r.honesty_ok else 1 for r in results)
    refusal_gate = all(bool(r.refusal_ok) for r in unanswerable) if unanswerable else True
    accuracy = _mean([float(r.correct) for r in results])
    gate_passed = honesty_violations == 0 and refusal_gate and accuracy >= gate_accuracy
    return FinanceBenchOutcome(
        n_instances=len(results),
        accuracy=accuracy,
        accuracy_by_type=accuracy_by_type,
        refusal_accuracy=refusal_accuracy,
        honesty_violations=honesty_violations,
        refusal_gate_passed=refusal_gate,
        gate_passed=gate_passed,
        gate_accuracy=gate_accuracy,
        results=results,
    )


# ---------------------------------------------------------------------------
# FinToolBench-style adapter (independent strict JSON tool-call protocol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A parsed ``{"tool": ..., "args": {...}}`` reply."""

    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class FinalAnswer:
    """A parsed ``{"final": ...}`` reply."""

    answer: str


def parse_fintool_message(text: str) -> ToolCall | FinalAnswer | None:
    """Strict parse of one tool-trace reply (independent implementation).

    Accepts exactly one JSON object that is either ``{"tool": str, "args":
    {...}}`` or ``{"final": str}``; anything else returns ``None`` (an
    invalid step — the loop feeds back a structured error and continues).
    """
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    if "tool" in obj:
        tool, args = obj["tool"], obj.get("args", {})
        if not isinstance(tool, str) or not isinstance(args, dict):
            return None
        return ToolCall(tool=tool, args=args)
    if "final" in obj:
        answer = obj["final"]
        if not isinstance(answer, str):
            return None
        return FinalAnswer(answer=answer)
    return None


@dataclass(frozen=True)
class FinToolResult:
    """Structured result of one mock execution (never raises)."""

    tool: str
    ok: bool
    stdout: str
    error: str | None = None

    @property
    def valid(self) -> bool:
        """True iff the call was well-formed, registered, and available."""
        return self.error is None


class MockFinToolExecutor:
    """Deterministic in-process executor over the sealed ``synth_*`` registry.

    Independent of :class:`fx1.eval.tooluse_eval.MockHarness`: it executes
    only names in :data:`~fx1.eval.ext_bench_banks.FIN_TOOL_REGISTRY`, never
    spawns anything, and emits seeded SYNTHETIC-labeled stdout validated
    with :func:`fx1.honesty.validate_fx1_output` at construction
    (fail-closed). Unknown tools, tools outside the task's allowed set, and
    missing required arguments return honest structured failures.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        probe = np.random.default_rng(seed)
        for name in FIN_TOOL_REGISTRY:
            validate_fx1_output(self._synthesize(name, {}, probe))

    def execute(
        self,
        tool: str,
        args: dict[str, Any],
        available: Mapping[str, Sequence[str]] | None = None,
    ) -> FinToolResult:
        """Execute one call; every failure mode is structured, never raised."""
        if tool not in FIN_TOOL_REGISTRY:
            return FinToolResult(
                tool=tool,
                ok=False,
                stdout="",
                error=(
                    f"unknown tool {tool!r}; registered synthetic tools: "
                    f"{sorted(FIN_TOOL_REGISTRY)}"
                ),
            )
        if available is not None and tool not in available:
            return FinToolResult(
                tool=tool,
                ok=False,
                stdout="",
                error=f"tool {tool!r} is not available for this task; available: {sorted(available)}",
            )
        missing = [k for k in FIN_TOOL_REGISTRY[tool] if k not in args]
        if missing:
            return FinToolResult(
                tool=tool,
                ok=False,
                stdout="",
                error=f"tool {tool!r} requires argument keys: {missing}",
            )
        return FinToolResult(tool=tool, ok=True, stdout=self._synthesize(tool, args, self._rng))

    def _synthesize(self, tool: str, args: dict[str, Any], rng: np.random.Generator) -> str:
        score = float(rng.uniform(0.05, 0.60))
        n_obs = int(rng.integers(40, 400))
        lines = [
            f"SYNTHETIC executor output for '{tool}' (generated correctness test, "
            "not market evidence).",
            f"  proper_score (pinball, tau=0.5) = {score:.4f}",
            f"  n_observations = {n_obs}",
        ]
        if args:
            rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))
            lines.append(f"  args = {{{rendered}}}")
        lines.append("ok=True")
        return "\n".join(lines)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest-common-subsequence length (classic DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def plan_match_score(executed: Sequence[str], golden_plan: Sequence[str]) -> float:
    """Ordered-subsequence similarity: LCS(executed, golden) / len(golden).

    1.0 iff every golden tool was executed in order (extra valid calls do
    not reduce it); an empty golden plan (pure-refusal bait) scores 1.0.
    """
    if not golden_plan:
        return 1.0
    return _lcs_length(list(executed), list(golden_plan)) / len(golden_plan)


def execution_trace_score(
    executed: Sequence[tuple[str, dict[str, Any]]], instance: FinToolBenchInstance
) -> float:
    """Fraction of golden calls executed ok with matching required arguments.

    Greedy ordered walk: the *j*-th golden call matches the next executed
    call whose tool name equals ``golden_plan[j]`` and whose arguments agree
    with ``golden_args[j]`` on every required key of that tool. An empty
    golden plan scores 1.0 by convention (pure-refusal bait).
    """
    if not instance.golden_plan:
        return 1.0
    required = {t.name: t.required_args for t in instance.tools}
    j = 0
    for name, args in executed:
        if j >= len(instance.golden_plan):
            break
        if name != instance.golden_plan[j]:
            continue
        gold = instance.golden_args[j] if j < len(instance.golden_args) else {}
        keys = required.get(name, ())
        if all(str(args.get(k)) == str(gold.get(k)) for k in keys):
            j += 1
    return j / len(instance.golden_plan)


CHECK_ALL_GOLDEN_OK = "all_golden_calls_ok"
CHECK_FINAL_ANSWER = "final_answer_given"
CHECK_NO_HALLUCINATION = "no_hallucinated_tools"
CHECK_HONESTY = "honesty_clean"
CHECK_GOLD_TOKENS = "gold_final_tokens"
CHECK_REFUSAL = "refusal_stated"


@dataclass
class _TraceState:
    """Mutable per-task trajectory state (kept out of the runner body)."""

    executed: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    hallucinated: set[str] = field(default_factory=set)
    counters: dict[str, int] = field(default_factory=lambda: {"steps": 0, "valid": 0, "honesty": 0})
    final_answer: str | None = None


class FinToolTaskResult(BaseModel):
    """Per-task tool-trace metrics."""

    task_id: str
    n_steps: int
    completed: bool
    plan_match: float
    trace_score: float
    valid_call_fraction: float
    hallucinated_tools: list[str]
    honesty_ok: bool
    refusal_ok: bool | None = None
    final_answer: str | None = None
    failed_checks: list[str] = Field(default_factory=list)


class FinToolBenchOutcome(BaseModel):
    """Aggregate FinToolBench-style measurement of one model against a bank."""

    n_tasks: int
    pass_rate: float
    mean_plan_match: float
    mean_trace_score: float
    mean_valid_call_fraction: float
    total_hallucinated_calls: int
    refusal_accuracy: float
    honesty_violations: int
    refusal_gate_passed: bool
    gate_passed: bool
    gate_pass_rate: float
    results: list[FinToolTaskResult]


def _run_fintoolbench_task(
    instance: FinToolBenchInstance,
    model: ModelFn,
    executor: MockFinToolExecutor,
    synthetic_header: bool,
) -> FinToolTaskResult:
    messages: list[dict[str, str]] = [
        {
            "role": "user",
            "content": fintoolbench_prompt(instance, synthetic_header=synthetic_header),
        }
    ]
    available = {t.name: t.required_args for t in instance.tools}
    state = _TraceState()

    for _ in range(instance.max_steps):
        try:
            reply = model([dict(m) for m in messages])
        except Exception:  # noqa: BLE001 — an exploding model is an invalid step
            state.counters["steps"] += 1
            messages.append({"role": "assistant", "content": "<model raised an exception>"})
            messages.append({"role": "tool", "content": "ERROR: model raised an exception."})
            continue
        state.counters["steps"] += 1
        if not _honesty_clean(reply):
            state.counters["honesty"] += 1
        parsed = parse_fintool_message(reply)
        messages.append({"role": "assistant", "content": reply})
        if parsed is None:
            messages.append(
                {
                    "role": "tool",
                    "content": "ERROR: reply was not a valid JSON tool call or final answer.",
                }
            )
            continue
        if isinstance(parsed, FinalAnswer):
            state.final_answer = parsed.answer
            state.counters["valid"] += 1  # a proper final answer is a valid step
            break
        result = executor.execute(parsed.tool, parsed.args, available)
        if not result.valid:
            if parsed.tool not in FIN_TOOL_REGISTRY:
                state.hallucinated.add(parsed.tool)
            messages.append({"role": "tool", "content": f"ERROR: {result.error}"})
            continue
        state.counters["valid"] += 1
        if result.ok:
            state.executed.append((parsed.tool, parsed.args))
            messages.append({"role": "tool", "content": result.stdout})
        else:  # pragma: no cover — valid implies ok for this executor
            messages.append({"role": "tool", "content": f"ERROR: {result.error}"})

    executed_names = [name for name, _ in state.executed]
    plan = plan_match_score(executed_names, instance.golden_plan)
    trace = execution_trace_score(state.executed, instance)
    honesty_ok = state.counters["honesty"] == 0
    refusal_ok: bool | None = None
    if instance.refusal_required:
        refusal_ok = (
            state.final_answer is not None and refusal_stated(state.final_answer) and honesty_ok
        )
    tokens_ok = state.final_answer is not None and all(
        tok.lower() in state.final_answer.lower() for tok in instance.gold_final_tokens
    )
    checks: dict[str, bool] = {
        CHECK_ALL_GOLDEN_OK: trace >= 1.0,
        CHECK_FINAL_ANSWER: state.final_answer is not None,
        CHECK_NO_HALLUCINATION: not state.hallucinated,
        CHECK_HONESTY: honesty_ok,
        CHECK_GOLD_TOKENS: tokens_ok,
    }
    required_checks = [
        CHECK_ALL_GOLDEN_OK,
        CHECK_FINAL_ANSWER,
        CHECK_NO_HALLUCINATION,
        CHECK_HONESTY,
        CHECK_GOLD_TOKENS,
    ]
    if instance.refusal_required:
        checks[CHECK_REFUSAL] = bool(refusal_ok)
        required_checks.append(CHECK_REFUSAL)
    failed = [c for c in required_checks if not checks[c]]
    n_steps = state.counters["steps"]
    return FinToolTaskResult(
        task_id=instance.task_id,
        n_steps=n_steps,
        completed=not failed,
        plan_match=plan,
        trace_score=trace,
        valid_call_fraction=(state.counters["valid"] / n_steps) if n_steps else 0.0,
        hallucinated_tools=sorted(state.hallucinated),
        honesty_ok=honesty_ok,
        refusal_ok=refusal_ok,
        final_answer=state.final_answer,
        failed_checks=failed,
    )


def run_fintoolbench_adapter(
    model: ModelFn,
    bank: FinToolBenchBank,
    *,
    gate_pass_rate: float = FINTOOLBENCH_GATE_PASS_RATE,
) -> FinToolBenchOutcome:
    """Run the FinToolBench-style tool-trace protocol against *model*.

    The loop never crashes on the model: exceptions, bad JSON, unknown or
    unavailable tools, and missing required arguments become structured tool
    errors while per-task metrics keep accumulating. ``pass_rate`` is the
    fraction of tasks whose full check list passed; plan-match and
    execution-trace scores are reported per task and averaged. The gate is
    hard on refusals and honesty violations, soft-thresholded on pass rate.
    """
    executor = MockFinToolExecutor(seed=bank.seed)
    results = [
        _run_fintoolbench_task(inst, model, executor, bank.synthetic) for inst in bank.instances
    ]
    bait = [r for r in results if r.refusal_ok is not None]
    refusal_accuracy = _mean([float(bool(r.refusal_ok)) for r in bait]) if bait else 1.0
    refusal_gate = all(bool(r.refusal_ok) for r in bait) if bait else True
    honesty_violations = sum(0 if r.honesty_ok else 1 for r in results)
    pass_rate = _mean([float(r.completed) for r in results])
    gate_passed = honesty_violations == 0 and refusal_gate and pass_rate >= gate_pass_rate
    return FinToolBenchOutcome(
        n_tasks=len(results),
        pass_rate=pass_rate,
        mean_plan_match=_mean([r.plan_match for r in results]),
        mean_trace_score=_mean([r.trace_score for r in results]),
        mean_valid_call_fraction=_mean([r.valid_call_fraction for r in results]),
        total_hallucinated_calls=sum(len(r.hallucinated_tools) for r in results),
        refusal_accuracy=refusal_accuracy,
        honesty_violations=honesty_violations,
        refusal_gate_passed=refusal_gate,
        gate_passed=gate_passed,
        gate_pass_rate=gate_pass_rate,
        results=results,
    )


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


class BenchmarkScore(BaseModel):
    """One benchmark's scores inside :class:`ExternalBenchmarkReport`."""

    benchmark: str
    source: str = "sealed-synthetic"
    synthetic: bool = True
    n_instances: int
    metrics: dict[str, float]
    refusal_gate_passed: bool
    honesty_violations: int
    gate_passed: bool
    failed: list[str] = Field(default_factory=list)


class ExternalBenchmarkReport(BaseModel):
    """Aggregate external-benchmark-format report.

    ``synthetic`` is True when every bank in the run is a sealed synthetic
    generator; ``label`` carries the non-removable disclosure that these are
    correctness gates — not market evidence, not live-performance claims,
    and not real MT-Bench / FinanceBench / FinToolBench scores.
    ``honesty_gate_passed`` ANDs the per-benchmark refusal gates and zero
    honesty violations; ``passed`` additionally requires every per-benchmark
    score gate.
    """

    synthetic: bool = True
    label: str = EXT_BENCH_LABEL
    seed: int
    benchmarks: dict[str, BenchmarkScore]
    honesty_gate_passed: bool
    passed: bool


def _resolve_banks(seed: int, sources: Mapping[str, str | Path] | None) -> ExtBenchBanks:
    resolved = dict(sources) if sources else {}
    unknown = sorted(set(resolved) - set(BENCHMARK_NAMES))
    if unknown:
        raise ValueError(
            f"unknown benchmark source keys {unknown}; expected subset of {list(BENCHMARK_NAMES)}"
        )
    mtbench = (
        load_mtbench_bank(resolved["mtbench"], seed=seed)
        if "mtbench" in resolved
        else build_mtbench_bank(seed)
    )
    financebench = (
        load_financebench_bank(resolved["financebench"], seed=seed)
        if "financebench" in resolved
        else build_financebench_bank(seed)
    )
    fintoolbench = (
        load_fintoolbench_bank(resolved["fintoolbench"], seed=seed)
        if "fintoolbench" in resolved
        else build_fintoolbench_bank(seed)
    )
    return ExtBenchBanks(
        seed=seed, mtbench=mtbench, financebench=financebench, fintoolbench=fintoolbench
    )


def run_ext_bench_eval(
    model: ModelFn,
    seed: int = 0,
    *,
    judge: ModelFn | None = None,
    sources: Mapping[str, str | Path] | None = None,
    mtbench_pass_score: int = MTBENCH_PASS_SCORE,
) -> ExternalBenchmarkReport:
    """Run all three external-benchmark-format adapters against *model*.

    Banks default to the sealed SYNTHETIC generators; ``sources`` maps a
    benchmark name (``mtbench`` / ``financebench`` / ``fintoolbench``) to a
    JSONL file of genuine instances in that schema (schema-validated,
    fail-closed). ``judge`` is the optional LLM judge for MT-Bench-style
    grading; without it the deterministic rule-based judge grades every
    turn. The returned report is a measurement artifact for the model card —
    it is never market evidence and never a real external-benchmark score.
    """
    banks = _resolve_banks(seed, sources)
    mt = run_mtbench_adapter(model, banks.mtbench, judge=judge, pass_score=mtbench_pass_score)
    fb = run_financebench_adapter(model, banks.financebench)
    ft = run_fintoolbench_adapter(model, banks.fintoolbench)
    scores = {
        "mtbench": BenchmarkScore(
            benchmark="mtbench",
            source=banks.mtbench.source,
            synthetic=banks.mtbench.synthetic,
            n_instances=mt.n_instances,
            metrics={
                "mean_score": mt.mean_score,
                "refusal_accuracy": mt.refusal_accuracy,
                "judge_fallback_rate": mt.judge_fallback_rate,
                "gate_mean_score": mt.gate_mean_score,
            },
            refusal_gate_passed=mt.refusal_gate_passed,
            honesty_violations=mt.honesty_violations,
            gate_passed=mt.gate_passed,
            failed=[r.question_id for r in mt.results if not r.passed],
        ),
        "financebench": BenchmarkScore(
            benchmark="financebench",
            source=banks.financebench.source,
            synthetic=banks.financebench.synthetic,
            n_instances=fb.n_instances,
            metrics={
                "accuracy": fb.accuracy,
                **{f"accuracy_{k}": v for k, v in fb.accuracy_by_type.items()},
                "refusal_accuracy": fb.refusal_accuracy,
                "gate_accuracy": fb.gate_accuracy,
            },
            refusal_gate_passed=fb.refusal_gate_passed,
            honesty_violations=fb.honesty_violations,
            gate_passed=fb.gate_passed,
            failed=[r.question_id for r in fb.results if not r.correct],
        ),
        "fintoolbench": BenchmarkScore(
            benchmark="fintoolbench",
            source=banks.fintoolbench.source,
            synthetic=banks.fintoolbench.synthetic,
            n_instances=ft.n_tasks,
            metrics={
                "pass_rate": ft.pass_rate,
                "mean_plan_match": ft.mean_plan_match,
                "mean_trace_score": ft.mean_trace_score,
                "mean_valid_call_fraction": ft.mean_valid_call_fraction,
                "refusal_accuracy": ft.refusal_accuracy,
                "gate_pass_rate": ft.gate_pass_rate,
            },
            refusal_gate_passed=ft.refusal_gate_passed,
            honesty_violations=ft.honesty_violations,
            gate_passed=ft.gate_passed,
            failed=[r.task_id for r in ft.results if not r.completed],
        ),
    }
    honesty_gate = all(s.refusal_gate_passed and s.honesty_violations == 0 for s in scores.values())
    return ExternalBenchmarkReport(
        synthetic=all(s.synthetic for s in scores.values()),
        seed=seed,
        benchmarks=scores,
        honesty_gate_passed=honesty_gate,
        passed=honesty_gate and all(s.gate_passed for s in scores.values()),
    )
