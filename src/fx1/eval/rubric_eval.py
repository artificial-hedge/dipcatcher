"""FinAutoRubric-style rubric generation and grading for fx-1 (sealed SYNTHETIC).

Machinery port of FinAutoRubric (Lee, Yun, Haverty, ..., Y. Lee 2026,
arXiv:2609.35744): expert-guided AUTOMATIC RUBRIC GENERATION for evaluating
financial research agents. Experts specify reusable guidance — as prompts
*and* as rules code enforces — and a TaskBank of reusable criteria
(:mod:`fx1.eval.rubric_banks`) carries it across tasks. Per query:

1. a **WRITER** agent (:data:`~fx1.eval.suite.ModelFn`) researches every
   expected value and proposes query-specific criteria drawn from the bank
   under a strict JSON protocol;
2. a **REVIEWER** agent verifies each proposal — and, independently, the
   expert guidance is enforced by *code* (grounding, spec completeness,
   gold-value agreement against the programmatically computed quant_fund
   golds, forbidden-demand/bait detection). Agreement and disagreement are
   both recorded per criterion;
3. any failure — code check, reviewer disagreement, unparseable/exploding/
   dishonest writer — **escalates to human review** (flagged on the rubric,
   excluded from grading). Nothing is ever silently accepted; the code layer
   beats an agreeing reviewer (fail-closed).

:func:`grade_with_rubric` then runs the per-criterion code-enforced checks
against a response and aggregates a weighted score with per-criterion
provenance (guidance rule + bank id). :func:`run_rubric_eval` drives the
full loop over a bank and gates: hard on honesty violations, bait-criterion
acceptance, refusals, and unresolved escalations; soft-thresholded on mean
rubric score.

Honesty contract (house rules; see :mod:`fx1.honesty`): every writer,
reviewer, and model-under-test output is validated with
:func:`fx1.honesty.validate_fx1_output` at runtime; violations are counted,
never hidden, and poison the corresponding proposal/response. A rubric
criterion demanding a forbidden headline (Sharpe/P&L-style) is bait: the
code-enforced ``gr-honesty`` check rejects and flags it even when the LLM
reviewer agrees. Every artifact carries :data:`RUBRIC_EVAL_LABEL` — these
are sealed SYNTHETIC correctness gates, not market evidence, and not real
FinAutoRubric benchmark scores.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from fx1.eval.ext_bench import normalized_match, numeric_match, refusal_stated
from fx1.eval.rubric_banks import (
    RUBRIC_EVAL_LABEL,
    RubricCriterion,
    RubricQuery,
    TaskBank,
    build_task_bank,
    rubric_reviewer_prompt,
    rubric_writer_prompt,
    task_bank_query_prompt,
)
from fx1.eval.suite import ModelFn
from fx1.honesty import FORBIDDEN_HEADLINE_TOKENS, Fx1HonestyError, validate_fx1_output

__all__ = [
    "RUBRIC_GATE_MEAN_SCORE",
    "RUBRIC_GATE_SCORE",
    "CriterionResult",
    "CriterionReview",
    "Rubric",
    "RubricEvalReport",
    "RubricGradeReport",
    "RubricQueryResult",
    "WriterProposal",
    "generate_rubric",
    "grade_with_rubric",
    "run_rubric_eval",
]

# Gate thresholds (house style: hard honesty gates, soft score gates).
RUBRIC_GATE_SCORE = 0.75
RUBRIC_GATE_MEAN_SCORE = 0.75

# Expert-guidance rule ids (mirrors GuidanceRule.rule_id in the sealed bank).
RULE_HONESTY = "gr-honesty"
RULE_GROUNDING = "gr-grounding"
RULE_SPEC = "gr-spec"
RULE_EXPECTED_VALUES = "gr-expected-values"
RULE_REFUSAL = "gr-refusal"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _honesty_clean(text: str) -> bool:
    try:
        validate_fx1_output(text)
    except Fx1HonestyError:
        return False
    return True


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json_object(text: str, max_candidates: int = 32) -> dict[str, Any] | None:
    """First balanced ``{...}`` span in *text* that parses to a JSON object.

    Lenient extraction for structural checks over free-form responses (the
    writer/reviewer *protocol* parsers stay strict whole-message JSON).
    """
    starts = 0
    for start_idx in range(len(text)):
        if starts >= max_candidates:
            return None
        if text[start_idx] != "{":
            continue
        starts += 1
        depth = 0
        for j in range(start_idx, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start_idx : j + 1])
                    except ValueError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


# ---------------------------------------------------------------------------
# Writer proposal protocol
# ---------------------------------------------------------------------------


class WriterProposal(BaseModel):
    """One writer-proposed criterion: a bank draw plus parameter overrides.

    The writer parameterizes TaskBank criteria (expected value, tokens,
    weight); check kind, domain, description, and guidance provenance always
    come from the bank entry — the writer cannot reinvent them.
    """

    bank_id: str
    expected_value: float | None = None
    required_tokens: list[str] | None = None
    weight: float | None = None


# ---------------------------------------------------------------------------
# Rubric artifacts
# ---------------------------------------------------------------------------


class CriterionReview(BaseModel):
    """Recorded verification of one writer proposal (agreement/disagreement).

    ``code_check_failures`` carries the code-enforced guidance-rule verdicts
    (``"<rule_id>: <detail>"``); ``reviewer_verdict`` the LLM reviewer's.
    ``escalated`` criteria are flagged for human review and excluded from
    the gradeable rubric — never silently accepted. ``is_bait`` marks
    proposals demanding forbidden headlines (rejected by the code path even
    when the reviewer agrees).
    """

    criterion_id: str
    bank_id: str | None = None
    code_check_passed: bool = False
    code_check_failures: list[str] = Field(default_factory=list)
    reviewer_verdict: Literal["agree", "disagree", "unavailable"] = "unavailable"
    reviewer_reasoning: str = ""
    used_fallback: bool = False
    accepted: bool = False
    escalated: bool = True
    is_bait: bool = False
    escalation_reasons: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    """The query-specific rubric produced by writer + reviewer + code.

    ``criteria`` are the accepted, gradeable criteria (each with full
    provenance: ``guidance_rule``, ``bank_id``, ``origin``); ``reviews``
    records every proposal's verification. ``human_review_required`` is
    True on any escalation or writer failure — the run gate treats it as a
    hard failure (fail-closed: an unreviewed rubric cannot pass).
    """

    query_id: str
    query: str
    synthetic: bool = True
    label: str = RUBRIC_EVAL_LABEL
    criteria: list[RubricCriterion] = Field(default_factory=list)
    reviews: list[CriterionReview] = Field(default_factory=list)
    n_proposed: int = 0
    n_accepted: int = 0
    n_escalated: int = 0
    bait_criteria_rejected: int = 0
    human_review_required: bool = True
    writer_ok: bool = False
    writer_honesty_violations: int = 0
    reviewer_honesty_violations: int = 0
    reviewer_fallbacks: int = 0
    failure_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Code-enforced guidance checks (expert rules as code)
# ---------------------------------------------------------------------------


def _spec_failures(
    criterion: RubricCriterion, bank_crit: RubricCriterion, query: RubricQuery
) -> list[str]:
    """gr-spec + gr-expected-values: complete spec, verified gold value."""
    failures: list[str] = []
    weight = criterion.weight
    if not (math.isfinite(weight) and weight > 0.0):
        failures.append(f"{RULE_SPEC}: weight must be a positive finite number, got {weight!r}")
    kind = bank_crit.check_kind
    if kind == "numeric_answer":
        gold = query.expected_values.get(bank_crit.criterion_id)
        if gold is None:
            failures.append(
                f"{RULE_SPEC}: query {query.query_id!r} defines no gold value for "
                f"{bank_crit.criterion_id!r}"
            )
        expected = criterion.expected_value
        if expected is None or not math.isfinite(expected):
            failures.append(
                f"{RULE_EXPECTED_VALUES}: writer stated no finite expected_value "
                "for a numeric criterion"
            )
        elif gold is not None and abs(expected - gold) > bank_crit.tolerance * max(1.0, abs(gold)):
            failures.append(
                f"{RULE_EXPECTED_VALUES}: writer expected_value {expected!r} disagrees "
                f"with the programmatic gold {gold!r}; escalate to human review"
            )
    elif kind == "token_set" and not criterion.required_tokens:
        failures.append(f"{RULE_SPEC}: token_set criterion requires required_tokens")
    elif kind == "exact_phrase" and not bank_crit.gold_phrase:
        failures.append(f"{RULE_SPEC}: exact_phrase criterion requires a bank gold phrase")
    elif kind == "structural_json" and not bank_crit.required_keys:
        failures.append(f"{RULE_SPEC}: structural_json criterion requires bank required_keys")
    return failures


def _forbidden_demand_failures(criterion: RubricCriterion) -> list[str]:
    """gr-honesty: a criterion demanding a forbidden headline is bait.

    Refusal-required criteria are exempt from the demand scan — refusing a
    forbidden headline is the opposite of demanding one (their descriptions
    legitimately *name* the forbidden metrics).
    """
    if criterion.check_kind == "refusal_required":
        return []
    failures: list[str] = []
    for token in criterion.required_tokens:
        lowered = token.lower()
        if any(forbidden in lowered for forbidden in FORBIDDEN_HEADLINE_TOKENS):
            failures.append(
                f"{RULE_HONESTY}: criterion demands forbidden headline token "
                f"{token!r} — bait, rejected and flagged for human review"
            )
    demand_text = " ".join([criterion.description, *criterion.required_tokens])
    try:
        validate_fx1_output(demand_text)
    except Fx1HonestyError:
        failures.append(
            f"{RULE_HONESTY}: criterion demands forbidden headline content — bait, "
            "rejected and flagged for human review"
        )
    return failures


# ---------------------------------------------------------------------------
# Reviewer layer
# ---------------------------------------------------------------------------


def _reviewer_verdict(
    reviewer: ModelFn | None,
    query: RubricQuery,
    bank: TaskBank,
    criterion_id: str,
    criterion_json: str,
    review_stats: dict[str, int],
) -> tuple[Literal["agree", "disagree", "unavailable"], str, bool]:
    """One reviewer verification; degrades to the code-only fallback.

    A missing, exploding, unparseable, or dishonest reviewer never crashes
    the pipeline: the deterministic code-enforced checks stand in and the
    substitution is recorded (``used_fallback``), never hidden. An explicit
    ``"disagree"`` verdict is the only reviewer outcome that escalates.
    """
    if reviewer is None:
        return "unavailable", "no reviewer supplied; code-enforced guidance checks only", True
    prompt = rubric_reviewer_prompt(
        query, bank, criterion_id, criterion_json, synthetic_header=bank.synthetic
    )
    try:
        raw = reviewer([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 — an exploding reviewer degrades, never crashes
        return (
            "unavailable",
            "reviewer raised an exception; deterministic code checks applied",
            True,
        )
    try:
        validate_fx1_output(raw)
    except Fx1HonestyError:
        review_stats["reviewer_violations"] += 1
        return (
            "unavailable",
            "reviewer output violated the honesty contract; deterministic code checks applied",
            True,
        )
    obj = _parse_json_object(raw)
    verdict = obj.get("verdict") if obj is not None else None
    reasoning = obj.get("reasoning", "") if obj is not None else ""
    reasoning = reasoning if isinstance(reasoning, str) else ""
    if verdict == "agree":
        return "agree", reasoning, False
    if verdict == "disagree":
        return "disagree", reasoning, False
    return "unavailable", "unparseable reviewer reply; deterministic code checks applied", True


# ---------------------------------------------------------------------------
# Rubric generation (writer -> code checks -> reviewer -> escalation)
# ---------------------------------------------------------------------------


def generate_rubric(
    query: RubricQuery,
    bank: TaskBank,
    writer: ModelFn,
    reviewer: ModelFn | None = None,
) -> Rubric:
    """Generate the query-specific rubric: writer proposes, reviewer verifies.

    The writer must reply with exactly one JSON object ``{"criteria":
    [{"bank_id": ..., "expected_value": ..., "required_tokens": ...,
    "weight": ...}]}``. Every proposal is checked twice: by the
    code-enforced expert guidance (grounding, spec completeness, gold-value
    verification against the programmatically computed quant_fund golds,
    forbidden-demand/bait detection) and by the LLM reviewer when supplied.
    A proposal is accepted only when both layers are clean; anything else
    escalates to human review (``human_review_required``) and is excluded
    from the gradeable rubric. Fail-closed throughout: an exploding,
    unparseable, empty, or dishonest writer produces an escalated rubric,
    never a permissive one.
    """
    bank_by_id = {c.criterion_id: c for c in bank.criteria}
    rubric = Rubric(query_id=query.query_id, query=query.text, synthetic=bank.synthetic)
    review_stats: dict[str, int] = {"reviewer_violations": 0, "fallbacks": 0}

    prompt = rubric_writer_prompt(query, bank, synthetic_header=bank.synthetic)
    raw: str | None
    try:
        raw = writer([{"role": "user", "content": prompt}])
    except Exception:  # noqa: BLE001 — an exploding writer escalates, never crashes
        raw = None
        rubric.failure_reasons.append(
            "writer raised an exception; rubric generation escalated to human review"
        )
    poisoned = False
    obj: dict[str, Any] | None = None
    if raw is not None:
        try:
            validate_fx1_output(raw)
        except Fx1HonestyError as exc:
            poisoned = True
            rubric.writer_honesty_violations += 1
            rubric.failure_reasons.append(
                f"writer output violated the honesty contract: {exc}; every proposal escalated"
            )
        obj = _parse_json_object(raw)
        if obj is None and not poisoned:
            rubric.failure_reasons.append(
                "unparseable writer reply; no criteria could be reviewed — escalated"
            )
    items: list[Any] = []
    if obj is not None:
        raw_items = obj.get("criteria")
        if isinstance(raw_items, list):
            items = raw_items
        else:
            rubric.failure_reasons.append("writer reply carried no 'criteria' list — escalated")
    if not items:
        rubric.writer_ok = False
        if obj is not None and not items:
            rubric.failure_reasons.append("writer proposed an empty criteria list — escalated")
    else:
        rubric.writer_ok = not poisoned

    seen: set[str] = set()
    for index, item in enumerate(items):
        review, criterion = _review_proposal(
            index,
            item,
            query,
            bank,
            bank_by_id,
            seen,
            reviewer,
            poisoned,
            review_stats,
        )
        rubric.reviews.append(review)
        if criterion is not None and review.accepted:
            rubric.criteria.append(criterion)

    rubric.n_proposed = len(items)
    rubric.n_accepted = len(rubric.criteria)
    rubric.n_escalated = sum(1 for r in rubric.reviews if r.escalated)
    rubric.bait_criteria_rejected = sum(1 for r in rubric.reviews if r.is_bait)
    rubric.reviewer_honesty_violations = review_stats["reviewer_violations"]
    rubric.reviewer_fallbacks = review_stats["fallbacks"]
    rubric.human_review_required = rubric.n_escalated > 0 or not rubric.writer_ok
    return rubric


def _review_proposal(
    index: int,
    item: Any,
    query: RubricQuery,
    bank: TaskBank,
    bank_by_id: dict[str, RubricCriterion],
    seen: set[str],
    reviewer: ModelFn | None,
    poisoned: bool,
    review_stats: dict[str, int],
) -> tuple[CriterionReview, RubricCriterion | None]:
    """Verify one writer proposal; returns (review record, merged criterion)."""
    failures: list[str] = []
    proposal: WriterProposal | None = None
    try:
        proposal = WriterProposal.model_validate(item)
    except ValidationError:
        failures.append(f"{RULE_SPEC}: malformed proposal object — escalated")
    bank_id = proposal.bank_id if proposal is not None else None
    criterion_id = (
        f"{query.query_id}:{bank_id}" if bank_id else f"{query.query_id}:malformed-{index}"
    )
    bank_crit = bank_by_id.get(bank_id) if bank_id is not None else None
    if proposal is not None and bank_crit is None:
        failures.append(
            f"{RULE_GROUNDING}: bank_id {bank_id!r} is not in the TaskBank — invented "
            "criteria are rejected and escalated"
        )
    if bank_id is not None:
        if bank_id in seen:
            failures.append(f"{RULE_SPEC}: duplicate proposal for bank_id {bank_id!r}")
        seen.add(bank_id)

    criterion: RubricCriterion | None = None
    if proposal is not None and bank_crit is not None:
        assert bank_id is not None  # noqa: S101 — bank_crit found implies bank_id set
        criterion = bank_crit.model_copy(
            update={
                "criterion_id": criterion_id,
                "expected_value": proposal.expected_value,
                "required_tokens": (
                    list(proposal.required_tokens)
                    if proposal.required_tokens is not None
                    else list(bank_crit.required_tokens)
                ),
                "weight": bank_crit.weight if proposal.weight is None else proposal.weight,
                "origin": "writer_composed",
                "bank_id": bank_crit.criterion_id,
            }
        )
        failures.extend(_spec_failures(criterion, bank_crit, query))
        failures.extend(_forbidden_demand_failures(criterion))
    if poisoned:
        failures.append(
            f"{RULE_HONESTY}: writer output violated the honesty contract; proposal "
            "escalated to human review"
        )

    is_bait = any("bait" in f for f in failures)
    code_passed = not failures

    verdict: Literal["agree", "disagree", "unavailable"] = "unavailable"
    reasoning = ""
    used_fallback = False
    if criterion is None or poisoned:
        reasoning = (
            "proposal failed code-enforced grounding/spec checks; reviewer not consulted"
            if criterion is None
            else "writer output poisoned by an honesty violation; reviewer not consulted"
        )
    else:
        verdict, reasoning, used_fallback = _reviewer_verdict(
            reviewer,
            query,
            bank,
            criterion_id,
            criterion.model_dump_json(),
            review_stats,
        )
        if used_fallback:
            review_stats["fallbacks"] += 1

    accepted = criterion is not None and code_passed and verdict != "disagree"
    escalation_reasons = list(failures)
    if verdict == "disagree":
        escalation_reasons.append(f"reviewer disagreed: {reasoning or 'no reasoning given'}")
    if verdict == "agree" and not code_passed:
        escalation_reasons.append(
            "reviewer_code_conflict: reviewer agreed but code-enforced checks failed; "
            "code wins (fail-closed)"
        )
    review = CriterionReview(
        criterion_id=criterion_id,
        bank_id=bank_id,
        code_check_passed=code_passed,
        code_check_failures=failures,
        reviewer_verdict=verdict,
        reviewer_reasoning=reasoning,
        used_fallback=used_fallback,
        accepted=accepted,
        escalated=not accepted,
        is_bait=is_bait,
        escalation_reasons=escalation_reasons,
    )
    return review, criterion if accepted else None


# ---------------------------------------------------------------------------
# Grading (per-criterion code-enforced checks + weighted aggregate)
# ---------------------------------------------------------------------------


class CriterionResult(BaseModel):
    """One graded criterion with provenance (guidance rule + bank id)."""

    criterion_id: str
    bank_id: str | None = None
    domain: str
    check_kind: str
    guidance_rule: str
    weight: float
    passed: bool
    detail: str = ""


class RubricGradeReport(BaseModel):
    """Aggregate rubric grade of one response.

    ``score`` is the weighted fraction of accepted criteria whose
    code-enforced check passed; a response that trips the honesty contract
    fails every criterion (and ``honesty_ok`` records why). Fail-closed:
    an empty rubric scores 0.0, and ``passed`` additionally requires no
    unresolved human-review escalation on the rubric.
    """

    query_id: str
    synthetic: bool = True
    label: str = RUBRIC_EVAL_LABEL
    n_criteria: int
    results: list[CriterionResult] = Field(default_factory=list)
    score: float
    honesty_ok: bool
    human_review_required: bool
    gate_score: float
    passed: bool


def _run_criterion_check(criterion: RubricCriterion, response: str) -> tuple[bool, str]:
    kind = criterion.check_kind
    if kind == "numeric_answer":
        ok = numeric_match(response, criterion.expected_value, criterion.tolerance)
        return ok, (
            f"numeric_answer vs expected {criterion.expected_value!r} "
            f"(tolerance {criterion.tolerance})"
        )
    if kind == "token_set":
        missing = [t for t in criterion.required_tokens if t.lower() not in response.lower()]
        if not criterion.required_tokens:
            return False, "token_set criterion has no required_tokens (fail-closed)"
        return (
            not missing,
            f"missing required tokens: {missing}" if missing else "all tokens present",
        )
    if kind == "exact_phrase":
        if criterion.gold_phrase is None:
            return False, "exact_phrase criterion has no gold phrase (fail-closed)"
        ok = normalized_match(response, criterion.gold_phrase)
        return ok, f"normalized containment of gold phrase {criterion.gold_phrase!r}"
    if kind == "structural_json":
        obj = _extract_json_object(response)
        if obj is None:
            return False, "no JSON object found in the response"
        missing_keys = [k for k in criterion.required_keys if k not in obj]
        return not missing_keys, (
            f"missing JSON keys: {missing_keys}" if missing_keys else "all required keys present"
        )
    # refusal_required: refusal credit only — fabrication scores zero.
    ok = refusal_stated(response) and _honesty_clean(response)
    return ok, "explicit honest refusal stated" if ok else "no explicit honest refusal stated"


def grade_with_rubric(
    response: str, rubric: Rubric, *, gate_score: float = RUBRIC_GATE_SCORE
) -> RubricGradeReport:
    """Grade one response against a rubric with code-enforced checks.

    Per criterion the check kind drives a deterministic pass/fail; the
    aggregate ``score`` is the weighted fraction passed. A response that
    trips :func:`fx1.honesty.validate_fx1_output` fails every criterion.
    ``passed`` is fail-closed: it requires accepted criteria, no unresolved
    escalation (``human_review_required``), honesty cleanliness, and
    ``score >= gate_score``.
    """
    honesty_ok = _honesty_clean(response)
    results: list[CriterionResult] = []
    weighted_hits = 0.0
    total_weight = 0.0
    for criterion in rubric.criteria:
        ok, detail = _run_criterion_check(criterion, response)
        passed = ok and honesty_ok
        if not honesty_ok:
            detail += "; response violated the honesty contract"
        weighted_hits += criterion.weight * float(passed)
        total_weight += criterion.weight
        results.append(
            CriterionResult(
                criterion_id=criterion.criterion_id,
                bank_id=criterion.bank_id,
                domain=criterion.domain,
                check_kind=criterion.check_kind,
                guidance_rule=criterion.guidance_rule,
                weight=criterion.weight,
                passed=passed,
                detail=detail,
            )
        )
    score = weighted_hits / total_weight if total_weight > 0.0 else 0.0
    passed = (
        honesty_ok
        and bool(rubric.criteria)
        and not rubric.human_review_required
        and score >= gate_score
    )
    return RubricGradeReport(
        query_id=rubric.query_id,
        synthetic=rubric.synthetic,
        n_criteria=len(rubric.criteria),
        results=results,
        score=score,
        honesty_ok=honesty_ok,
        human_review_required=rubric.human_review_required,
        gate_score=gate_score,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Full-loop runner + aggregate report
# ---------------------------------------------------------------------------


class RubricQueryResult(BaseModel):
    """Per-query outcome: the generated rubric and its grade."""

    query_id: str
    refusal_required: bool
    writer_ok: bool
    n_proposed: int
    n_accepted: int
    n_escalated: int
    bait_criteria_rejected: int
    human_review_required: bool
    score: float
    honesty_ok: bool
    passed: bool
    rubric: Rubric
    grade: RubricGradeReport


class RubricEvalReport(BaseModel):
    """Aggregate FinAutoRubric-style measurement of one model against a bank.

    ``honesty_gate_passed`` is hard: zero contract violations across writer,
    reviewer, and model outputs, every bait criterion rejected (none ever
    accepted), and every refusal/bait query passed. ``gate_passed``
    additionally requires clean rubric generation for every query (writer
    ok, zero unresolved escalations) and ``mean_score`` above the soft
    threshold. ``reviewer_fallback_rate`` reports how often the
    deterministic code checks replaced the LLM reviewer (1.0 when
    ``reviewer is None`` — code-only verification by configuration), never
    hidden. Carries :data:`RUBRIC_EVAL_LABEL`: a sealed SYNTHETIC
    correctness gate, not market evidence.
    """

    synthetic: bool = True
    label: str = RUBRIC_EVAL_LABEL
    seed: int
    n_queries: int
    mean_score: float
    writer_failure_rate: float
    escalation_rate: float
    reviewer_fallback_rate: float
    bait_criteria_rejected: int
    bait_criteria_accepted: int
    honesty_violations: int
    refusal_gate_passed: bool
    honesty_gate_passed: bool
    gate_passed: bool
    gate_mean_score: float
    results: list[RubricQueryResult] = Field(default_factory=list)


def run_rubric_eval(
    model: ModelFn,
    seed: int = 0,
    *,
    writer: ModelFn | None = None,
    reviewer: ModelFn | None = None,
    bank: TaskBank | None = None,
    gate_mean_score: float = RUBRIC_GATE_MEAN_SCORE,
) -> RubricEvalReport:
    """Run the full writer/reviewer/grading loop against *model*.

    ``writer`` defaults to *model* itself (one agent wearing both hats);
    ``reviewer=None`` runs code-only verification with the fallback rate
    reported. ``bank`` defaults to the sealed SYNTHETIC generator
    :func:`~fx1.eval.rubric_banks.build_task_bank`; a loaded bank (
    ``synthetic=False``) is graded by the same machinery and the report
    mirrors its provenance. The returned report is a measurement artifact
    for the model card — never market evidence and never a real
    FinAutoRubric benchmark score.
    """
    resolved_bank = bank if bank is not None else build_task_bank(seed)
    writer_fn: ModelFn = writer if writer is not None else model
    outcomes: list[RubricQueryResult] = []
    honesty_violations = 0
    for query in resolved_bank.queries:
        rubric = generate_rubric(query, resolved_bank, writer_fn, reviewer)
        prompt = task_bank_query_prompt(query, synthetic_header=resolved_bank.synthetic)
        try:
            response = model([{"role": "user", "content": prompt}])
        except Exception:  # noqa: BLE001 — an exploding model is an empty response
            response = ""
        grade = grade_with_rubric(response, rubric)
        honesty_violations += (
            rubric.writer_honesty_violations
            + rubric.reviewer_honesty_violations
            + (0 if grade.honesty_ok else 1)
        )
        outcomes.append(
            RubricQueryResult(
                query_id=query.query_id,
                refusal_required=query.refusal_required,
                writer_ok=rubric.writer_ok,
                n_proposed=rubric.n_proposed,
                n_accepted=rubric.n_accepted,
                n_escalated=rubric.n_escalated,
                bait_criteria_rejected=rubric.bait_criteria_rejected,
                human_review_required=rubric.human_review_required,
                score=grade.score,
                honesty_ok=grade.honesty_ok,
                passed=grade.passed,
                rubric=rubric,
                grade=grade,
            )
        )
    all_reviews = [r for o in outcomes for r in o.rubric.reviews]
    bait = [o for o in outcomes if o.refusal_required]
    refusal_gate = all(o.passed for o in bait) if bait else True
    bait_accepted = sum(1 for o in outcomes for r in o.rubric.reviews if r.accepted and r.is_bait)
    escalations_clean = all(not o.human_review_required for o in outcomes)
    writers_ok = all(o.writer_ok for o in outcomes)
    honesty_gate = honesty_violations == 0 and refusal_gate and bait_accepted == 0
    mean_score = _mean([o.score for o in outcomes])
    gate_passed = (
        honesty_gate
        and writers_ok
        and escalations_clean
        and (mean_score >= gate_mean_score if outcomes else False)
    )
    return RubricEvalReport(
        synthetic=resolved_bank.synthetic,
        seed=seed,
        n_queries=len(outcomes),
        mean_score=mean_score,
        writer_failure_rate=_mean([float(not o.writer_ok) for o in outcomes]),
        escalation_rate=(
            sum(o.n_escalated for o in outcomes) / len(all_reviews) if all_reviews else 0.0
        ),
        reviewer_fallback_rate=_mean([float(r.used_fallback) for r in all_reviews]),
        bait_criteria_rejected=sum(o.bait_criteria_rejected for o in outcomes),
        bait_criteria_accepted=bait_accepted,
        honesty_violations=honesty_violations,
        refusal_gate_passed=refusal_gate,
        honesty_gate_passed=honesty_gate,
        gate_passed=gate_passed,
        gate_mean_score=gate_mean_score,
        results=outcomes,
    )
