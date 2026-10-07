"""Family-level wiring for the fx-1 evaluation suite (SYNTHETIC correctness).

The house eval lane is built from seven families:

- **honesty / domain / general** — the original :data:`fx1.eval.bank.DEFAULT_BANK`
  task kinds, graded by :func:`fx1.eval.suite.score_task` (every response goes
  through :func:`fx1.honesty.validate_fx1_output`).
- **ts_reasoning / calibration / tooluse / retrieval** — the four capability
  batteries, each with its own grader and its own honesty-bait gate.

:func:`run_families` dispatches a single *model_fn* over any selection and
returns one auditable :class:`FamilyEvalSummary` with per-family pass/total
counts, per-family gates, family-specific metrics, and the legacy
``by_kind``/``results`` bank fields for backward compatibility. Everything is
a seeded **SYNTHETIC** correctness test — never market evidence; no family
surfaces Sharpe/Sortino/Calmar/P&L/NAV headlines.

Gate semantics (fail closed):

- ``honesty_gate_passed`` is true only when at least one honesty-bearing
  family ran (honesty bank kind, ts_reasoning bait, tooluse honesty,
  retrieval bait) *and* every one of them passed. A selection with no
  honesty evidence reports ``False`` — it does not vacuously pass.
- ``gate_passed`` is true only when at least one gate-bearing family ran
  (honesty, ts_reasoning, calibration, tooluse, retrieval) and all of their
  gates passed. ``domain``/``general`` are scored, not gated.
- ``ship_eligible`` mirrors ``honesty_gate_passed`` (the legacy contract:
  domain/general deltas are compared by the caller against recorded base runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from fx1.eval.bank import DEFAULT_BANK
from fx1.eval.calibration_eval import (
    DEFAULT_ECE_THRESHOLD,
    DEFAULT_Z_THRESHOLD,
    run_calibration_eval,
)
from fx1.eval.retrieval_eval import run_retrieval_eval
from fx1.eval.suite import EvalTask, ModelFn, run_suite
from fx1.eval.tooluse_eval import run_tooluse_eval
from fx1.eval.ts_reasoning import run_ts_reasoning_eval

FAMILY_HONESTY = "honesty"
FAMILY_DOMAIN = "domain"
FAMILY_GENERAL = "general"
FAMILY_TS_REASONING = "ts_reasoning"
FAMILY_CALIBRATION = "calibration"
FAMILY_TOOLUSE = "tooluse"
FAMILY_RETRIEVAL = "retrieval"

#: The three original bank kinds (the historical ``fx1 eval`` behaviour).
BANK_FAMILIES: tuple[str, ...] = (FAMILY_HONESTY, FAMILY_DOMAIN, FAMILY_GENERAL)
#: The four capability batteries wired in on top of the bank.
CAPABILITY_FAMILIES: tuple[str, ...] = (
    FAMILY_TS_REASONING,
    FAMILY_CALIBRATION,
    FAMILY_TOOLUSE,
    FAMILY_RETRIEVAL,
)
#: Every selectable family, in canonical reporting order.
FAMILY_NAMES: tuple[str, ...] = BANK_FAMILIES + CAPABILITY_FAMILIES
#: Selector meaning "every family".
ALL_FAMILIES = "all"
#: Default selection: the legacy bank, exactly as ``fx1 eval`` ran before
#: the capability families were wired in.
DEFAULT_FAMILIES: tuple[str, ...] = BANK_FAMILIES

DATA_LABEL = "SYNTHETIC — generated correctness tests, not market evidence"
SCHEMA_VERSION = "fx1.eval.families/v1"

_KIND_BY_FAMILY: dict[str, str] = {
    FAMILY_HONESTY: "honesty",
    FAMILY_DOMAIN: "domain",
    FAMILY_GENERAL: "general",
}


class FamilyResult(BaseModel):
    """Per-family result row: pass/total counts plus family-specific evidence.

    ``gate`` is the family's own fail-closed gate (``None`` when the family is
    scored only, e.g. domain/general). ``honesty_gate`` is set for
    honesty-bearing families only; ``None`` means the family carries no
    honesty-bait items.
    """

    name: str
    kind: Literal["bank", "capability"]
    passed: int
    total: int
    pass_rate: float
    gate: bool | None = None
    gate_name: str | None = None
    honesty_gate: bool | None = None
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)


class FamilyEvalSummary(BaseModel):
    """Machine-readable per-family fx-1 eval summary (SYNTHETIC correctness)."""

    schema_version: str = SCHEMA_VERSION
    data_label: str = DATA_LABEL
    seed: int
    families: list[FamilyResult]
    passed: int
    total: int
    pass_rate: float
    gate_passed: bool
    honesty_gate_passed: bool
    ship_eligible: bool
    # Legacy bank fields (empty when no bank family was selected) so existing
    # consumers of the old ``run_suite`` summary keep working.
    results: list[dict[str, Any]] = Field(default_factory=list)
    by_kind: dict[str, dict[str, int]] = Field(default_factory=dict)

    def family(self, name: str) -> FamilyResult:
        """Return one family row by name (KeyError if not selected)."""
        for row in self.families:
            if row.name == name:
                return row
        raise KeyError(name)


def resolve_families(selection: Sequence[str] | None) -> tuple[str, ...]:
    """Normalise a family selection into canonical order (deduplicated).

    Each entry may be a family name or a comma-separated list; ``"all"``
    expands to every :data:`FAMILY_NAMES` entry. ``None``/empty selects
    :data:`DEFAULT_FAMILIES` (the legacy bank). Unknown names raise ValueError.
    """
    if not selection:
        return DEFAULT_FAMILIES
    requested: set[str] = set()
    for raw in selection:
        for token in str(raw).split(","):
            name = token.strip().lower()
            if not name:
                continue
            if name == ALL_FAMILIES:
                requested.update(FAMILY_NAMES)
            elif name in FAMILY_NAMES:
                requested.add(name)
            else:
                valid = ", ".join((*FAMILY_NAMES, ALL_FAMILIES))
                raise ValueError(f"unknown eval family {name!r}; choose from {valid}")
    ordered = tuple(name for name in FAMILY_NAMES if name in requested)
    return ordered or DEFAULT_FAMILIES


def _bank_row(
    name: str, suite_results: list[dict[str, Any]], by_kind: dict[str, dict[str, int]]
) -> FamilyResult:
    kind = _KIND_BY_FAMILY[name]
    counts = by_kind.get(kind, {"passed": 0, "total": 0})
    passed, total = int(counts["passed"]), int(counts["total"])
    gate: bool | None = None
    honesty_gate: bool | None = None
    gate_name: str | None = None
    if name == FAMILY_HONESTY:
        gate = total > 0 and passed == total
        honesty_gate = gate
        gate_name = "honesty_gate"
    return FamilyResult(
        name=name,
        kind="bank",
        passed=passed,
        total=total,
        pass_rate=(passed / total) if total else 0.0,
        gate=gate,
        gate_name=gate_name,
        honesty_gate=honesty_gate,
        metrics={"by_kind": kind},
        detail={"results": [r for r in suite_results if r["kind"] == kind]},
    )


def _ts_reasoning_row(model_fn: ModelFn, seed: int, n_instances: int) -> FamilyResult:
    report = run_ts_reasoning_eval(model_fn, seed=seed, n_instances=n_instances)
    passed = sum(1 for r in report.results if r.passed)
    return FamilyResult(
        name=FAMILY_TS_REASONING,
        kind="capability",
        passed=passed,
        total=report.n_tasks,
        pass_rate=report.overall,
        gate=report.honesty_gate_passed,
        gate_name="honesty-bait gate",
        honesty_gate=report.honesty_gate_passed,
        metrics={"overall": report.overall, **dict(report.by_family)},
        detail={
            "seed": report.seed,
            "by_family": report.by_family,
            "results": [r.model_dump() for r in report.results],
        },
    )


def _calibration_row(
    model_fn: ModelFn, seed: int, n_bins: int, ece_threshold: float, z_threshold: float
) -> FamilyResult:
    report = run_calibration_eval(
        model_fn, seed=seed, n_bins=n_bins, ece_threshold=ece_threshold, z_threshold=z_threshold
    )
    passed = int(report.passed)
    return FamilyResult(
        name=FAMILY_CALIBRATION,
        kind="capability",
        # One gate, not per-question credit: the calibration contract is the
        # ECE/Z gate itself, so the family contributes a single pass/total row.
        passed=passed,
        total=1,
        pass_rate=float(passed),
        gate=report.passed,
        gate_name="calibration gate (ECE + Spiegelhalter Z)",
        honesty_gate=None,
        metrics={
            "ece": report.ece,
            "spiegelhalter_z": report.spiegelhalter_z,
            "n_questions": report.n_questions,
            "n_unparseable": report.n_unparseable,
            "ece_threshold": ece_threshold,
            "z_threshold": z_threshold,
        },
        detail={
            "seed": report.seed,
            "bins": [asdict(b) for b in report.bins],
            "extracted": [float(x) for x in report.extracted],
        },
    )


def _tooluse_row(model_fn: ModelFn, seed: int, n_tasks: int) -> FamilyResult:
    report = run_tooluse_eval(model_fn, seed=seed, n_tasks=n_tasks)
    honesty_ok = report.honesty_violations == 0
    passed = sum(1 for o in report.outcomes if o.completed)
    return FamilyResult(
        name=FAMILY_TOOLUSE,
        kind="capability",
        passed=passed,
        total=report.n_tasks,
        pass_rate=report.pass_rate,
        gate=honesty_ok,
        gate_name="honesty gate",
        honesty_gate=honesty_ok,
        metrics={
            "pass_rate": report.pass_rate,
            "mean_valid_call_fraction": report.mean_valid_call_fraction,
            "mean_plan_match": report.mean_plan_match,
            "total_hallucinated_calls": report.total_hallucinated_calls,
            "honesty_violations": report.honesty_violations,
        },
        detail={"seed": report.seed, "outcomes": [asdict(o) for o in report.outcomes]},
    )


def _retrieval_row(
    model_fn: ModelFn, seed: int, n_questions: int, max_retrieves: int
) -> FamilyResult:
    report = run_retrieval_eval(
        model_fn, seed=seed, n_questions=n_questions, max_retrieves=max_retrieves
    )
    passed = sum(1 for r in report.results if r.correct)
    return FamilyResult(
        name=FAMILY_RETRIEVAL,
        kind="capability",
        passed=passed,
        total=report.n_questions,
        pass_rate=report.accuracy,
        gate=report.honesty_gate_passed,
        gate_name="honesty-bait gate",
        honesty_gate=report.honesty_gate_passed,
        metrics={
            "accuracy": report.accuracy,
            "retrieval_precision": report.retrieval_precision,
            "citation_accuracy": report.citation_accuracy,
        },
        detail={"seed": report.seed, "results": [asdict(r) for r in report.results]},
    )


def run_families(
    model_fn: ModelFn,
    families: Sequence[str] | None = None,
    *,
    seed: int = 0,
    ts_n_instances: int = 40,
    calibration_n_bins: int = 10,
    calibration_ece_threshold: float = DEFAULT_ECE_THRESHOLD,
    calibration_z_threshold: float = DEFAULT_Z_THRESHOLD,
    tooluse_n_tasks: int = 12,
    retrieval_n_questions: int = 24,
    retrieval_max_retrieves: int = 3,
) -> FamilyEvalSummary:
    """Run the selected *families* against *model_fn* and aggregate the results.

    Returns per-family pass/total counts, per-family gates, family-specific
    metrics, and the legacy bank ``results``/``by_kind`` fields. All seeded
    families are built from *seed*; capability parameters default to each
    battery's own defaults. The model is never allowed to crash the run — each
    battery counts exceptions fail-closed.
    """
    names = resolve_families(families)
    bank_kinds = {_KIND_BY_FAMILY[name] for name in names if name in _KIND_BY_FAMILY}
    bank_tasks: list[EvalTask] = [t for t in DEFAULT_BANK if t.kind in bank_kinds]
    suite_results: list[dict[str, Any]] = []
    by_kind: dict[str, dict[str, int]] = {}
    if bank_tasks:
        suite = run_suite(model_fn, bank_tasks)
        suite_results = suite.results
        by_kind = suite.by_kind

    rows: list[FamilyResult] = []
    for name in names:
        if name in _KIND_BY_FAMILY:
            rows.append(_bank_row(name, suite_results, by_kind))
        elif name == FAMILY_TS_REASONING:
            rows.append(_ts_reasoning_row(model_fn, seed, ts_n_instances))
        elif name == FAMILY_CALIBRATION:
            rows.append(
                _calibration_row(
                    model_fn,
                    seed,
                    calibration_n_bins,
                    calibration_ece_threshold,
                    calibration_z_threshold,
                )
            )
        elif name == FAMILY_TOOLUSE:
            rows.append(_tooluse_row(model_fn, seed, tooluse_n_tasks))
        else:  # FAMILY_RETRIEVAL
            rows.append(
                _retrieval_row(model_fn, seed, retrieval_n_questions, retrieval_max_retrieves)
            )

    total = sum(row.total for row in rows)
    passed = sum(row.passed for row in rows)
    gate_components = [row.gate for row in rows if row.gate is not None]
    honesty_components = [row.honesty_gate for row in rows if row.honesty_gate is not None]
    # Fail closed: "no gate ran" is not a pass (a selection without honesty
    # evidence can never satisfy the training / ship gate).
    gate_passed = bool(gate_components) and all(gate_components)
    honesty_gate_passed = bool(honesty_components) and all(honesty_components)
    return FamilyEvalSummary(
        seed=seed,
        families=rows,
        passed=passed,
        total=total,
        pass_rate=round(passed / total, 6) if total else 0.0,
        gate_passed=gate_passed,
        honesty_gate_passed=honesty_gate_passed,
        ship_eligible=honesty_gate_passed,
        results=suite_results,
        by_kind=by_kind,
    )


__all__ = [
    "ALL_FAMILIES",
    "BANK_FAMILIES",
    "CAPABILITY_FAMILIES",
    "DATA_LABEL",
    "DEFAULT_FAMILIES",
    "FAMILY_CALIBRATION",
    "FAMILY_DOMAIN",
    "FAMILY_GENERAL",
    "FAMILY_HONESTY",
    "FAMILY_NAMES",
    "FAMILY_RETRIEVAL",
    "FAMILY_TOOLUSE",
    "FAMILY_TS_REASONING",
    "SCHEMA_VERSION",
    "FamilyEvalSummary",
    "FamilyResult",
    "resolve_families",
    "run_families",
]
