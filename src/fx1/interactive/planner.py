"""Qualified planner for ``/superpower``.

Selection uses catalog task tags and aliases (word-boundary match with simple
plural folding). Purpose keywords only add rank after a tag or alias hit;
they never select a capability on their own. Ordering is score descending,
then id ascending.

Only executable capabilities become steps. Dependencies are inserted and
count toward ``max_steps``. ``post_checks`` stay on the step. Declared
defaults are used for optional arguments. A required ``output_dir`` is the
first configured output root, and a required input path is that root only
when the capability does not require a caller dataset. Required choices are
filled from goal tokens that name them. On a dataset-bound capability with
attached datasets, arguments whose ``bind`` is ``dataset`` or ``goal`` are
filled from those datasets and the goal, and ``--out-dir`` comes from
``roots['output_dir']`` when that root is present. Availability columns and
any other unbound argument are never guessed. Bound values go through
:func:`validate_args`. A model may propose arguments; each proposal is
validated and never coerced. Any structural problem in the model reply keeps
the deterministic plan.

Alias and task phrases score first. A goal token of length 5 or more that
belongs to exactly one executable capability also scores, so ``serial``
selects ``serial-watch`` without the full alias. ``research`` and
``research-only`` stay distinct ids even when they share ``argv[0]``.

When the goal refers to user data, or ``datasets`` is non-empty, a
synthetic-only step is kept only with an explicit SYNTHETIC note. A planned
dataset-bound runner drops synthetic lanes that cover the same task unless
the goal asks for a synthetic, self-test, engine-correctness, or sanity
check. A ``needs_dataset`` exclusion is recorded unless a dataset-bound step
was actually planned; with no dataset attached the detail asks for
``/data attach``. Ambiguous datasets or score tasks record ``needs_input``.
Dataset-bound means ``requires_dataset``, an evidence class other than
``SYNTHETIC``, and a data label other than ``SYNTHETIC``.
Experimental qualification is reported as ``unqualified``. Budget trimming
drops a suffix of the ordered plan and does not reorder steps.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.contracts import (
    ArgSpec,
    Capability,
    CapabilityCatalog,
    ContractViolation,
    contains_shell_metacharacters,
    validate_args,
)

ModelFn = Callable[[str], str]
EXCLUSION_CODES = frozenset(
    {
        "not_executable",
        "retired",
        "unqualified",
        "needs_dataset",
        "needs_input",
        "budget",
        "max_steps",
        "invalid_args",
        "unknown_id",
        "duplicate",
        "no_match",
        "dependency_missing",
    }
)
SYNTHETIC_NOTE = "SYNTHETIC correctness evidence only — it does not evaluate your data"
_NO_MATCH = "No suitable capability"
_NEEDS_DATASET = "no qualified dataset-bound runner matched"
_ATTACH_DATASET = "attach a CSV/Parquet with /data attach <path>"
_SCORE_QUESTION = (
    "Which score: volatility (QLIKE), quantile (pinball/CRPS) or probability (Brier/ECE)?"
)
_SCORE_CHOICES = frozenset({"volatility", "quantile", "probability"})
_SYNTHETIC_REQUESTS = ("synthetic", "self test", "engine correctness", "sanity")
_STOP = frozenset(
    {
        "about",
        "across",
        "after",
        "again",
        "before",
        "between",
        "cannot",
        "could",
        "during",
        "inside",
        "model",
        "narrowed",
        "other",
        "should",
        "their",
        "there",
        "these",
        "those",
        "under",
        "using",
        "where",
        "which",
        "while",
        "within",
        "without",
        "would",
    }
)
_USER_WORDS = ("attached", "supplied")
_PATH_RE = re.compile(
    r"(?:(?:^|[\s\"'])(?:\.{1,2}[/\\]|[/\\]|[A-Za-z]:[/\\])\S+)"
    r"|(?:\b[\w.-]+\.(?:csv|parquet|jsonl|json|feather|arrow)\b)",
    re.IGNORECASE,
)
Ranked = tuple[int, Capability, str]
Roots = Mapping[str, Sequence[Path]]


@dataclass(frozen=True)
class DatasetRef:
    """A dataset the caller says is already available."""

    path: str
    kind: str
    label: str


@dataclass(frozen=True)
class PlanStep:
    """One scheduled capability invocation."""

    capability_id: str
    args: Mapping[str, Any]
    rationale: str
    depends_on: tuple[str, ...]
    evidence_class: str
    consequential: bool
    est_runtime_s: float
    step_key: str
    scores: tuple[str, ...] = ()
    post_checks: tuple[Mapping[str, Any], ...] = ()
    max_output_chars: int = 0


@dataclass(frozen=True)
class Exclusion:
    """Why a capability or proposal was left out of the plan."""

    capability_id: str | None
    code: str
    detail: str

    def __post_init__(self) -> None:
        if self.code not in EXCLUSION_CODES:
            raise ValueError(f"unknown exclusion code {self.code}")


@dataclass
class ExecutionPlan:
    """A validated selection for one goal."""

    goal: str
    steps: list[PlanStep]
    exclusions: list[Exclusion]
    notes: list[str]
    source: str
    catalog_digest: str
    max_steps: int
    budget_s: float | None = None

    def describe(self, width: int = 100) -> str:
        """Human-readable plan: steps, scores, limits, exclusions, and notes."""
        return _wrap(_describe_text(self), width)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe provenance record."""
        payload = {
            "budget_s": self.budget_s,
            "catalog_digest": self.catalog_digest,
            "exclusions": [_exclusion_dict(item) for item in self.exclusions],
            "goal": self.goal,
            "max_steps": self.max_steps,
            "notes": list(self.notes),
            "source": self.source,
            "steps": [_step_dict(step) for step in self.steps],
        }
        encoded = json.dumps(payload, sort_keys=True)
        loaded: dict[str, Any] = json.loads(encoded)
        return loaded


def plan_goal(
    goal: str,
    catalog: CapabilityCatalog,
    *,
    max_steps: int = 4,
    budget_s: float | None = None,
    datasets: Sequence[DatasetRef] = (),
    roots: Roots | None = None,
    model_fn: ModelFn | None = None,
) -> ExecutionPlan:
    """Plan ``goal`` from ``catalog``.

    ``model_fn`` may replace the deterministic selection. Structural model
    failures keep that selection and record a note. The returned plan's
    steps satisfy :func:`validate_plan` for the same limits.
    """
    _check_limits(max_steps, budget_s)
    root_map: Roots = {} if roots is None else roots
    deterministic = _deterministic_plan(
        goal,
        catalog,
        max_steps=max_steps,
        budget_s=budget_s,
        datasets=tuple(datasets),
        roots=root_map,
    )
    if model_fn is None:
        return deterministic
    return _refine(
        deterministic,
        goal,
        catalog,
        model_fn,
        max_steps=max_steps,
        budget_s=budget_s,
        datasets=tuple(datasets),
        roots=root_map,
    )


def validate_plan(
    plan: ExecutionPlan,
    catalog: CapabilityCatalog,
    *,
    max_steps: int,
    budget_s: float | None,
    roots: Roots | None = None,
) -> list[Exclusion]:
    """Return problems with ``plan``. An empty list means the steps are valid."""
    _check_limits(max_steps, budget_s)
    root_map: Roots = {} if roots is None else roots
    problems: list[Exclusion] = []
    problems.extend(_count_problems(plan, max_steps))
    problems.extend(_duplicate_problems(plan))
    problems.extend(_step_problems(plan, catalog, root_map))
    problems.extend(_budget_problems(plan, budget_s))
    problems.extend(_order_problems(plan, catalog))
    return problems


def order_steps(steps: Sequence[PlanStep], catalog: CapabilityCatalog) -> list[PlanStep]:
    """Stable dependency order. Cyclic steps keep their input order at the end."""
    indexed = list(enumerate(steps))
    deps = {index: _dep_indexes(step, index, indexed, catalog) for index, step in indexed}
    return _topo(indexed, deps)


def make_step_key(capability_id: str, args: Mapping[str, Any]) -> str:
    """Stable idempotency key for one capability invocation."""
    payload = {"args": _plain(args), "capability": capability_id}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _deterministic_plan(
    goal: str,
    catalog: CapabilityCatalog,
    *,
    max_steps: int,
    budget_s: float | None,
    datasets: Sequence[DatasetRef],
    roots: Roots,
) -> ExecutionPlan:
    ranked = _rank(goal, catalog)
    user_data = _user_data(goal, datasets)
    steps, exclusions = _select(
        ranked,
        catalog,
        roots,
        max_steps,
        goal=goal,
        user_data=user_data,
        datasets=datasets,
    )
    ordered = order_steps(steps, catalog)
    kept, budget_ex = _apply_budget(ordered, budget_s)
    exclusions.extend(budget_ex)
    notes: list[str] = []
    _apply_honesty(
        kept,
        catalog,
        user_data=user_data,
        exclusions=exclusions,
        notes=notes,
        datasets=datasets,
        goal=goal,
    )
    _note_if_empty(kept, notes)
    return _new_plan(
        goal,
        kept,
        exclusions,
        notes,
        source="deterministic",
        catalog=catalog,
        max_steps=max_steps,
        budget_s=budget_s,
    )


def _refine(
    fallback: ExecutionPlan,
    goal: str,
    catalog: CapabilityCatalog,
    model_fn: ModelFn,
    *,
    max_steps: int,
    budget_s: float | None,
    datasets: Sequence[DatasetRef],
    roots: Roots,
) -> ExecutionPlan:
    raw = _call_model(model_fn, _prompt(goal, catalog))
    if isinstance(raw, Exclusion):
        return _with_note(fallback, raw.detail)
    steps, exclusions = _steps_from_model(raw["steps"], catalog, roots, max_steps, goal)
    ordered = order_steps(steps, catalog)
    kept, budget_ex = _apply_budget(ordered, budget_s)
    exclusions.extend(budget_ex)
    exclusions[0:0] = _status_exclusions(fallback)
    notes: list[str] = []
    model_note = _model_note(str(raw.get("rationale", "")))
    if model_note is not None:
        notes.append(model_note)
    user_data = _user_data(goal, datasets)
    _apply_honesty(
        kept,
        catalog,
        user_data=user_data,
        exclusions=exclusions,
        notes=notes,
        datasets=datasets,
        goal=goal,
    )
    if not kept and not any(item.code == "no_match" for item in exclusions):
        exclusions.append(Exclusion(None, "no_match", _NO_MATCH))
    _note_if_empty(kept, notes)
    return _new_plan(
        goal,
        kept,
        exclusions,
        notes,
        source="model+validated",
        catalog=catalog,
        max_steps=max_steps,
        budget_s=budget_s,
    )


def _call_model(model_fn: ModelFn, prompt: str) -> Mapping[str, Any] | Exclusion:
    try:
        raw = model_fn(prompt)
    except Exception:  # noqa: BLE001 — model failures fall back to the deterministic plan
        return Exclusion(None, "invalid_args", "model refinement rejected: model call failed")
    if not isinstance(raw, str):
        return Exclusion(None, "invalid_args", "model refinement rejected: reply was not text")
    try:
        return _parse_model_payload(raw)
    except (ValueError, RecursionError, OverflowError) as exc:
        return Exclusion(None, "invalid_args", f"model refinement rejected: {exc}")


def _new_plan(
    goal: str,
    steps: list[PlanStep],
    exclusions: list[Exclusion],
    notes: list[str],
    *,
    source: str,
    catalog: CapabilityCatalog,
    max_steps: int,
    budget_s: float | None,
) -> ExecutionPlan:
    return ExecutionPlan(
        goal=goal,
        steps=steps,
        exclusions=_dedupe_exclusions(exclusions),
        notes=_dedupe_notes(notes),
        source=source,
        catalog_digest=catalog.digest,
        max_steps=max_steps,
        budget_s=None if budget_s is None else float(budget_s),
    )


def _with_note(plan: ExecutionPlan, note: str) -> ExecutionPlan:
    if note not in plan.notes:
        plan.notes.append(note)
    return plan


def _select(
    ranked: Sequence[Ranked],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    *,
    goal: str,
    user_data: bool,
    datasets: Sequence[DatasetRef],
) -> tuple[list[PlanStep], list[Exclusion]]:
    status = [_status_exclusion(cap) for _, cap, _detail in ranked if not cap.executable]
    runnable = [row for row in ranked if row[1].executable]
    if not runnable and not status:
        return [], [Exclusion(None, "no_match", _NO_MATCH)]
    if user_data:
        return _select_user_data(runnable, status, catalog, roots, max_steps, goal, datasets)
    steps, problems = _take_steps(runnable, catalog, roots, max_steps, goal, datasets)
    return steps, [*status, *problems]


def _select_user_data(
    runnable: Sequence[Ranked],
    status: list[Exclusion],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
    datasets: Sequence[DatasetRef],
) -> tuple[list[PlanStep], list[Exclusion]]:
    bound = [row for row in runnable if _dataset_bound(row[1])]
    other = [row for row in runnable if not _dataset_bound(row[1])]
    if not bound:
        return _prefixed(status, *_take_steps(other, catalog, roots, max_steps, goal, datasets))
    steps, problems = _take_steps(bound, catalog, roots, max_steps, goal, datasets)
    if steps:
        return _with_explicit_synthetic(
            steps, problems, other, status, catalog, roots, max_steps, goal, datasets
        )
    return _without_synthetic_fallback(
        other, status, problems, catalog, roots, max_steps, goal, datasets
    )


def _prefixed(
    status: Sequence[Exclusion], steps: list[PlanStep], problems: list[Exclusion]
) -> tuple[list[PlanStep], list[Exclusion]]:
    return steps, [*status, *problems]


def _with_explicit_synthetic(
    steps: list[PlanStep],
    problems: list[Exclusion],
    other: Sequence[Ranked],
    status: Sequence[Exclusion],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
    datasets: Sequence[DatasetRef],
) -> tuple[list[PlanStep], list[Exclusion]]:
    if not _wants_synthetic(goal):
        return steps, [*status, *problems]
    extra = _append_same_task_synthetic(steps, other, catalog, roots, max_steps, goal, datasets)
    return steps, [*status, *problems, *extra]


def _without_synthetic_fallback(
    other: Sequence[Ranked],
    status: Sequence[Exclusion],
    problems: list[Exclusion],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
    datasets: Sequence[DatasetRef],
) -> tuple[list[PlanStep], list[Exclusion]]:
    plain = [row for row in other if not _synthetic_only(row[1])]
    fallback, more = _take_steps(plain, catalog, roots, max_steps, goal, datasets)
    return fallback, [*status, *problems, *more]


def _take_steps(
    rows: Sequence[Ranked],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str = "",
    datasets: Sequence[DatasetRef] = (),
) -> tuple[list[PlanStep], list[Exclusion]]:
    chosen: list[PlanStep] = []
    seen: set[str] = set()
    exclusions: list[Exclusion] = []
    _absorb_ranked(rows, chosen, seen, exclusions, catalog, roots, max_steps, goal, datasets)
    return chosen, exclusions


def _absorb_ranked(
    rows: Sequence[Ranked],
    chosen: list[PlanStep],
    seen: set[str],
    exclusions: list[Exclusion],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
    datasets: Sequence[DatasetRef],
) -> None:
    for _score, cap, detail in rows:
        if cap.id in seen:
            continue
        rationale = f"selected because the goal matched {detail}"
        added, problem = _consider(
            cap, rationale, catalog, roots, chosen, seen, max_steps, goal, datasets
        )
        if problem is not None:
            exclusions.append(problem)
        chosen.extend(added)


def _consider(
    cap: Capability,
    detail: str,
    catalog: CapabilityCatalog,
    roots: Roots,
    chosen: Sequence[PlanStep],
    seen: set[str],
    max_steps: int,
    goal: str = "",
    datasets: Sequence[DatasetRef] = (),
) -> tuple[list[PlanStep], Exclusion | None]:
    steps, problem = _try_chain(cap, catalog, roots, detail, goal, datasets)
    if problem is not None:
        return [], problem
    fresh = [step for step in steps if step.capability_id not in seen]
    if len(chosen) + len(fresh) > max_steps:
        return [], Exclusion(
            cap.id, "max_steps", f"adding {cap.id} would exceed max_steps={max_steps}"
        )
    for step in fresh:
        seen.add(step.capability_id)
    return fresh, None


def _try_chain(
    cap: Capability,
    catalog: CapabilityCatalog,
    roots: Roots,
    detail: str,
    goal: str = "",
    datasets: Sequence[DatasetRef] = (),
) -> tuple[list[PlanStep], Exclusion | None]:
    ordered: list[str] = []
    if not _linearize(cap.id, catalog, {}, ordered):
        return [], Exclusion(
            cap.id, "dependency_missing", f"{cap.id} dependency is missing or not executable"
        )
    deps = [item for item in ordered if item != cap.id]
    steps, problem = _default_steps(deps, catalog, roots, cap.id, goal, datasets)
    if problem is not None:
        return [], problem
    built = _step_from_defaults(cap, roots, detail, goal, datasets)
    if isinstance(built, Exclusion):
        return [], built
    steps.append(built)
    return steps, None


def _default_steps(
    dep_ids: Sequence[str],
    catalog: CapabilityCatalog,
    roots: Roots,
    parent: str,
    goal: str = "",
    datasets: Sequence[DatasetRef] = (),
) -> tuple[list[PlanStep], Exclusion | None]:
    steps: list[PlanStep] = []
    for dep_id in dep_ids:
        dep = catalog.get(dep_id)
        if dep is None:
            return [], Exclusion(parent, "dependency_missing", f"{parent} dependency is missing")
        built = _step_from_defaults(dep, roots, f"prerequisite of {parent}", goal, datasets)
        if isinstance(built, Exclusion):
            return [], built
        steps.append(built)
    return steps, None


def _step_from_defaults(
    cap: Capability,
    roots: Roots,
    rationale: str,
    goal: str = "",
    datasets: Sequence[DatasetRef] = (),
) -> PlanStep | Exclusion:
    bound, problem = _prebound(cap, goal, datasets, roots)
    if problem is not None:
        return problem
    provided = _suggested_args(cap, goal, roots)
    provided.update(bound)
    try:
        args = validate_args(cap, provided, roots=roots)
    except ContractViolation as exc:
        return Exclusion(cap.id, "invalid_args", f"{exc.code}:{exc.arg}")
    text = _compose_rationale(rationale, cap, args, goal)
    return _new_step(cap, args, _safe_text(text, f"selected {cap.id}"))


def _linearize(
    cap_id: str, catalog: CapabilityCatalog, state: dict[str, int], out: list[str]
) -> bool:
    mark = state.get(cap_id, 0)
    if mark == 2:
        return True
    if mark == 1:
        return False
    cap = catalog.get(cap_id)
    if cap is None or not cap.executable:
        return False
    state[cap_id] = 1
    for dep in cap.dependencies:
        if not _linearize(dep, catalog, state, out):
            return False
    state[cap_id] = 2
    out.append(cap_id)
    return True


def _apply_budget(
    steps: Sequence[PlanStep], budget_s: float | None
) -> tuple[list[PlanStep], list[Exclusion]]:
    if budget_s is None:
        return list(steps), []
    total = 0.0
    limit = float(budget_s)
    for index, step in enumerate(steps):
        if total + step.est_runtime_s > limit:
            excluded = [
                Exclusion(item.capability_id, "budget", "estimated runtime exceeds budget")
                for item in steps[index:]
            ]
            return list(steps[:index]), excluded
        total += step.est_runtime_s
    return list(steps), []


def _apply_honesty(
    steps: Sequence[PlanStep],
    catalog: CapabilityCatalog,
    *,
    user_data: bool,
    exclusions: list[Exclusion],
    notes: list[str],
    datasets: Sequence[DatasetRef] = (),
    goal: str = "",
) -> None:
    if not user_data:
        return
    if _any_synthetic(steps, catalog) and SYNTHETIC_NOTE not in notes:
        notes.append(SYNTHETIC_NOTE)
    if _dataset_gap_recorded(steps, catalog, exclusions):
        return
    exclusions.append(Exclusion(None, "needs_dataset", _dataset_request(datasets, goal)))


def _dataset_gap_recorded(
    steps: Sequence[PlanStep], catalog: CapabilityCatalog, exclusions: Sequence[Exclusion]
) -> bool:
    if _any_dataset_bound(steps, catalog) or _asked_for_input(exclusions):
        return True
    return any(item.code == "needs_dataset" for item in exclusions)


def _asked_for_input(exclusions: Sequence[Exclusion]) -> bool:
    return any(item.code == "needs_input" for item in exclusions)


def _dataset_request(datasets: Sequence[DatasetRef], goal: str) -> str:
    """Fallback when no dataset-bound runner was planned.

    A matched dataset-bound runner that has no file records ``/data attach``
    itself. This fallback stays the generic gap so a synthetic-only plan can
    still say that no dataset-bound runner matched.
    """
    del datasets, goal
    return _NEEDS_DATASET


def _any_synthetic(steps: Sequence[PlanStep], catalog: CapabilityCatalog) -> bool:
    return any(_step_is(step, catalog, _synthetic_only) for step in steps)


def _any_dataset_bound(steps: Sequence[PlanStep], catalog: CapabilityCatalog) -> bool:
    return any(_step_is(step, catalog, _dataset_bound) for step in steps)


def _step_is(
    step: PlanStep, catalog: CapabilityCatalog, predicate: Callable[[Capability], bool]
) -> bool:
    cap = catalog.get(step.capability_id)
    return cap is not None and predicate(cap)


def _note_if_empty(steps: Sequence[PlanStep], notes: list[str]) -> None:
    if not steps and _NO_MATCH not in notes:
        notes.append(_NO_MATCH)


def _rank(goal: str, catalog: CapabilityCatalog) -> list[Ranked]:
    norm = _normalize(goal)
    rare = _rare_tokens(catalog)
    rows: list[Ranked] = []
    for cap in catalog.capabilities.values():
        score, detail = _score_capability(norm, cap, rare)
        if score >= 8:
            rows.append((score, cap, detail))
    rows.sort(key=lambda row: (-row[0], row[1].id))
    return rows


def _score_capability(norm: str, cap: Capability, rare: Mapping[str, str]) -> tuple[int, str]:
    aliases, tasks = _hits(norm, cap)
    tokens = _rare_hits(norm, cap, rare)
    if not aliases and not tasks and not tokens:
        return 0, ""
    score = 10 * len(aliases) + 8 * len(tasks) + 10 * len(tokens)
    score += _purpose_overlap(norm, cap.purpose)
    return score, ", ".join([*aliases, *tasks, *tokens])


def _rare_tokens(catalog: CapabilityCatalog) -> dict[str, str]:
    owners: dict[str, str] = {}
    shared: set[str] = set()
    for cap in catalog.capabilities.values():
        if not cap.executable:
            continue
        for token in _surface_tokens(cap):
            _claim_token(owners, shared, token, cap.id)
    for token in shared:
        owners.pop(token, None)
    return owners


def _claim_token(owners: dict[str, str], shared: set[str], token: str, cap_id: str) -> None:
    if len(token) < 5 or token in _STOP:
        return
    previous = owners.get(token)
    if previous is None:
        owners[token] = cap_id
        return
    if previous != cap_id:
        shared.add(token)


def _surface_tokens(cap: Capability) -> set[str]:
    found: set[str] = set()
    for alias in cap.aliases:
        found.update(_normalize(alias).split())
    for task in cap.tasks:
        found.update(_normalize(task.replace("_", " ")).split())
    return found


def _rare_hits(norm: str, cap: Capability, rare: Mapping[str, str]) -> list[str]:
    covered = _covered_tokens(norm, cap)
    hits: list[str] = []
    for token in norm.split():
        if token in covered or rare.get(token) != cap.id:
            continue
        hits.append(token)
    return hits


def _covered_tokens(norm: str, cap: Capability) -> set[str]:
    covered: set[str] = set()
    for alias in cap.aliases:
        phrase = _normalize(alias)
        if _phrase_in(norm, phrase):
            covered.update(phrase.split())
    for task in cap.tasks:
        phrase = _normalize(task.replace("_", " "))
        if _phrase_in(norm, phrase):
            covered.update(phrase.split())
    return covered


def _hits(norm: str, cap: Capability) -> tuple[list[str], list[str]]:
    aliases = [alias for alias in cap.aliases if _phrase_in(norm, _normalize(alias))]
    tasks = [task for task in cap.tasks if _phrase_in(norm, _normalize(task.replace("_", " ")))]
    return aliases, tasks


def _purpose_overlap(norm_goal: str, purpose: str) -> int:
    goal_tokens = set(norm_goal.split())
    count = 0
    seen: set[str] = set()
    for token in _normalize(purpose).split():
        if not _purpose_token(token, seen, goal_tokens):
            continue
        seen.add(token)
        count += 1
    return min(count, 4)


def _purpose_token(token: str, seen: set[str], goal_tokens: set[str]) -> bool:
    if len(token) < 6 or token in _STOP or token in seen:
        return False
    return token in goal_tokens


def _phrase_in(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(_fold_token(part) for part in cleaned.split())


def _fold_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _user_data(goal: str, datasets: Sequence[DatasetRef]) -> bool:
    if len(datasets) > 0:
        return True
    return _mentions_user_data(goal)


def _mentions_user_data(goal: str) -> bool:
    folded = f" {_normalize(goal)} "
    if " my data " in folded or " my forecast " in folded or " this dataset " in folded:
        return True
    if " my " in folded and " forecast " in folded:
        return True
    lowered = goal.lower()
    if any(re.search(rf"\b{word}\b", lowered) for word in _USER_WORDS):
        return True
    return _PATH_RE.search(goal) is not None


def _dataset_bound(cap: Capability) -> bool:
    if not cap.requires_dataset or cap.evidence_class == "SYNTHETIC":
        return False
    labels = {label.upper() for label in cap.data_labels}
    return bool(labels - {"SYNTHETIC"})


def _synthetic_only(cap: Capability) -> bool:
    if cap.evidence_class == "SYNTHETIC":
        return True
    labels = {label.upper() for label in cap.data_labels}
    return bool(labels) and labels <= {"SYNTHETIC"}


def _wants_synthetic(goal: str) -> bool:
    norm = _normalize(goal)
    return any(_phrase_in(norm, phrase) for phrase in _SYNTHETIC_REQUESTS)


def _append_same_task_synthetic(
    steps: list[PlanStep],
    other: Sequence[Ranked],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
    datasets: Sequence[DatasetRef],
) -> list[Exclusion]:
    terms = _bound_hit_terms(steps, catalog, goal)
    rows = [row for row in other if _synthetic_only(row[1]) and _covers_terms(row[1], terms)]
    seen = {step.capability_id for step in steps}
    problems: list[Exclusion] = []
    _absorb_ranked(rows, steps, seen, problems, catalog, roots, max_steps, goal, datasets)
    return problems


def _bound_hit_terms(steps: Sequence[PlanStep], catalog: CapabilityCatalog, goal: str) -> set[str]:
    found: set[str] = set()
    for step in steps:
        cap = catalog.get(step.capability_id)
        if cap is None:
            continue
        found.update(_step_hit_terms(cap, step, goal))
    return found


def _step_hit_terms(cap: Capability, step: PlanStep, goal: str) -> set[str]:
    found: set[str] = set()
    for spec in cap.args:
        if spec.bind != "goal" or spec.name not in step.args:
            continue
        choice = str(step.args[spec.name])
        found.update(_hit_terms(goal, spec.goal_terms.get(choice, ())))
    return found


def _covers_terms(cap: Capability, terms: set[str]) -> bool:
    if not terms:
        return False
    blob = _capability_blob(cap)
    return any(_phrase_in(blob, _normalize(term)) for term in terms)


def _capability_blob(cap: Capability) -> str:
    tasks = [task.replace("_", " ") for task in cap.tasks]
    return _normalize(" ".join((cap.purpose, *cap.aliases, *tasks)))


def _prebound(
    cap: Capability, goal: str, datasets: Sequence[DatasetRef], roots: Roots
) -> tuple[dict[str, Any], Exclusion | None]:
    if not _dataset_bound(cap) or not _has_bind(cap):
        return {}, None
    if not datasets:
        return _missing_dataset(cap, goal)
    provided, problem = _bind_declared(cap, goal, datasets)
    if problem is not None:
        return {}, problem
    provided.update(_bind_output_dir(cap, roots))
    return provided, None


def _missing_dataset(cap: Capability, goal: str) -> tuple[dict[str, Any], Exclusion | None]:
    if _mentions_user_data(goal) and _needs_attached_dataset(cap):
        return {}, Exclusion(cap.id, "needs_dataset", _ATTACH_DATASET)
    return {}, None


def _needs_attached_dataset(cap: Capability) -> bool:
    return any(spec.bind == "dataset" and spec.required for spec in cap.args)


def _has_bind(cap: Capability) -> bool:
    return any(bool(spec.bind) for spec in cap.args)


def _bind_declared(
    cap: Capability, goal: str, datasets: Sequence[DatasetRef]
) -> tuple[dict[str, Any], Exclusion | None]:
    provided: dict[str, Any] = {}
    for spec in _bind_specs(cap):
        problem = _bind_one(spec, goal, datasets, cap.id, provided)
        if problem is not None:
            return {}, problem
    return provided, None


def _bind_specs(cap: Capability) -> list[ArgSpec]:
    specs = [spec for spec in cap.args if spec.bind]
    specs.sort(key=lambda spec: (spec.bind != "dataset", spec.name))
    return specs


def _bind_one(
    spec: ArgSpec,
    goal: str,
    datasets: Sequence[DatasetRef],
    cap_id: str,
    provided: dict[str, Any],
) -> Exclusion | None:
    if spec.bind == "dataset":
        return _bind_dataset_arg(spec, goal, datasets, cap_id, provided)
    if spec.bind == "goal":
        return _bind_goal_arg(spec, goal, cap_id, provided)
    return None


def _bind_dataset_arg(
    spec: ArgSpec,
    goal: str,
    datasets: Sequence[DatasetRef],
    cap_id: str,
    provided: dict[str, Any],
) -> Exclusion | None:
    chosen, problem = _choose_dataset(goal, datasets, cap_id)
    if chosen is None:
        return problem
    provided[spec.name] = chosen.path
    return None


def _bind_goal_arg(
    spec: ArgSpec, goal: str, cap_id: str, provided: dict[str, Any]
) -> Exclusion | None:
    choice, problem = _choose_goal(goal, spec, cap_id)
    if choice is None:
        return problem
    provided[spec.name] = choice
    return None


def _choose_dataset(
    goal: str, datasets: Sequence[DatasetRef], cap_id: str
) -> tuple[DatasetRef | None, Exclusion | None]:
    if len(datasets) == 1:
        return datasets[0], None
    named = [item for item in datasets if _goal_names_file(goal, item.path)]
    if len(named) == 1:
        return named[0], None
    return None, Exclusion(cap_id, "needs_input", _dataset_question(datasets))


def _goal_names_file(goal: str, path: str) -> bool:
    name = Path(path).name
    if not name:
        return False
    pattern = rf"(?<![\w.-]){re.escape(name)}(?![\w.-])"
    return re.search(pattern, goal, flags=re.IGNORECASE) is not None


def _dataset_question(datasets: Sequence[DatasetRef]) -> str:
    labels = _dataset_labels(datasets)
    if len(labels) == 1:
        return f"Which dataset: {labels[0]}?"
    if len(labels) == 2:
        return f"Which dataset: {labels[0]} or {labels[1]}?"
    head = ", ".join(labels[:-1])
    return f"Which dataset: {head}, or {labels[-1]}?"


def _dataset_labels(datasets: Sequence[DatasetRef]) -> list[str]:
    names = [Path(item.path).name or item.path for item in datasets]
    if len(set(names)) == len(names):
        return names
    return [item.path for item in datasets]


def _choose_goal(goal: str, spec: ArgSpec, cap_id: str) -> tuple[str | None, Exclusion | None]:
    matched = _matched_choices(goal, spec)
    if len(matched) == 1:
        return matched[0], None
    return None, Exclusion(cap_id, "needs_input", _goal_question(spec))


def _matched_choices(goal: str, spec: ArgSpec) -> list[str]:
    norm = _normalize(goal)
    return [choice for choice, terms in spec.goal_terms.items() if _terms_hit(norm, terms)]


def _terms_hit(norm: str, terms: Sequence[str]) -> bool:
    return any(_phrase_in(norm, _normalize(term)) for term in terms)


def _goal_question(spec: ArgSpec) -> str:
    if set(spec.goal_terms) == _SCORE_CHOICES:
        return _SCORE_QUESTION
    labels = ", ".join(spec.goal_terms)
    return f"Which {spec.name}: {labels}?"


def _bind_output_dir(cap: Capability, roots: Roots) -> dict[str, str]:
    locations = list(roots.get("output_dir", ()))
    if not locations:
        return {}
    spec = _spec_named(cap, "--out-dir")
    if spec is None or spec.path_role != "output_dir":
        return {}
    return {spec.name: str(locations[0])}


def _spec_named(cap: Capability, name: str) -> ArgSpec | None:
    for spec in cap.args:
        if spec.name == name:
            return spec
    return None


def _compose_rationale(base: str, cap: Capability, args: Mapping[str, Any], goal: str) -> str:
    clause = _binding_clause(cap, args, goal)
    if not clause:
        return base
    return f"{base}. {clause}"


def _binding_clause(cap: Capability, args: Mapping[str, Any], goal: str) -> str:
    parts = [_clause_for(spec, args, goal) for spec in cap.args]
    return ". ".join(part for part in parts if part)


def _clause_for(spec: ArgSpec, args: Mapping[str, Any], goal: str) -> str:
    if spec.name not in args or not spec.bind:
        return ""
    if spec.bind == "dataset":
        return _dataset_clause(args[spec.name])
    if spec.bind == "goal":
        return _goal_clause(spec, args[spec.name], goal)
    return ""


def _dataset_clause(path: object) -> str:
    text = str(path)
    if contains_shell_metacharacters(text):
        return "bound dataset"
    return f"bound dataset {text}"


def _goal_clause(spec: ArgSpec, choice: object, goal: str) -> str:
    hits = _hit_terms(goal, spec.goal_terms.get(str(choice), ()))
    why = ", ".join(hits) if hits else str(choice)
    return f"inferred {choice} because the goal matched {why}"


def _hit_terms(goal: str, terms: Sequence[str]) -> list[str]:
    norm = _normalize(goal)
    return [term for term in terms if _phrase_in(norm, _normalize(term))]


def _status_exclusion(cap: Capability) -> Exclusion:
    if cap.qualification_status == "retired":
        code = "retired"
    elif cap.qualification_status != "qualified":
        code = "unqualified"
    else:
        code = "not_executable"
    return Exclusion(cap.id, code, cap.executable_reason or code)


def _status_exclusions(plan: ExecutionPlan) -> list[Exclusion]:
    return [
        item
        for item in plan.exclusions
        if item.code in {"retired", "unqualified", "not_executable"}
    ]


def _suggested_args(cap: Capability, goal: str, roots: Roots) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    tokens = set(_normalize(goal).split())
    norm = _normalize(goal)
    for spec in cap.args:
        if spec.has_default or not spec.required:
            continue
        value = _suggest_required(spec, cap, tokens, norm, roots)
        if value is not None:
            suggestions[spec.name] = value
    return suggestions


def _suggest_required(spec: Any, cap: Capability, tokens: set[str], norm: str, roots: Roots) -> Any:
    if spec.type == "path":
        return _suggest_path(spec, cap, roots)
    if spec.choices:
        return _suggest_choices(spec, tokens, norm)
    return None


def _suggest_path(spec: Any, cap: Capability, roots: Roots) -> str | None:
    if spec.path_role == "output_dir":
        return _first_root(roots, "output_dir")
    if spec.path_role == "input" and not cap.requires_dataset:
        return _first_root(roots, "input")
    return None


def _first_root(roots: Roots, role: str) -> str | None:
    values = roots.get(role) or ()
    for value in values:
        return str(value)
    return None


def _suggest_choices(spec: Any, tokens: set[str], norm: str) -> Any:
    matched = [choice for choice in spec.choices if _choice_mentioned(choice, tokens, norm)]
    if not matched:
        return None
    if spec.type == "csv":
        return matched
    return matched[0]


def _choice_mentioned(choice: str, tokens: set[str], norm: str) -> bool:
    folded = _normalize(choice.replace("_", " "))
    if not folded:
        return False
    return folded in tokens or _phrase_in(norm, folded)


def _estimate(cap: Capability) -> float:
    if cap.max_runtime_s > 0:
        return float(cap.max_runtime_s)
    return float(cap.timeout_s)


def _new_step(cap: Capability, args: Mapping[str, Any], rationale: str) -> PlanStep:
    frozen = MappingProxyType(dict(args))
    checks = tuple(MappingProxyType(dict(check)) for check in cap.post_checks)
    return PlanStep(
        capability_id=cap.id,
        args=frozen,
        rationale=rationale,
        depends_on=cap.dependencies,
        evidence_class=cap.evidence_class,
        consequential=cap.consequential,
        est_runtime_s=_estimate(cap),
        step_key=make_step_key(cap.id, frozen),
        scores=cap.scores,
        post_checks=checks,
        max_output_chars=cap.max_output_chars,
    )


def _safe_text(text: str, fallback: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > 400 or contains_shell_metacharacters(cleaned):
        return fallback
    try:
        validate_fx1_output(cleaned)
    except Fx1HonestyError:
        return fallback
    return cleaned


def _prompt(goal: str, catalog: CapabilityCatalog) -> str:
    lines = [_prompt_line(cap) for cap in catalog.executable()]
    listing = "\n".join(lines)
    return (
        "Select qualified dipcatcher capabilities for the goal. "
        "Reply with one JSON object only: "
        '{"steps":[{"capability":"<id>","args":{},"why":"..."}],"rationale":"..."}. '
        "Use only the ids listed. Do not invent flags.\n"
        f"Available:\n{listing}\n\nGoal: {goal}"
    )


def _prompt_line(cap: Capability) -> str:
    names = ", ".join(spec.name for spec in cap.args) or "(none)"
    return f"- {cap.id}: {cap.purpose} | args: {names}"


def _parse_model_payload(raw: str) -> dict[str, Any]:
    try:
        value = _first_object(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("malformed JSON") from exc
    steps = value.get("steps")
    if not isinstance(steps, list):
        raise ValueError("steps must be a list")
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    return {"rationale": rationale, "steps": steps}


def _first_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    value, _end = json.JSONDecoder().raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


def _steps_from_model(
    entries: Sequence[Any],
    catalog: CapabilityCatalog,
    roots: Roots,
    max_steps: int,
    goal: str,
) -> tuple[list[PlanStep], list[Exclusion]]:
    chosen: list[PlanStep] = []
    seen_keys: set[str] = set()
    exclusions: list[Exclusion] = []
    for entry in entries:
        chain, problem = _model_step(entry, catalog, roots, goal)
        if problem is not None:
            exclusions.append(problem)
            continue
        _absorb(chain, chosen, seen_keys, max_steps, exclusions)
    return chosen, exclusions


def _model_step(
    entry: Any, catalog: CapabilityCatalog, roots: Roots, goal: str
) -> tuple[list[PlanStep], Exclusion | None]:
    if not isinstance(entry, Mapping):
        return [], Exclusion(None, "invalid_args", "step is not an object")
    cap_id = entry.get("capability")
    if not isinstance(cap_id, str) or not cap_id or contains_shell_metacharacters(cap_id):
        shown = cap_id if isinstance(cap_id, str) else None
        return [], Exclusion(shown, "unknown_id", "unknown capability id")
    cap = catalog.get(cap_id)
    if cap is None:
        return [], Exclusion(cap_id, "unknown_id", "unknown capability id")
    if not cap.executable:
        return [], _status_exclusion(cap)
    problem = _check_model_fields(cap_id, entry.get("args", {}), entry.get("why", ""))
    if problem is not None:
        return [], problem
    why = entry.get("why", "")
    args = entry.get("args", {})
    return _model_chain(cap, args, why, catalog, roots, goal)


def _check_model_fields(cap_id: str, args: Any, why: Any) -> Exclusion | None:
    if not isinstance(args, Mapping):
        return Exclusion(cap_id, "invalid_args", "args must be an object")
    if not isinstance(why, str):
        return Exclusion(cap_id, "invalid_args", "why must be a string")
    return None


def _model_chain(
    cap: Capability,
    args: Mapping[str, Any],
    why: str,
    catalog: CapabilityCatalog,
    roots: Roots,
    goal: str,
) -> tuple[list[PlanStep], Exclusion | None]:
    ordered: list[str] = []
    if not _linearize(cap.id, catalog, {}, ordered):
        return [], Exclusion(
            cap.id, "dependency_missing", f"{cap.id} dependency is missing or not executable"
        )
    deps = [item for item in ordered if item != cap.id]
    steps, problem = _default_steps(deps, catalog, roots, cap.id, goal)
    if problem is not None:
        return [], problem
    # The model never chooses where a run writes: the per-run output root is bound here.
    bound = {**dict(args), **_bind_output_dir(cap, roots)}
    try:
        validated = validate_args(cap, bound, roots=roots)
    except ContractViolation as exc:
        return [], Exclusion(cap.id, "invalid_args", f"{exc.code}:{exc.arg}")
    steps.append(_new_step(cap, validated, _safe_text(why, f"selected {cap.id}")))
    return steps, None


def _absorb(
    chain: Sequence[PlanStep],
    chosen: list[PlanStep],
    seen_keys: set[str],
    max_steps: int,
    exclusions: list[Exclusion],
) -> None:
    fresh = [step for step in chain if step.step_key not in seen_keys]
    if not fresh:
        cap_id = chain[-1].capability_id if chain else None
        exclusions.append(Exclusion(cap_id, "duplicate", "duplicate step_key"))
        return
    if len(chosen) + len(fresh) > max_steps:
        exclusions.append(
            Exclusion(chain[-1].capability_id, "max_steps", f"plan exceeds max_steps={max_steps}")
        )
        return
    for step in fresh:
        chosen.append(step)
        seen_keys.add(step.step_key)


def _model_note(rationale: str) -> str | None:
    safe = _safe_text(rationale, "")
    if not safe:
        return None
    return f"model rationale: {safe}"


def _dep_indexes(
    step: PlanStep,
    index: int,
    indexed: Sequence[tuple[int, PlanStep]],
    catalog: CapabilityCatalog,
) -> set[int]:
    names = set(_required_refs(step, catalog))
    found: set[int] = set()
    for other_index, other in indexed:
        if other_index != index and _ref_matches(other, names):
            found.add(other_index)
    return found


def _ref_matches(step: PlanStep, names: set[str]) -> bool:
    return step.capability_id in names or step.step_key in names


def _required_refs(step: PlanStep, catalog: CapabilityCatalog) -> list[str]:
    refs = list(step.depends_on)
    cap = catalog.get(step.capability_id)
    if cap is None:
        return refs
    for dep in cap.dependencies:
        if dep not in refs:
            refs.append(dep)
    return refs


def _topo(indexed: Sequence[tuple[int, PlanStep]], deps: Mapping[int, set[int]]) -> list[PlanStep]:
    remaining = {index for index, _step in indexed}
    emitted: list[PlanStep] = []
    while remaining:
        ready = sorted(index for index in remaining if deps[index].isdisjoint(remaining))
        if not ready:
            return _append_remaining(indexed, remaining, emitted)
        nxt = ready[0]
        remaining.remove(nxt)
        emitted.append(indexed[nxt][1])
    return emitted


def _append_remaining(
    indexed: Sequence[tuple[int, PlanStep]], remaining: set[int], emitted: list[PlanStep]
) -> list[PlanStep]:
    for index, step in indexed:
        if index in remaining:
            emitted.append(step)
    return emitted


def _count_problems(plan: ExecutionPlan, max_steps: int) -> list[Exclusion]:
    return [
        Exclusion(step.capability_id, "max_steps", f"plan exceeds max_steps={max_steps}")
        for step in plan.steps[max_steps:]
    ]


def _duplicate_problems(plan: ExecutionPlan) -> list[Exclusion]:
    seen: set[str] = set()
    problems: list[Exclusion] = []
    for step in plan.steps:
        if step.step_key in seen:
            problems.append(Exclusion(step.capability_id, "duplicate", "duplicate step_key"))
        seen.add(step.step_key)
    return problems


def _step_problems(
    plan: ExecutionPlan, catalog: CapabilityCatalog, roots: Roots
) -> list[Exclusion]:
    problems: list[Exclusion] = []
    for step in plan.steps:
        problems.extend(_one_step_problems(step, catalog, roots))
    return problems


def _one_step_problems(step: PlanStep, catalog: CapabilityCatalog, roots: Roots) -> list[Exclusion]:
    cap = catalog.get(step.capability_id)
    if cap is None:
        return [Exclusion(step.capability_id, "unknown_id", "unknown capability id")]
    if not cap.executable:
        return [_status_exclusion(cap)]
    problems = _metadata_problems(step, cap)
    problems.extend(_arg_problems(step, cap, roots))
    return problems


def _metadata_problems(step: PlanStep, cap: Capability) -> list[Exclusion]:
    checks = (
        (step.evidence_class != cap.evidence_class, "evidence_class does not match the catalog"),
        (step.consequential != cap.consequential, "consequential does not match the catalog"),
        (step.scores != cap.scores, "scores do not match the catalog"),
        (step.est_runtime_s != _estimate(cap), "est_runtime_s does not match the catalog"),
        (
            step.max_output_chars != cap.max_output_chars,
            "max_output_chars does not match the catalog",
        ),
        (
            step.step_key != make_step_key(step.capability_id, step.args),
            "step_key does not match capability and args",
        ),
        (_post_checks_differ(step, cap), "post_checks do not match the catalog"),
    )
    return [
        Exclusion(step.capability_id, "invalid_args", detail) for failed, detail in checks if failed
    ]


def _post_checks_differ(step: PlanStep, cap: Capability) -> bool:
    got = tuple(str(item.get("kind", "")) for item in step.post_checks)
    expected = tuple(str(item.get("kind", "")) for item in cap.post_checks)
    return got != expected


def _arg_problems(step: PlanStep, cap: Capability, roots: Roots) -> list[Exclusion]:
    try:
        checked = validate_args(cap, dict(step.args), roots=roots)
    except ContractViolation as exc:
        return [Exclusion(step.capability_id, "invalid_args", f"{exc.code}:{exc.arg}")]
    if _signature(checked) != _signature(step.args):
        return [Exclusion(step.capability_id, "invalid_args", "args are not in canonical form")]
    return []


def _budget_problems(plan: ExecutionPlan, budget_s: float | None) -> list[Exclusion]:
    if budget_s is None:
        return []
    total = 0.0
    limit = float(budget_s)
    for index, step in enumerate(plan.steps):
        if total + step.est_runtime_s > limit:
            return [
                Exclusion(item.capability_id, "budget", "estimated runtime exceeds budget")
                for item in plan.steps[index:]
            ]
        total += step.est_runtime_s
    return []


def _order_problems(plan: ExecutionPlan, catalog: CapabilityCatalog) -> list[Exclusion]:
    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    problems: list[Exclusion] = []
    for step in plan.steps:
        if _missing_deps(step, catalog, seen_ids, seen_keys):
            problems.append(
                Exclusion(
                    step.capability_id, "dependency_missing", "prerequisite is not an earlier step"
                )
            )
        seen_ids.add(step.capability_id)
        seen_keys.add(step.step_key)
    return problems


def _missing_deps(
    step: PlanStep, catalog: CapabilityCatalog, seen_ids: set[str], seen_keys: set[str]
) -> bool:
    for ref in _required_refs(step, catalog):
        if ref not in seen_ids and ref not in seen_keys:
            return True
    return False


def _signature(args: Mapping[str, Any]) -> str:
    return json.dumps(_plain(args), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _describe_text(plan: ExecutionPlan) -> str:
    lines = [
        f"Execution plan ({plan.source})",
        f"Goal: {plan.goal}",
        f"Catalog digest: {plan.catalog_digest[:12]}",
        f"Steps: {len(plan.steps)} of {plan.max_steps}",
        _budget_line(plan),
    ]
    if not plan.steps:
        lines.append(_NO_MATCH)
    for index, step in enumerate(plan.steps, start=1):
        lines.extend(_step_block(index, step))
    lines.extend(_exclusion_lines(plan.exclusions))
    lines.extend(_note_lines(plan.notes))
    return "\n".join(lines)


def _budget_line(plan: ExecutionPlan) -> str:
    total = sum(step.est_runtime_s for step in plan.steps)
    if plan.budget_s is None:
        return f"Budget: unbounded; estimated total {total:g}s"
    return f"Budget: {plan.budget_s:g}s; estimated total {total:g}s"


def _step_block(index: int, step: PlanStep) -> list[str]:
    marker = "yes" if step.consequential else "no"
    scores = ", ".join(step.scores) if step.scores else "(none)"
    depends = ", ".join(step.depends_on) if step.depends_on else "(none)"
    checks = ", ".join(str(item.get("kind", "")) for item in step.post_checks) or "(none)"
    return [
        f"{index}. {step.capability_id} [{step.evidence_class}] consequential: {marker}",
        f"Why: {step.rationale}",
        f"Args: {_format_args(step.args)}",
        f"Proper scores: {scores}",
        f"Depends on: {depends}",
        f"Post-checks: {checks}",
        f"Time limit: {step.est_runtime_s:g}s; output limit: {step.max_output_chars} chars",
    ]


def _format_args(args: Mapping[str, Any]) -> str:
    if not args:
        return "(none)"
    return ", ".join(f"{key}={_format_value(value)}" for key, value in args.items())


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _exclusion_lines(exclusions: Sequence[Exclusion]) -> list[str]:
    lines = ["Exclusions:"]
    if not exclusions:
        lines.append("- (none)")
        return lines
    for item in exclusions:
        ident = item.capability_id or "-"
        lines.append(f"- {ident} {item.code}: {item.detail}")
    return lines


def _note_lines(notes: Sequence[str]) -> list[str]:
    lines = ["Notes:"]
    if not notes:
        lines.append("- (none)")
        return lines
    lines.extend(f"- {note}" for note in notes)
    return lines


def _wrap(text: str, width: int) -> str:
    limit = width if width >= 20 else 20
    wrapped: list[str] = []
    for line in text.splitlines():
        wrapped.extend(_wrap_line(line, limit))
    return "\n".join(wrapped)


def _wrap_line(line: str, limit: int) -> list[str]:
    if not line:
        return [""]
    pieces = textwrap.wrap(line, width=limit, break_long_words=False, break_on_hyphens=False)
    return pieces or [line]


def _exclusion_dict(item: Exclusion) -> dict[str, Any]:
    return {"capability_id": item.capability_id, "code": item.code, "detail": item.detail}


def _step_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "args": {key: _plain(value) for key, value in step.args.items()},
        "capability_id": step.capability_id,
        "consequential": step.consequential,
        "depends_on": list(step.depends_on),
        "est_runtime_s": step.est_runtime_s,
        "evidence_class": step.evidence_class,
        "max_output_chars": step.max_output_chars,
        "post_checks": [_plain(dict(item)) for item in step.post_checks],
        "rationale": step.rationale,
        "scores": list(step.scores),
        "step_key": step.step_key,
    }


def _dedupe_exclusions(items: Sequence[Exclusion]) -> list[Exclusion]:
    found: list[Exclusion] = []
    seen: set[tuple[str | None, str, str]] = set()
    for item in items:
        key = (item.capability_id, item.code, item.detail)
        if key in seen:
            continue
        seen.add(key)
        found.append(item)
    return found


def _dedupe_notes(notes: Sequence[str]) -> list[str]:
    found: list[str] = []
    for note in notes:
        if note not in found:
            found.append(note)
    return found


def _check_limits(max_steps: int, budget_s: float | None) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if budget_s is None:
        return
    if isinstance(budget_s, bool) or not isinstance(budget_s, (int, float)):
        raise ValueError("budget_s must be a nonnegative number")
    if not math.isfinite(float(budget_s)) or float(budget_s) < 0:
        raise ValueError("budget_s must be a nonnegative number")
