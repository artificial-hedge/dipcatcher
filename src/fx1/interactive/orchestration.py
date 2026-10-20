"""Typed execution for catalog-gated ``/superpower`` runs.

Planning (when a catalog is present) uses ``planner.plan_goal`` and
``validate_plan``. Harness steps run only when the catalog marks them
executable and the id is in ``HARNESS_REGISTRY``. Arguments are validated
and turned into an argv list; receipt-writing capabilities always receive
an ``--out-dir`` inside the per-run directory. The job ledger, when
supplied, records one ``superpower`` job and confirms cancellation instead
of leaving it ``CANCELLING``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fx1.harness import HARNESS_REGISTRY, Harness, OwnedRunner
from fx1.honesty import scrub_model_text
from fx1.interactive.approvals import ActionBounds, ApprovalGate
from fx1.interactive.atomicstore import ensure_private_dir
from fx1.interactive.contracts import (
    Capability,
    CapabilityCatalog,
    CatalogCache,
    CatalogUnavailable,
    ContractViolation,
    build_argv,
    validate_args,
)
from fx1.interactive.evidence_ledger import (
    EvidenceKind,
    EvidenceLedger,
    Verdict,
    extract_data_label,
    extract_receipt_paths,
)
from fx1.interactive.jobs import IllegalTransition, JobState, StepRecord
from fx1.interactive.planner import (
    DatasetRef,
    Exclusion,
    ExecutionPlan,
    PlanStep,
    make_step_key,
    order_steps,
    validate_plan,
)
from fx1.interactive.planner import plan_goal as planner_plan_goal
from fx1.interactive.redact import redact_text
from fx1.interactive.run_report import (
    CapabilityView,
    ComposedRun,
    answer_notes,
    compose_run_answer,
    offline_summary,
    synthesis_prompt,
)
from fx1.interactive.runctx import new_run_id, require_id, step_id
from fx1.interactive.superpower import (
    _WEB_TRIGGERS,
    CAPABILITIES,
    KIND_HARNESS,
    CapabilityRun,
    CatalogResult,
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    SuperpowerResult,
    _score_goal,
    parse_family_selection,
)

CONSEQUENTIAL_IDS = frozenset({"train", "optimize", "paper"})
_REGISTERED = frozenset(command.name for command in HARNESS_REGISTRY)
_UNREGISTERED = "not registered in HARNESS_REGISTRY"
ModelFn = Callable[[str], str]
Selector = Callable[[str], dict[str, Any]]


@dataclass
class RunState:
    """Mutable bookkeeping for one ``run_plan`` call."""

    plan: SuperpowerPlan
    ctx: SuperpowerContext
    result: SuperpowerResult
    ledger: EvidenceLedger
    run_id: str
    run_dir: Path | None
    job_id: str | None = None
    pieces: list[tuple[str, str, str]] = field(default_factory=list)
    verified_paths: set[str] = field(default_factory=set)
    artifact_paths: set[str] = field(default_factory=set)
    outcomes: dict[str, str] = field(default_factory=dict)
    cancel_sent: bool = False
    cancel_done: bool = False
    current_step_id: str | None = None
    owned: Harness | None = None
    step_index: int = 0
    active_catalog: CapabilityCatalog | None = None


@dataclass
class _CallResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False


def load_catalog(
    harness: Any,
    *,
    config_dir: Path | None = None,
    ttl_s: float = 300.0,
    clock: Callable[[], float] | None = None,
) -> CatalogResult:
    """Fetch ``capabilities list --json``. Failure returns a redacted reason."""

    def runner(name: str, args: list[str]) -> tuple[int, str]:
        result = harness.run(name, list(args))
        return _result_exit(result), _result_stdout(result)

    try:
        catalog = CatalogCache(runner, config_dir=config_dir, ttl_s=ttl_s, clock=clock).get()
    except CatalogUnavailable as exc:
        return CatalogResult(catalog=None, reason=str(exc))
    except Exception as exc:  # noqa: BLE001 — fail closed, never the keyword planner
        return CatalogResult(catalog=None, reason=_failure_reason(exc))
    return CatalogResult(catalog=catalog, reason=None)


def plan_catalog_unavailable(goal: str, reason: str) -> SuperpowerPlan:
    """Harness steps are not executable when the catalog cannot be loaded."""
    cleaned = _failure_text(reason)
    return SuperpowerPlan(
        goal=goal,
        capabilities=_qualified_capabilities(goal, ()),
        source="catalog-unavailable",
        notes=[f"catalog unavailable: {cleaned}", "No suitable capability"],
        orchestration_exclusions=[
            {"capability_id": "", "code": "not_executable", "detail": cleaned},
        ],
        catalog_unavailable=cleaned,
    )


def plan_with_catalog(
    goal: str,
    catalog: CapabilityCatalog,
    *,
    datasets: Sequence[DatasetRef] = (),
    roots: Mapping[str, Sequence[Path]] | None = None,
    model_fn: ModelFn | None = None,
    max_harness: int = 4,
    family_selector: Selector | None = None,
    budget_s: float | None = None,
) -> SuperpowerPlan:
    """Select executable, registered capabilities and re-validate after refinement.

    ``budget_s`` is the operator's estimated-runtime ceiling. ``None`` leaves
    the catalog plan unbounded, matching a session that has not set one.
    """
    gated = gate_to_registry(catalog)
    root_map: Mapping[str, Sequence[Path]] = {} if roots is None else roots
    planning = gated if family_selector is None else _selector_owns_only(gated)
    execution = planner_plan_goal(
        goal,
        planning,
        max_steps=max_harness,
        budget_s=budget_s,
        datasets=tuple(datasets),
        roots=root_map,
        model_fn=model_fn,
    )
    _revalidate(execution, planning, root_map, max_harness)
    family_ids, family_exclusions, constrained = _bind_research(
        goal, execution, family_selector, gated, root_map
    )
    _revalidate(execution, gated, root_map, max_harness)
    if constrained and not _research_step_ids(execution, gated):
        family_ids = []
    return _wrap_plan(goal, execution, gated, family_ids, family_exclusions, constrained)


def _selector_owns_only(catalog: CapabilityCatalog) -> CapabilityCatalog:
    """Planning view where the family surface, not goal tokens, supplies a required ``--only``.

    Execution binds ``--only`` from the surface and re-validates against the
    original contract, so a step that is never bound is dropped, not run.
    """
    relaxed = {cap_id: _relax_only(cap) for cap_id, cap in catalog.capabilities.items()}
    return CapabilityCatalog(
        capabilities=MappingProxyType(relaxed),
        digest=catalog.digest,
        document=catalog.document,
    )


def _relax_only(cap: Capability) -> Capability:
    if not _is_research(cap.id, cap):
        return cap
    args = tuple(
        replace(spec, required=False) if spec.name == "--only" and spec.required else spec
        for spec in cap.args
    )
    return replace(cap, args=args)


def gate_to_registry(catalog: CapabilityCatalog) -> CapabilityCatalog:
    """Drop executability for ids that are not in ``HARNESS_REGISTRY``."""
    gated = {cap_id: _gate_cap(cap) for cap_id, cap in catalog.capabilities.items()}
    return CapabilityCatalog(
        capabilities=MappingProxyType(gated),
        digest=catalog.digest,
        document=catalog.document,
    )


def execute_superpower(plan: SuperpowerPlan, ctx: SuperpowerContext) -> SuperpowerResult:
    """Run one plan: memory, web, harness, save, then the single answer path."""
    state = _open_run(plan, ctx)
    if _cancelled(ctx):
        _note_cancel(state)
    _run_flash_retrieve(state)
    _run_web(state)
    _run_harness(state)
    _run_flash_save(state)
    _finish_answer(state)
    _close_job(state)
    return state.result


def _is_registered(cap: Capability) -> bool:
    """The harness runs ``argv[0]``, so that name (not the capability id) must be registered."""
    return bool(cap.argv) and cap.argv[0] in _REGISTERED


def _gate_cap(cap: Capability) -> Capability:
    if _is_registered(cap) or not cap.executable:
        return cap
    return replace(cap, executable=False, executable_reason=_UNREGISTERED)


def _revalidate(
    execution: ExecutionPlan,
    catalog: CapabilityCatalog,
    roots: Mapping[str, Sequence[Path]],
    max_steps: int,
) -> None:
    problems = validate_plan(
        execution,
        catalog,
        max_steps=max_steps,
        budget_s=execution.budget_s,
        roots=roots,
    )
    if not problems:
        return
    bad = {item.capability_id for item in problems if item.capability_id}
    if bad:
        execution.steps = [step for step in execution.steps if step.capability_id not in bad]
    execution.exclusions.extend(problems)


def _bind_research(
    goal: str,
    execution: ExecutionPlan,
    selector: Selector | None,
    catalog: CapabilityCatalog,
    roots: Mapping[str, Sequence[Path]],
) -> tuple[list[str], list[dict[str, str]], bool]:
    ids = _research_step_ids(execution, catalog)
    if selector is None or not ids:
        return [], [], False
    payload, error = _call_selector(selector, goal)
    if error is not None:
        _drop_research(execution, ids, error)
        return [], [{"id": ids[0], "reason": error}], True
    return _apply_selection(execution, ids, payload or {}, catalog, roots)


def _call_selector(selector: Selector, goal: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = selector(goal)
    except Exception as exc:  # noqa: BLE001 — a refused surface must not run the scorecard
        return None, f"family surface refused: {exc}"
    if not isinstance(payload, dict):
        return None, "family surface refused: payload was not an object"
    return payload, None


def _apply_selection(
    execution: ExecutionPlan,
    ids: Sequence[str],
    payload: Mapping[str, Any],
    catalog: CapabilityCatalog,
    roots: Mapping[str, Sequence[Path]],
) -> tuple[list[str], list[dict[str, str]], bool]:
    try:
        family_ids, raw_exclusions = parse_family_selection(dict(payload))
    except ValueError as exc:
        reason = f"family surface refused: {exc}"
        _drop_research(execution, ids, reason)
        return [], [{"id": ids[0], "reason": reason}], True
    stored = _string_rows(raw_exclusions)
    if not family_ids:
        _drop_research(execution, ids, _joined_reason(stored) or "no suitable qualified family")
        return [], stored, True
    problems = _write_only(execution, ids, catalog, family_ids, roots)
    stored = [*stored, *({"id": cap_id, "reason": text} for cap_id, text in problems)]
    return family_ids, stored, True


def _write_only(
    execution: ExecutionPlan,
    ids: Sequence[str],
    catalog: CapabilityCatalog,
    family_ids: Sequence[str],
    roots: Mapping[str, Sequence[Path]],
) -> list[tuple[str, str]]:
    """Bind ``--only`` on each research step; refused steps are dropped, never widened."""
    problems: list[tuple[str, str]] = []
    for cap_id in ids:
        problem = _bind_one(execution, catalog, cap_id, family_ids, roots)
        if problem is not None:
            _drop_research(execution, [cap_id], problem)
            problems.append((cap_id, problem))
    return problems


def _bind_one(
    execution: ExecutionPlan,
    catalog: CapabilityCatalog,
    cap_id: str,
    family_ids: Sequence[str],
    roots: Mapping[str, Sequence[Path]],
) -> str | None:
    cap = catalog.get(cap_id)
    spec = None if cap is None else _only_spec(cap)
    if cap is None or spec is None:
        return f"catalog {cap_id} contract has no --only argument; refusing the full scorecard"
    step = _find_step(execution, cap_id)
    if step is None:
        return None
    try:
        updated = _step_with_only(step, cap, spec, family_ids, roots)
    except ContractViolation as exc:
        return f"family --only rejected: {exc.code}:{exc.arg}"
    _replace_step(execution, updated)
    return None


def _step_with_only(
    step: PlanStep,
    cap: Capability,
    spec: Any,
    family_ids: Sequence[str],
    roots: Mapping[str, Sequence[Path]],
) -> PlanStep:
    args = dict(step.args)
    args[spec.name] = list(family_ids) if spec.type == "csv" else ",".join(family_ids)
    validated = validate_args(cap, args, roots=roots)
    frozen = MappingProxyType(dict(validated))
    return replace(step, args=frozen, step_key=make_step_key(cap.id, frozen))


def _research_step_ids(execution: ExecutionPlan, catalog: CapabilityCatalog) -> list[str]:
    """Plan steps that run the ``research`` command, whatever their capability id."""
    return [
        step.capability_id
        for step in execution.steps
        if _is_research(step.capability_id, catalog.get(step.capability_id))
    ]


def _is_research(cap_id: str, cap: Capability | None) -> bool:
    if cap is None:
        return cap_id == "research"
    return cap.argv[:1] == ("research",)


def _drop_research(execution: ExecutionPlan, ids: Sequence[str], detail: str) -> None:
    gone = set(ids)
    execution.steps = [
        step
        for step in execution.steps
        if step.capability_id not in gone and not gone.intersection(step.depends_on)
    ]
    for cap_id in dict.fromkeys(ids):
        execution.exclusions.append(Exclusion(cap_id, "not_executable", detail))


def _wrap_plan(
    goal: str,
    execution: ExecutionPlan,
    catalog: CapabilityCatalog,
    family_ids: list[str],
    family_exclusions: list[dict[str, str]],
    constrained: bool,
) -> SuperpowerPlan:
    exclusions = _exclusion_dicts(execution.exclusions)
    for row in family_exclusions:
        exclusions.append(
            {
                "capability_id": row.get("id", "research"),
                "code": "not_executable",
                "detail": row.get("reason", ""),
            }
        )
    return SuperpowerPlan(
        goal=goal,
        capabilities=_qualified_capabilities(goal, execution.steps),
        source=execution.source,
        families=list(family_ids),
        family_exclusions=family_exclusions,
        research_constrained=constrained,
        orchestration_exclusions=_dedupe_rows(exclusions),
        notes=list(execution.notes),
        typed_steps=tuple(execution.steps),
        catalog_digest=execution.catalog_digest,
        bound_catalog=catalog,
    )


def _qualified_capabilities(goal: str, steps: Sequence[PlanStep]) -> list[PlannedCapability]:
    selected = [
        PlannedCapability(
            "flash_retrieve",
            "memory is cheap and may already hold this research",
        )
    ]
    selected.extend(PlannedCapability(step.capability_id, step.rationale) for step in steps)
    if _score_goal(goal, _WEB_TRIGGERS):
        selected.append(
            PlannedCapability(
                "web_research",
                "heuristic: goal asks for external information; "
                "web findings are not sealed receipts",
            )
        )
    selected.extend(
        (
            PlannedCapability(
                "flash_save",
                "new findings belong in flash context with sources and uncertainty",
            ),
            PlannedCapability(
                "synthesize",
                "compose the final answer and explain the capability choices",
            ),
        )
    )
    return selected


def _open_run(plan: SuperpowerPlan, ctx: SuperpowerContext) -> RunState:
    run_id = _resolve_run_id(ctx)
    ledger = ctx.evidence if ctx.evidence is not None else EvidenceLedger()
    run_dir = _ensure_run_dir(ctx, run_id, bool(plan.typed_steps))
    result = SuperpowerResult(goal=plan.goal, plan=plan, run_id=run_id)
    state = RunState(
        plan=plan,
        ctx=ctx,
        result=result,
        ledger=ledger,
        run_id=run_id,
        run_dir=run_dir,
        active_catalog=_active_catalog(plan, ctx),
    )
    _open_job(state)
    return state


def _open_job(state: RunState) -> None:
    ledger = state.ctx.job_ledger
    if ledger is None:
        return
    record = ledger.create(
        kind="superpower",
        goal=state.plan.goal,
        session_id=_resolve_session(state.ctx),
        run_id=state.run_id,
        budget=_budget_payload(state.ctx),
    )
    state.job_id = record.job_id
    state.result.job_id = record.job_id
    if _cancelled(state.ctx):
        _note_cancel(state)
        return
    ledger.transition(record.job_id, JobState.RUNNING)


def _run_harness(state: RunState) -> None:
    if state.plan.catalog_unavailable:
        _skip_listed_harness(state, state.plan.catalog_unavailable)
        return
    if state.plan.typed_steps:
        _run_typed(state)
        return
    _run_legacy(state)


def _run_typed(state: RunState) -> None:
    catalog = state.active_catalog
    if catalog is None:
        _skip_listed_harness(state, "catalog unavailable at execution")
        return
    for step in order_steps(state.plan.typed_steps, catalog):
        _run_typed_step(state, step)


def _run_typed_step(state: RunState, step: PlanStep) -> None:
    reason = _typed_block(state, step)
    if reason is not None:
        run = _skip_capability(state, step.capability_id, reason)
        state.outcomes[step.capability_id] = run.status
        return
    run = _execute_typed(state, step)
    state.outcomes[step.capability_id] = run.status


def _typed_block(state: RunState, step: PlanStep) -> str | None:
    dependency = _dependency_block(state, step)
    if dependency is not None:
        return dependency
    if _stop_for_cancel(state):
        return "cancelled before execution"
    blocked = _budget_block(state.ctx)
    if blocked:
        return blocked
    return _eligibility_block(state, step)


def _eligibility_block(state: RunState, step: PlanStep) -> str | None:
    cap = None if state.active_catalog is None else state.active_catalog.get(step.capability_id)
    if cap is None:
        return "missing from catalog"
    if not _is_registered(cap):
        return _UNREGISTERED
    if not cap.executable:
        return cap.executable_reason or "capability is not executable"
    refused = _research_refusal(state.plan, step.capability_id, cap)
    if refused:
        return refused
    return _family_cap(state, step.capability_id, cap)


def _execute_typed(state: RunState, step: PlanStep) -> CapabilityRun:
    catalog = state.active_catalog
    cap = None if catalog is None else catalog.get(step.capability_id)
    if cap is None:
        return _skip_capability(state, step.capability_id, "missing from catalog")
    try:
        argv = _argv_for(state, step, cap)
    except ContractViolation as exc:
        return _fail_capability(state, step.capability_id, f"args rejected: {exc.code}:{exc.arg}")
    denied = _approval_block(
        state,
        step.capability_id,
        _step_consequential(state, step),
        argv,
        f"{state.run_id}:{step.step_key}",
    )
    if denied is not None:
        return denied
    return _attempt_loop(state, step.capability_id, cap, argv, step)


def _attempt_loop(
    state: RunState,
    cap_id: str,
    cap: Capability,
    argv: list[str],
    step: PlanStep,
) -> CapabilityRun:
    first = _attempt(state, cap_id, cap, argv, f"{state.run_id}:{step.step_key}")
    if first.status != "failed" or _registry_stopping() or not _prepare_retry(state):
        return first
    return _attempt(state, cap_id, cap, argv, f"{state.run_id}:{step.step_key}")


def _registry_stopping() -> bool:
    """True while a terminal stop has latched the process-wide registry."""
    from fx1.interactive.procs import REGISTRY

    return REGISTRY.stopping


def _attempt(
    state: RunState,
    cap_id: str,
    cap: Capability | None,
    argv: list[str],
    idem_key: str,
) -> CapabilityRun:
    begun, blocked = _begin_step(state, cap_id, argv, idem_key)
    if blocked:
        return _skip_capability(state, cap_id, blocked)
    if begun is not None and begun.replayed:
        return _append_run(state, cap_id, cap_id, "ran", "idempotent replay of a succeeded step")
    return _finish_attempt(state, cap_id, cap, argv, begun)


def _finish_attempt(
    state: RunState,
    cap_id: str,
    cap: Capability | None,
    argv: list[str],
    begun: StepRecord | None,
) -> CapabilityRun:
    name, extra = _split_call(cap, argv, cap_id)
    raw = _invoke(state, name, extra, _step_token(state, begun))
    status = _status_of(raw)
    if status == "cancelled":
        _note_cancel(state)
    detail = _detail_of(raw, status)
    paths = _capture_output(state, cap, _result_stdout(raw), _step_token(state, begun))
    _finish_begun(state, begun, status, raw, detail, paths)
    _push_piece(state, "harness", cap_id, _result_stdout(raw))
    return _append_run(state, cap_id, cap_id, status, detail, paths)


def _run_legacy(state: RunState) -> None:
    for cap_id in dict.fromkeys(state.plan.ids()):
        meta = CAPABILITIES.get(cap_id)
        if not meta or meta.get("kind") != KIND_HARNESS:
            continue
        _run_legacy_one(state, cap_id, meta)


def _run_legacy_one(state: RunState, cap_id: str, meta: Mapping[str, Any]) -> None:
    reason = _legacy_block(state, cap_id)
    if reason is not None:
        _skip_capability(state, cap_id, reason)
        return
    extra = _legacy_extra(state.plan, cap_id)
    argv = [cap_id, *(extra or [])]
    idem_key = f"{state.run_id}:{cap_id}"
    if _approval_block(state, cap_id, bool(meta.get("consequential")), argv, idem_key) is not None:
        return
    _attempt(state, cap_id, None, argv, idem_key)


def _legacy_block(state: RunState, cap_id: str) -> str | None:
    if _stop_for_cancel(state):
        return "cancelled before execution"
    blocked = _budget_block(state.ctx)
    if blocked:
        return blocked
    return _research_refusal(state.plan, cap_id)


def _legacy_extra(plan: SuperpowerPlan, cap_id: str) -> list[str] | None:
    if cap_id == "research" and plan.families:
        return ["--only", ",".join(plan.families)]
    return None


def _skip_listed_harness(state: RunState, reason: str) -> None:
    for cap_id in dict.fromkeys(state.plan.ids()):
        meta = CAPABILITIES.get(cap_id)
        if not meta or meta.get("kind") != KIND_HARNESS:
            continue
        _skip_capability(state, cap_id, reason)


def _argv_for(state: RunState, step: PlanStep, cap: Capability) -> list[str]:
    args = dict(step.args)
    out = _receipt_dir(state, cap)
    flag = _output_flag(cap)
    if out is not None and flag is not None:
        args[flag] = str(out)
    roots = _exec_roots(state, out)
    validated = validate_args(cap, args, roots=roots)
    argv = build_argv(cap, validated, roots=roots)
    if out is not None:
        _require_pinned(out, state.run_dir)
    if out is not None and flag is None:
        return [*argv, "--out-dir", str(out)]
    return argv


def _require_pinned(path: Path, root: Path | None) -> None:
    if root is None or _escaped_output(path, root):
        raise ContractViolation(
            "path_escape", "--out-dir", "output directory escaped the run directory"
        )


def _escaped_output(path: Path, root: Path) -> bool:
    if _component_is_link(root, path):
        return True
    try:
        resolved = path.resolve()
        base = root.resolve()
    except OSError:
        return True
    return resolved != base and not resolved.is_relative_to(base)


def _component_is_link(root: Path, path: Path) -> bool:
    if root.is_symlink() or path.is_symlink():
        return True
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            return True
        current = current / part
        if current.is_symlink():
            return True
    return False


def _receipt_dir(state: RunState, cap: Capability) -> Path | None:
    if state.run_dir is None or not _writes_receipts(cap):
        return None
    path = state.run_dir / "out" / _safe_name(cap.id)
    _prepare_dir(path)
    return path


def _capture_output(
    state: RunState,
    cap: Capability | None,
    stdout: str,
    step_label: str | None,
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for token in _receipt_tokens(stdout, cap):
        _add_receipt(state, token, step_label, seen, paths)
    for token in extract_receipt_paths(stdout, marker="json="):
        _add_partial(state, token, stdout, step_label, seen, paths)
    return paths


def _add_receipt(
    state: RunState,
    token: str,
    step_label: str | None,
    seen: set[str],
    paths: list[str],
) -> None:
    if token in seen:
        return
    seen.add(token)
    item = state.ledger.add_receipt(token, roots=_verify_roots(state), step=step_label)
    _remember(state, item.ref, token, verified=item.verdict is Verdict.VERIFIED)
    paths.append(token)


def _add_partial(
    state: RunState,
    token: str,
    stdout: str,
    step_label: str | None,
    seen: set[str],
    paths: list[str],
) -> None:
    if token in seen:
        return
    seen.add(token)
    item = state.ledger.add_artifact(
        token,
        roots=_verify_roots(state),
        step=step_label,
        detail="partial family record — not a sealed research receipt; verified=false",
        data_label=extract_data_label(stdout),
    )
    _remember(state, item.ref, token, verified=False)
    paths.append(token)


def _receipt_tokens(stdout: str, cap: Capability | None) -> list[str]:
    markers = ["receipt="]
    if cap is not None:
        markers.extend(_check_markers(cap))
    found: list[str] = []
    for marker in markers:
        if not marker.startswith("receipt"):
            continue
        for token in extract_receipt_paths(stdout, marker=marker):
            if token not in found:
                found.append(token)
    return found


def _check_markers(cap: Capability) -> list[str]:
    found: list[str] = []
    own = cap.receipt_stdout_marker
    if own:
        found.append(own)
    for check in cap.post_checks:
        source = str(check.get("from") or "")
        if source in {"", "receipt_stdout_marker"}:
            continue
        found.append(source if source.endswith("=") else f"{source}=")
    return found


def _run_flash_retrieve(state: RunState) -> None:
    if not _selected(state, "flash_retrieve") or state.ctx.flash_store is None:
        return
    try:
        _retrieve_flash(state)
    except Exception as exc:  # noqa: BLE001 — memory must not block the run
        _fail_named(state, "flash_retrieve", "flash context (retrieve)", exc)


def _retrieve_flash(state: RunState) -> None:
    from fx1.flash.retrieve import context_block, release_private, retrieve

    store = state.ctx.flash_store
    if store is None:
        return
    hits = release_private(retrieve(store, state.plan.goal, k=4), state.ctx.approvals)
    if not hits:
        _append_run(
            state,
            "flash_retrieve",
            "flash context (retrieve)",
            "ran",
            "no stored context scored above threshold for this goal",
        )
        return
    _record_hits(state, store, hits, context_block(hits))


def _record_hits(state: RunState, store: Any, hits: Sequence[Any], block: str) -> None:
    for hit in hits:
        store.mark_used(hit.entry.id)
        state.ledger.add_flash(str(hit.entry.id), verified_flag=False)
    _push_piece(state, "flash", "flash-context", block)
    _append_run(
        state,
        "flash_retrieve",
        "flash context (retrieve)",
        "ran",
        f"{len(hits)} relevant memory entries retrieved",
    )


def _run_web(state: RunState) -> None:
    if not _selected(state, "web_research") or state.ctx.researcher is None:
        return
    if _stop_for_cancel(state):
        return
    blocked = _budget_block(state.ctx)
    if blocked:
        _skip_capability(state, "web_research", blocked, name="deep web research")
        return
    _execute_web(state)


def _execute_web(state: RunState) -> None:
    researcher = state.ctx.researcher
    if researcher is None:
        return
    try:
        report = researcher.run(
            state.plan.goal,
            budget=state.ctx.research_budget,
            cancel_event=state.ctx.cancel_event,
            on_progress=lambda progress: _on_web(state, progress),
        )
    except Exception as exc:  # noqa: BLE001
        _fail_named(state, "web_research", "deep web research", exc)
        return
    _record_web(state, report)


def _record_web(state: RunState, report: Any) -> None:
    markdown = report.to_markdown()
    _push_piece(state, "web", "web-research", markdown)
    urls = _web_urls(report)
    for url in urls:
        state.ledger.add_web(url)
    detail = (
        f"{len([source for source in report.sources if source.ok])} sources; "
        f"{len(report.groups)} claim topics"
    )
    if report.cancelled:
        detail = f"{detail} (cancelled)"
    _append_run(state, "web_research", "deep web research", "ran", detail, urls)


def _run_flash_save(state: RunState) -> None:
    if not _selected(state, "flash_save") or not state.ctx.save_to_flash:
        return
    if state.ctx.flash_store is None:
        return
    try:
        _save_flash(state)
    except Exception as exc:  # noqa: BLE001
        _fail_named(state, "flash_save", "flash context (save)", exc)


def _save_flash(state: RunState) -> None:
    store = state.ctx.flash_store
    if store is None:
        return
    selected = list(dict.fromkeys(state.plan.ids()))
    goal = redact_text(state.plan.goal)
    entry = store.add(
        text=redact_text(f"/superpower run on: {goal} — capabilities: {', '.join(selected)}"),
        task=goal,
        tags=["superpower"],
        uncertainty="medium",
    )
    state.ledger.add_flash(str(entry.id), verified_flag=False)
    _append_run(
        state,
        "flash_save",
        "flash context (save)",
        "ran",
        f"saved run summary as flash entry {entry.id[:8]}",
        [entry.id],
    )


def _finish_answer(state: RunState) -> None:
    notes = answer_notes(state.plan.notes, state.plan.orchestration_exclusions)
    if "synthesize" not in state.plan.ids():
        _store_evidence(state, _ledger_status(state.ledger))
        return
    if state.ctx.model_fn is None:
        _finish_offline(state, notes)
        return
    _finish_model(state, notes)


def _finish_offline(state: RunState, notes: Sequence[str]) -> None:
    draft = offline_summary(state.plan.goal, _views(state.result.runs), state.result.errors)
    composed = _compose(state, draft, notes)
    _apply_composed(state, composed, offline=True)


def _finish_model(state: RunState, notes: Sequence[str]) -> None:
    model = state.ctx.model_fn
    if model is None:
        return
    prompt = synthesis_prompt(state.plan.goal, _views(state.result.runs), state.pieces)
    try:
        draft = model(prompt)
    except Exception as exc:  # noqa: BLE001
        state.result.errors.append(f"synthesize: {exc}")
        _append_run(state, "synthesize", "synthesis", "failed", redact_text(str(exc)))
        _store_evidence(state, _ledger_status(state.ledger))
        return
    if not isinstance(draft, str):
        _append_run(state, "synthesize", "synthesis", "failed", "model reply was not text")
        _store_evidence(state, _ledger_status(state.ledger))
        return
    _apply_composed(state, _compose(state, draft, notes), offline=False)


def _compose(state: RunState, draft: str, notes: Sequence[str]) -> ComposedRun:
    return compose_run_answer(
        draft,
        state.ledger,
        notes=notes,
        verified_paths=set(state.verified_paths),
        artifact_paths=set(state.artifact_paths),
    )


def _apply_composed(state: RunState, composed: ComposedRun, *, offline: bool) -> None:
    state.result.summary = redact_text(composed.summary)
    state.result.honesty_violation = composed.honesty_violation
    status = "withheld" if composed.honesty_violation else composed.status
    _store_evidence(state, status)
    if composed.honesty_violation:
        _append_run(state, "synthesize", "synthesis", "failed", "honesty gate rejected output")
        return
    detail = _synthesis_detail(offline)
    _append_run(state, "synthesize", "synthesis", "ran", detail)


def _close_job(state: RunState) -> None:
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None:
        return
    if state.cancel_sent or _any_status(state, "cancelled"):
        _confirm_cancel(state)
        return
    try:
        ledger.transition(state.job_id, _job_target(state))
    except IllegalTransition as exc:
        state.result.errors.append(f"job transition: {exc}")


def _confirm_cancel(state: RunState) -> None:
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None or state.cancel_done:
        return
    if not state.cancel_sent:
        _note_cancel(state)
    if state.cancel_done:
        return
    try:
        ledger.confirm_terminated(state.job_id)
    except IllegalTransition as exc:
        state.result.errors.append(f"cancel confirmation pending: {exc}")
        return
    state.cancel_done = True


def _note_cancel(state: RunState) -> None:
    state.cancel_sent = True
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None or state.cancel_done:
        return
    try:
        record = ledger.request_cancel(state.job_id)
    except IllegalTransition as exc:
        state.result.errors.append(f"cancel request: {exc}")
        return
    if record.state is JobState.CANCELLED:
        state.cancel_done = True


def _begin_step(
    state: RunState,
    name: str,
    argv: Sequence[str],
    idem_key: str,
) -> tuple[StepRecord | None, str | None]:
    state.step_index += 1
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None:
        return None, None
    try:
        record = ledger.begin_step(
            state.job_id,
            name=name,
            argv=list(argv),
            idempotent=True,
            idempotency_key=idem_key,
            step_index=state.step_index - 1,
        )
    except IllegalTransition as exc:
        return None, str(exc)
    return record, None


def _finish_begun(
    state: RunState,
    begun: StepRecord | None,
    status: str,
    raw: Any,
    detail: str,
    paths: Sequence[str],
) -> None:
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None or begun is None or begun.replayed:
        return
    try:
        ledger.finish_step(
            state.job_id,
            begun.step_id,
            state=_step_state(status),
            exit_code=_result_exit(raw),
            receipt_paths=list(paths),
            detail=detail,
        )
    except IllegalTransition as exc:
        state.result.errors.append(f"finish step: {exc}")


def _invoke(state: RunState, name: str, extra: list[str] | None, step_token: str | None) -> Any:
    harness = _resolve_harness(state)
    _announce(state.ctx, name)
    state.current_step_id = step_token
    try:
        return _call_harness(harness, name, extra, state, step_token)
    except Exception as exc:  # noqa: BLE001
        return _CallResult(exit_code=1, stderr=redact_text(str(exc)))
    finally:
        state.current_step_id = None


def _call_harness(
    harness: Any,
    name: str,
    extra: list[str] | None,
    state: RunState,
    step_token: str | None,
) -> Any:
    payload = extra or None
    try:
        return harness.run(
            name,
            payload,
            cancel_event=state.ctx.cancel_event,
            env_extra=_child_env(state, step_token),
            step=step_token,
        )
    except TypeError:
        return harness.run(name, payload)


def _resolve_harness(state: RunState) -> Any:
    if state.ctx.harness is not None:
        return state.ctx.harness
    if state.owned is None:
        state.owned = Harness(OwnedRunner(emit=_process_emit(state)))
    return state.owned


def _process_emit(state: RunState) -> Callable[[str, Mapping[str, Any]], None]:
    def emit(kind: str, fields: Mapping[str, Any]) -> None:
        _telemetry(state, kind, _spawn_fields(fields))
        if kind == "proc.spawn":
            _attach_spawn(state, fields)

    return emit


def _attach_spawn(state: RunState, fields: Mapping[str, Any]) -> None:
    ledger = state.ctx.job_ledger
    pid = fields.get("pid")
    if ledger is None or state.job_id is None or state.current_step_id is None:
        return
    if isinstance(pid, bool) or not isinstance(pid, int):
        return
    pgid = fields.get("pgid")
    gid = pgid if isinstance(pgid, int) and not isinstance(pgid, bool) else None
    try:
        ledger.attach_process(state.job_id, state.current_step_id, pid=pid, pgid=gid)
    except (IllegalTransition, ValueError):
        return


def _approval_block(
    state: RunState,
    cap_id: str,
    consequential: bool,
    argv: Sequence[str],
    idem_key: str,
) -> CapabilityRun | None:
    """Deny record when this argv is not already approved. Replay does not ask."""
    if _replay_ready(state, idem_key):
        return None
    if _consequential_ok(state, cap_id, consequential, argv):
        return None
    return state.result.runs[-1]


def _replay_ready(state: RunState, idem_key: str) -> bool:
    if not idem_key:
        return False
    ledger = state.ctx.job_ledger
    if ledger is None or state.job_id is None:
        return False
    found = ledger.find_step_by_key(idem_key)
    if found is None:
        return False
    return found[1].state is JobState.SUCCEEDED


def _consequential_ok(
    state: RunState, cap_id: str, consequential: bool, argv: Sequence[str]
) -> bool:
    if not _needs_approval(cap_id, consequential):
        return True
    gate = state.ctx.approvals or ApprovalGate()
    question = _approval_question(cap_id)
    bounds = ActionBounds(argv=tuple(argv), writes=True)
    if gate.ask(question, kind="consequential", key=f"harness:{cap_id}", bounds=bounds):
        return True
    _append_run(
        state,
        cap_id,
        cap_id,
        "approval-denied",
        "operator did not approve this consequential command",
    )
    return False


def _needs_approval(cap_id: str, consequential: bool) -> bool:
    if cap_id in CONSEQUENTIAL_IDS:
        return True
    return consequential


def _approval_question(cap_id: str) -> str:
    meta = CAPABILITIES.get(cap_id, {})
    description = str(meta.get("description", ""))
    return f"run harness command {cap_id!r} (consequential: {description})?"


def _prepare_retry(state: RunState) -> bool:
    budget = state.ctx.budget
    charge = getattr(budget, "charge_retry", None)
    if not callable(charge):
        return False
    try:
        charge()
    except Exception:  # noqa: BLE001 — the budget object reports its own refusal
        return False
    return _budget_block(state.ctx) is None


def _budget_block(ctx: SuperpowerContext) -> str | None:
    """Charge one step. None means the step may run."""
    budget = ctx.budget
    if budget is None:
        return None
    try:
        if bool(budget.expired()):
            return "budget expired before this step"
        budget.charge_step()
    except Exception as exc:  # noqa: BLE001 — a budget object reports its own refusal
        return str(exc)
    return None


def _dependency_block(state: RunState, step: PlanStep) -> str | None:
    for dep in step.depends_on:
        status = state.outcomes.get(dep)
        if status is None or status == "ran":
            continue
        return f"prerequisite {dep} {status}"
    return None


def _research_refusal(
    plan: SuperpowerPlan, cap_id: str, cap: Capability | None = None
) -> str | None:
    if not _is_research(cap_id, cap) or not plan.research_constrained or plan.families:
        return None
    if not plan.family_exclusions:
        return "no suitable qualified family"
    return _joined_reason(plan.family_exclusions)


def _family_cap(state: RunState, cap_id: str, cap: Capability | None) -> str | None:
    if not _is_research(cap_id, cap):
        return None
    limit = getattr(state.ctx.budget, "max_families", None)
    if isinstance(limit, bool) or not isinstance(limit, int):
        return None
    if len(state.plan.families) > limit:
        return "family budget exceeded"
    return None


def _stop_for_cancel(state: RunState) -> bool:
    if not _cancelled(state.ctx):
        return False
    _note_cancel(state)
    return True


def _cancelled(ctx: SuperpowerContext) -> bool:
    return ctx.cancel_event is not None and ctx.cancel_event.is_set()


def _job_target(state: RunState) -> JobState:
    if _any_status(state, "timed-out"):
        return JobState.TIMED_OUT
    if _any_status(state, "failed"):
        return JobState.FAILED
    return JobState.SUCCEEDED


def _step_state(status: str) -> JobState:
    if status == "cancelled":
        return JobState.CANCELLED
    if status == "timed-out":
        return JobState.TIMED_OUT
    if status == "ran":
        return JobState.SUCCEEDED
    return JobState.FAILED


def _status_of(raw: Any) -> str:
    if bool(getattr(raw, "cancelled", False)):
        return "cancelled"
    if bool(getattr(raw, "timed_out", False)):
        return "timed-out"
    if _result_exit(raw) == 0:
        return "ran"
    return "failed"


def _detail_of(raw: Any, status: str) -> str:
    stdout = _result_stdout(raw)
    stderr = _result_stderr(raw)
    detail = _stdout_detail(stdout, stderr)
    return _with_flags(detail, raw, status)


def _stdout_detail(stdout: str, stderr: str) -> str:
    labels = _data_labels(stdout)
    prose, receipts = _peel_receipts(_tail(stdout, stderr))
    head = _labeled(labels, prose)[:500]
    if not receipts:
        return head
    block = "\n".join(receipts)
    if not head:
        return block
    return f"{head}\n{block}"


def _labeled(labels: str, prose: str) -> str:
    if labels and prose:
        return f"{labels}; {prose}"
    return labels or prose


def _peel_receipts(text: str) -> tuple[str, list[str]]:
    prose: list[str] = []
    receipts: list[str] = []
    for line in text.splitlines():
        if "receipt=" in line:
            receipts.append(line)
        elif line.strip():
            prose.append(line.strip())
    return " ".join(prose), receipts


def _with_flags(detail: str, raw: Any, status: str) -> str:
    text = detail
    if bool(getattr(raw, "truncated", False)):
        text = f"{text} (truncated)".strip()
    if status == "timed-out":
        text = f"timed-out; {text}".strip()
    if status == "cancelled":
        text = f"cancelled; {text}".strip()
    return text


def _data_labels(stdout: str) -> str:
    labels = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("DATA_LABEL=") or line.strip() == "SYNTHETIC"
    ]
    return "; ".join(labels[:4])


def _active_catalog(plan: SuperpowerPlan, ctx: SuperpowerContext) -> CapabilityCatalog | None:
    if plan.bound_catalog is not None:
        return plan.bound_catalog
    catalog = ctx.catalog
    if isinstance(catalog, CapabilityCatalog):
        return gate_to_registry(catalog)
    return None


def _ensure_run_dir(ctx: SuperpowerContext, run_id: str, needed: bool) -> Path | None:
    if ctx.run_dir is not None:
        return _prepare_dir(ctx.run_dir)
    if not needed:
        return None
    return _prepare_dir(_config_dir() / "runs" / require_id(run_id, "r"))


def _prepare_dir(path: Path) -> Path:
    ensure_private_dir(path)
    _chmod_dir_nofollow(path)
    return path


def _chmod_dir_nofollow(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def _config_dir() -> Path:
    override = os.environ.get("FX1_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".fx1"


def _resolve_run_id(ctx: SuperpowerContext) -> str:
    if not ctx.run_id:
        return new_run_id()
    return require_id(ctx.run_id, "r")


def _resolve_session(ctx: SuperpowerContext) -> str | None:
    if not ctx.session_id:
        return None
    return require_id(ctx.session_id, "s")


def _budget_payload(ctx: SuperpowerContext) -> dict[str, Any] | None:
    public = getattr(ctx.budget, "as_public", None)
    if not callable(public):
        return None
    payload = public()
    if isinstance(payload, dict):
        return payload
    return None


def _child_env(state: RunState, step_token: str | None) -> dict[str, str]:
    env: dict[str, str] = {"DIPCATCHER_RUN_ID": state.run_id}
    if state.ctx.session_id:
        env["DIPCATCHER_SESSION_ID"] = state.ctx.session_id
    if step_token:
        env["DIPCATCHER_STEP"] = step_token
    return env


def _step_token(state: RunState, begun: StepRecord | None) -> str | None:
    if begun is not None:
        return begun.step_id
    try:
        return step_id(state.run_id, max(state.step_index - 1, 0))
    except ValueError:
        return None


def _exec_roots(state: RunState, out: Path | None) -> dict[str, list[Path]]:
    roots = state.ctx.roots or {}
    merged: dict[str, list[Path]] = {key: list(values) for key, values in roots.items()}
    if state.run_dir is not None:
        merged.setdefault("output_dir", []).append(state.run_dir)
    if out is not None:
        merged.setdefault("output_dir", []).append(out)
    return merged


def _verify_roots(state: RunState) -> list[Path]:
    if state.run_dir is None:
        return []
    return [state.run_dir]


def _output_flag(cap: Capability) -> str | None:
    for spec in cap.args:
        if spec.path_role == "output_dir" or spec.name in {"--out-dir", "--receipts-dir"}:
            return spec.name
    return None


def _writes_receipts(cap: Capability) -> bool:
    if "receipts" in cap.writes:
        return True
    return cap.receipt_stdout_marker.startswith("receipt")


def _split_call(cap: Capability | None, argv: list[str], cap_id: str) -> tuple[str, list[str]]:
    if not argv:
        return cap_id, []
    head = 1 if cap is None else len(cap.argv)
    if head <= 0 or head > len(argv):
        return argv[0], argv[1:]
    return argv[0], argv[head:]


def _only_spec(cap: Capability) -> Any:
    for spec in cap.args:
        if spec.name in {"--only", "only"}:
            return spec
    return None


def _find_step(execution: ExecutionPlan, cap_id: str) -> PlanStep | None:
    for step in execution.steps:
        if step.capability_id == cap_id:
            return step
    return None


def _replace_step(execution: ExecutionPlan, updated: PlanStep) -> None:
    execution.steps = [
        updated if step.capability_id == updated.capability_id else step for step in execution.steps
    ]


def _step_consequential(state: RunState, step: PlanStep) -> bool:
    if step.consequential:
        return True
    cap = None if state.active_catalog is None else state.active_catalog.get(step.capability_id)
    return cap is not None and cap.consequential


def _selected(state: RunState, cap_id: str) -> bool:
    return cap_id in state.plan.ids()


def _any_status(state: RunState, status: str) -> bool:
    return any(run.status == status for run in state.result.runs)


def _append_run(
    state: RunState,
    cap_id: str,
    name: str,
    status: str,
    detail: str,
    artifacts: Sequence[str] | None = None,
) -> CapabilityRun:
    run = CapabilityRun(cap_id, name, status, scrub_model_text(detail), list(artifacts or []))
    state.result.runs.append(run)
    _telemetry(state, "superpower.step", {"capability": cap_id, "status": status})
    return run


def _skip_capability(
    state: RunState, cap_id: str, reason: str, *, name: str | None = None
) -> CapabilityRun:
    return _append_run(state, cap_id, name or cap_id, "skipped", reason)


def _fail_capability(state: RunState, cap_id: str, detail: str) -> CapabilityRun:
    state.result.errors.append(f"{cap_id}: {detail}")
    return _append_run(state, cap_id, cap_id, "failed", detail)


def _fail_named(state: RunState, cap_id: str, name: str, exc: BaseException) -> None:
    detail = redact_text(str(exc))
    state.result.errors.append(f"{cap_id}: {detail}")
    _append_run(state, cap_id, name, "failed", detail)


def _store_evidence(state: RunState, status: str) -> None:
    state.result.evidence = state.ledger.to_dict()
    state.result.status = status
    state.result.run_id = state.run_id


def _ledger_status(ledger: EvidenceLedger) -> str:
    items = ledger.items()
    if not items:
        return "no-evidence"
    verified = all(
        item.kind is EvidenceKind.RECEIPT and item.verdict is Verdict.VERIFIED for item in items
    )
    if verified:
        return "verified"
    if ledger.has_verified():
        return "partial"
    return "unverified"


def _views(runs: Sequence[CapabilityRun]) -> list[CapabilityView]:
    return [CapabilityView(run.id, run.name, run.status, run.detail) for run in runs]


def _synthesis_detail(offline: bool) -> str:
    if offline:
        return "no model endpoint available — offline summary of collected results"
    return "honesty gate passed"


def _push_piece(state: RunState, kind: str, source: str, text: str) -> None:
    limit = state.ctx.max_stdout_chars
    body = text if limit < 0 else text[-limit:]
    state.pieces.append((kind, source, body))
    state.artifact_paths.update(_cited_json(body))


def _cited_json(text: str) -> set[str]:
    return set(re.findall(r"[\w./\\:-]+\.json", text))


def _remember(state: RunState, ref: str, token: str, *, verified: bool) -> None:
    state.artifact_paths.add(ref)
    state.artifact_paths.add(token)
    if verified:
        state.verified_paths.add(ref)
        state.verified_paths.add(token)


def _announce(ctx: SuperpowerContext, cap_id: str) -> None:
    if ctx.on_event is not None:
        ctx.on_event("harness", f"running {cap_id} (registered command)")


def _on_web(state: RunState, progress: Any) -> None:
    if state.ctx.on_event is None:
        return
    state.ctx.on_event(f"web:{progress.event}", scrub_model_text(str(progress.detail)[:200]))


def _telemetry(state: RunState, kind: str, fields: Mapping[str, Any]) -> None:
    if state.ctx.telemetry_emit is None:
        return
    state.ctx.telemetry_emit(kind, dict(fields))


def _spawn_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {key: fields[key] for key in ("pid", "pgid", "step") if key in fields}


def _web_urls(report: Any) -> list[str]:
    urls: list[str] = []
    for source in list(report.sources)[:6]:
        url = getattr(source, "url", "")
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def _exclusion_dicts(items: Sequence[Exclusion]) -> list[dict[str, str]]:
    return [
        {
            "capability_id": item.capability_id or "",
            "code": item.code,
            "detail": item.detail,
        }
        for item in items
    ]


def _string_rows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [_family_row(item) for item in items]


def _family_row(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "id": str(item.get("id", "")),
        "reason": str(item.get("reason", "")),
        "code": str(item.get("code", "")),
    }


def _joined_reason(items: Sequence[Mapping[str, str]]) -> str:
    parts = [
        f"{item.get('id', '?')}: {item.get('reason', '')}" for item in items if item.get("reason")
    ]
    return "; ".join(parts)


def _dedupe_rows(items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, str]] = []
    for item in items:
        row = {
            "capability_id": str(item.get("capability_id", item.get("id", ""))),
            "code": str(item.get("code", "")),
            "detail": str(item.get("detail", item.get("reason", ""))),
        }
        key = (row["capability_id"], row["code"], row["detail"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _result_exit(result: Any) -> int:
    code = getattr(result, "exit_code", 1)
    if isinstance(code, bool) or not isinstance(code, int):
        return 1
    return code


def _result_stdout(result: Any) -> str:
    text = getattr(result, "stdout", "")
    return text if isinstance(text, str) else ""


def _result_stderr(result: Any) -> str:
    text = getattr(result, "stderr", "")
    return text if isinstance(text, str) else ""


def _tail(stdout: str, stderr: str) -> str:
    text = stdout if stdout.strip() else stderr
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    receipts = [_clip_receipt(line) for line in lines if "receipt=" in line]
    plain = [line for line in lines if "receipt=" not in line]
    body = plain[-1][:160] if plain else ""
    if not receipts:
        return body
    block = "\n".join(receipts)
    if not body:
        return block
    return f"{body}\n{block}"


def _clip_receipt(line: str) -> str:
    """Keep a ``receipt=`` path intact, with a hard cap against unbounded stdout."""
    if len(line) <= 1024:
        return line
    return line[:1021] + "..."


def _safe_name(cap_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cap_id)
    return cleaned or "step"


def _failure_reason(exc: BaseException) -> str:
    return _failure_text(f"{exc.__class__.__name__}: {exc}")


def _failure_text(text: str) -> str:
    cleaned = " ".join(redact_text(text).split())
    if len(cleaned) > 180:
        return cleaned[:177] + "..."
    return cleaned or "capabilities catalog unavailable"
