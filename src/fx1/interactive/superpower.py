"""/superpower — plan, select, coordinate, and explain research capabilities.

The registry is built from ``fx1.harness.HARNESS_REGISTRY`` (the exact
fail-closed set of lab surfaces fx-1 may invoke) plus three front-door
capabilities: web research, flash-context retrieval, and flash save. A
planner maps a goal to the *relevant subset* — deterministic keyword scoring,
optionally refined by a model pass — and the coordinator runs that subset in
dependency order (memory → web → harness → save → synthesis), gating
consequential harness commands behind explicit approval.

Two contracts never bend here:
- harness execution happens only through registered commands (subprocess,
  injectable runner), so the model can never reach an unregistered surface;
- the synthesized answer passes ``fx1.honesty.validate_fx1_output`` — on
  violation the answer is withheld and the gate's reason reported instead.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fx1.harness import HARNESS_REGISTRY, Harness, HarnessRole
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.contracts import CapabilityCatalog
from fx1.interactive.planner import DatasetRef, PlanStep

# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------

KIND_HARNESS = "harness"
KIND_WEB = "web"
KIND_FLASH = "flash"
KIND_REASON = "reason"

# Trigger terms that make a harness command relevant to a goal. Keys are
# capability ids; the catalog is data, not code, so it can be listed and
# audited (``/superpower help``) without executing anything.
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor", "manifest", "readiness", "health check"),
    "verify-research": ("verify", "receipt", "provenance", "audit", "immutable"),
    "validate": ("validate", "promotion gate", "walk-forward", "causal"),
    "monitor": ("monitor", "operational", "runtime"),
    "ingest": ("ingest", "bronze", "silver", "lake"),
    "collect": ("collect", "public feed", "opt-in"),
    "build-features": ("features", "pit", "point-in-time"),
    "build-labels": ("labels", "target", "labeling"),
    "research": (
        "benchmark",
        "proper score",
        "pinball",
        "crps",
        "qlike",
        "conformal",
        "e-values",
        "jackknife",
        "crc",
        "volatility",
        "regime",
        "tail risk",
        "drawdown",
        "liquidity",
        "ranking",
        "alpha",
        "distribution",
        "quantile bandit",
        "hypothesis",
        "family",
        "sweep",
    ),
    "northset": ("order book", "lob", "northset", "uncrossed", "identity"),
    "candle-book": ("candle", "ohlc", "candlestick"),
    "kyle-ofi": ("kyle", "lambda", "ofi", "flow imbalance"),
    "session-book": ("session", "book reconstruction"),
    "vendor-book-map": ("vendor", "book mapping"),
    "book-panel": ("book panel", "depth"),
    "backtest": ("backtest", "fill", "slippage", "participation", "cost model"),
    "forecast": ("forecast", "predict"),
    "kronos-forecast": ("kronos",),
    "report": ("report", "notebook", "write up", "writeup"),
    "tearsheet": ("tearsheet", "analytics export"),
    "train": ("train", "fine-tune", "finetune", "lora", "qlora"),
    "optimize": ("optimize", "hyperparameter", "optuna"),
    "paper": ("paper trading", "shadow run", "simulated broker"),
}

_CONSEQUENTIAL = frozenset({"train", "optimize", "paper"})

_WEB_TRIGGERS = (
    "web",
    "internet",
    "online",
    "search",
    "latest",
    "current",
    "news",
    "recent",
    "look up",
    "what does",
    "literature",
    "sources say",
    "find out",
    "compare publicly",
)

_HARNESS_CAPABILITIES: dict[str, dict[str, Any]] = {
    command.name: {
        "id": command.name,
        "name": command.name,
        "kind": KIND_HARNESS,
        "description": command.description,
        "role": command.role.value,
        "timeout_s": command.timeout_s,
        "consequential": command.name in _CONSEQUENTIAL
        or command.role == HarnessRole.MODEL_TRAINING,
        "cost": "heavy" if command.timeout_s >= 1800 else "moderate",
    }
    for command in HARNESS_REGISTRY
}

CAPABILITIES: dict[str, dict[str, Any]] = {
    **_HARNESS_CAPABILITIES,
    "web_research": {
        "id": "web_research",
        "name": "deep web research",
        "kind": KIND_WEB,
        "description": (
            "Iterative public-web research: search, read sources, follow "
            "references, cross-check claims, report conflicts."
        ),
        "role": "research",
        "timeout_s": 600,
        "consequential": False,
        "cost": "moderate",
    },
    "flash_retrieve": {
        "id": "flash_retrieve",
        "name": "flash context (retrieve)",
        "kind": KIND_FLASH,
        "description": (
            "Retrieve the relevant portions of persistent research memory for the current goal."
        ),
        "role": "research",
        "timeout_s": 5,
        "consequential": False,
        "cost": "cheap",
    },
    "flash_save": {
        "id": "flash_save",
        "name": "flash context (save)",
        "kind": KIND_FLASH,
        "description": "Persist findings with sources and uncertainty to flash context.",
        "role": "research",
        "timeout_s": 5,
        "consequential": False,
        "cost": "cheap",
    },
    "synthesize": {
        "id": "synthesize",
        "name": "synthesis",
        "kind": KIND_REASON,
        "description": (
            "Compose the final answer from capability results, explaining "
            "which capabilities were used and why."
        ),
        "role": "research",
        "timeout_s": 300,
        "consequential": False,
        "cost": "cheap",
    },
}

ModelFn = Callable[[str], str]
Selector = Callable[[str], dict[str, Any]]
TelemetryEmit = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class CatalogResult:
    """A capabilities fetch. ``catalog`` is None when the fetch failed closed."""

    catalog: CapabilityCatalog | None
    reason: str | None = None


def _score_goal(goal: str, terms: tuple[str, ...]) -> int:
    lowered = goal.lower()
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


@dataclass
class PlannedCapability:
    """One selected capability and why it was selected."""

    id: str
    rationale: str


@dataclass
class SuperpowerPlan:
    goal: str
    capabilities: list[PlannedCapability] = field(default_factory=list)
    source: str = "deterministic"
    families: list[str] = field(default_factory=list)
    family_exclusions: list[dict[str, str]] = field(default_factory=list)
    research_constrained: bool = False
    exclusions: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    typed_steps: tuple[PlanStep, ...] = ()
    catalog_digest: str = ""
    catalog_unavailable: str = ""
    bound_catalog: CapabilityCatalog | None = field(default=None, repr=False, compare=False)

    def ids(self) -> list[str]:
        return [cap.id for cap in self.capabilities]

    def describe(self) -> str:
        lines = [f"plan for: {self.goal}", f"planner: {self.source}"]
        _append_capability_lines(lines, self.capabilities)
        _append_family_lines(lines, self)
        _append_plan_notes(lines, self)
        return "\n".join(lines)


def _append_capability_lines(lines: list[str], capabilities: Sequence[PlannedCapability]) -> None:
    for cap in capabilities:
        meta = CAPABILITIES.get(cap.id, {})
        lines.append(
            f"- {cap.id} ({meta.get('kind', '?')}, {meta.get('cost', '?')}): {cap.rationale}"
        )


def _append_family_lines(lines: list[str], plan: SuperpowerPlan) -> None:
    if not plan.research_constrained:
        return
    chosen = ", ".join(plan.families) if plan.families else "none eligible"
    lines.append(f"families: {chosen}")
    for item in plan.family_exclusions:
        lines.append(f"  excluded {item.get('id', '?')}: {item.get('reason', '')}")


def _append_plan_notes(lines: list[str], plan: SuperpowerPlan) -> None:
    for item in plan.exclusions:
        cap = item.get("capability_id") or "-"
        lines.append(f"excluded {cap}: {item.get('code', '')}: {item.get('detail', '')}")
    for note in plan.notes:
        lines.append(f"note: {note}")
    if plan.catalog_unavailable:
        lines.append(f"catalog unavailable: {plan.catalog_unavailable}")


def plan_goal(
    goal: str,
    *,
    capabilities: dict[str, dict[str, Any]] | None = None,
    model_fn: ModelFn | None = None,
    max_harness: int = 4,
    family_selector: Selector | None = None,
    catalog: CapabilityCatalog | None = None,
    datasets: Sequence[DatasetRef] = (),
    roots: Mapping[str, Sequence[Path]] | None = None,
    catalog_unavailable: str | None = None,
    budget_s: float | None = None,
) -> SuperpowerPlan:
    """Select the relevant capability subset for *goal*.

    Without a catalog, the historical keyword planner is unchanged. With a
    catalog, only executable capabilities that are also in ``HARNESS_REGISTRY``
    become harness steps, and the typed planner re-validates after model
    refinement. ``catalog_unavailable`` (and no catalog) fails closed: harness
    steps are not executable. Web research is still planned only when the goal
    asks for external information.
    """
    if max_harness < 0:
        raise ValueError("max_harness must be nonnegative")
    if catalog is None and catalog_unavailable:
        from fx1.interactive.orchestration import plan_catalog_unavailable

        return plan_catalog_unavailable(goal, catalog_unavailable)
    if catalog is not None:
        from fx1.interactive.orchestration import plan_with_catalog

        return plan_with_catalog(
            goal,
            catalog,
            datasets=datasets,
            roots=roots,
            model_fn=model_fn,
            max_harness=max_harness,
            family_selector=family_selector,
            budget_s=budget_s,
        )
    return _plan_keyword(
        goal,
        capabilities=capabilities,
        model_fn=model_fn,
        max_harness=max_harness,
        family_selector=family_selector,
    )


def _plan_keyword(
    goal: str,
    *,
    capabilities: dict[str, dict[str, Any]] | None,
    model_fn: ModelFn | None,
    max_harness: int,
    family_selector: Selector | None,
) -> SuperpowerPlan:
    registry = capabilities or CAPABILITIES
    harness_ids = _matched_harness(goal, registry, max_harness)
    plan = SuperpowerPlan(
        goal=goal,
        capabilities=_keyword_spine(goal, registry, harness_ids),
        source="deterministic",
    )
    return _maybe_refine(plan, goal, registry, model_fn, max_harness, family_selector)


def _matched_harness(
    goal: str, registry: Mapping[str, Mapping[str, Any]], max_harness: int
) -> list[str]:
    scored: list[tuple[int, str]] = []
    for cap_id, meta in registry.items():
        if meta.get("kind") != KIND_HARNESS:
            continue
        score = _score_goal(goal, _TRIGGERS.get(cap_id, ()))
        if score:
            scored.append((score, cap_id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [cap_id for _, cap_id in scored[:max_harness]]


def _keyword_spine(
    goal: str, registry: Mapping[str, Mapping[str, Any]], harness_ids: Sequence[str]
) -> list[PlannedCapability]:
    selected = [
        PlannedCapability("flash_retrieve", "memory is cheap and may already hold this research"),
        *_harness_plans(goal, registry, harness_ids),
    ]
    if _wants_web(goal, harness_ids):
        selected.append(
            PlannedCapability(
                "web_research",
                "goal asks for current/external information, or no lab surface "
                "matched — public web research fills the gap",
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


def _harness_plans(
    goal: str, registry: Mapping[str, Mapping[str, Any]], harness_ids: Sequence[str]
) -> list[PlannedCapability]:
    found: list[PlannedCapability] = []
    lowered = goal.lower()
    for cap_id in harness_ids:
        meta = registry[cap_id]
        terms = _TRIGGERS.get(cap_id, ())
        matched = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)]
        found.append(
            PlannedCapability(
                cap_id,
                f"goal mentions {', '.join(repr(term) for term in matched[:4])}; "
                f"{meta.get('description', '')}",
            )
        )
    return found


def _wants_web(goal: str, harness_ids: Sequence[str]) -> bool:
    if _score_goal(goal, _WEB_TRIGGERS):
        return True
    return not harness_ids


def _maybe_refine(
    plan: SuperpowerPlan,
    goal: str,
    registry: Mapping[str, Mapping[str, Any]],
    model_fn: ModelFn | None,
    max_harness: int,
    family_selector: Selector | None,
) -> SuperpowerPlan:
    if model_fn is not None:
        plan = _model_refine(plan, goal, registry, model_fn, max_harness=max_harness)
    if family_selector is not None and "research" in plan.ids():
        return _attach_families(plan, family_selector)
    return plan


def parse_family_selection(payload: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:
    """Fail closed unless every selected family is an executable required id."""
    if not isinstance(payload, dict) or payload.get("schema") != "dipcatcher.family_surface.v1":
        raise ValueError("family surface schema mismatch")
    ids: list[str] = []
    for row in payload.get("selected") or []:
        if not isinstance(row, dict):
            raise ValueError("selected entries must be objects")
        family_id = row.get("id")
        eligible = row.get("executable") is True and row.get("qualification") == "required"
        if (
            not eligible
            or not isinstance(family_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", family_id) is None
        ):
            raise ValueError(f"ineligible family in selection: {family_id!r}")
        if family_id not in ids:
            ids.append(family_id)
    exclusions = [item for item in payload.get("exclusions") or [] if isinstance(item, dict)]
    return ids, exclusions


def _attach_families(
    plan: SuperpowerPlan, selector: Callable[[str], dict[str, Any]]
) -> SuperpowerPlan:
    """Constrain research to the selector's eligible ids. A failure does not run the catalog."""
    try:
        family_ids, exclusions = parse_family_selection(selector(plan.goal))
    except Exception as exc:  # noqa: BLE001 — fail closed, never fall through to a full catalog run
        plan.research_constrained = True
        plan.families = []
        plan.family_exclusions = [{"id": "research", "reason": f"family surface refused: {exc}"}]
        return plan
    plan.research_constrained = True
    plan.families = family_ids
    plan.family_exclusions = exclusions
    return plan


def _model_refine(
    fallback: SuperpowerPlan,
    goal: str,
    registry: Mapping[str, Mapping[str, Any]],
    model_fn: ModelFn,
    *,
    max_harness: int,
) -> SuperpowerPlan:
    """Validate and bound registered model selections, else keep fallback."""
    listing = "\n".join(
        f"- {cap_id}: {meta.get('description', '')}" for cap_id, meta in sorted(registry.items())
    )
    prompt = (
        "You select dipcatcher research capabilities for a goal. Reply with "
        'JSON only: {"capabilities": ["<id>"...], "rationale": "one '
        f'sentence per id"}}. Select at most {max_harness} harness capabilities; '
        f"do not repeat ids. Available: {listing}\n\nGoal: {goal}"
    )
    try:
        raw = model_fn(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in model reply")
        payload = json.loads(match.group(0))
        ids = payload.get("capabilities")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError("capabilities must be a list of strings")
        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValueError("rationale must be a string")
        valid: list[str] = []
        harness_count = 0
        for cap_id in dict.fromkeys(ids):
            if cap_id not in registry or cap_id in {"flash_retrieve", "synthesize"}:
                continue
            if registry[cap_id].get("kind") == KIND_HARNESS:
                if harness_count >= max_harness:
                    continue
                harness_count += 1
            valid.append(cap_id)
        # Model ordering cannot remove or relocate the mandatory spine.
        valid = ["flash_retrieve", *valid, "synthesize"]
        capabilities = [
            PlannedCapability(
                cap_id,
                f"model rationale: {rationale}" if rationale else "model-selected",
            )
            for cap_id in valid
        ]
        return SuperpowerPlan(goal=goal, capabilities=capabilities, source="model+validated")
    except Exception:  # noqa: BLE001 — model failure preserves the bounded fallback
        return fallback


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------


@dataclass
class CapabilityRun:
    id: str
    name: str
    status: str  # ran | failed | skipped | approval-denied
    detail: str = ""
    artifacts: list[str] = field(default_factory=list)


@dataclass
class SuperpowerResult:
    goal: str
    plan: SuperpowerPlan
    runs: list[CapabilityRun] = field(default_factory=list)
    summary: str = ""
    honesty_violation: str | None = None
    errors: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    status: str = "no-evidence"
    job_id: str | None = None
    run_id: str = ""

    def explain(self) -> str:
        lines = ["capabilities used and why:", ""]
        lines.extend(f"[{run.status}] {run.name}: {run.detail}" for run in self.runs)
        if self.honesty_violation:
            lines.extend(["", f"honesty gate: {self.honesty_violation}"])
        lines.extend(["", f"evidence status: {self.status}"])
        count = _evidence_count(self.evidence)
        if count is not None:
            lines.append(f"evidence items: {count}")
        return "\n".join(lines)


@dataclass
class SuperpowerContext:
    """Everything a run needs — all seams injectable for offline tests."""

    model_fn: ModelFn | None = None
    harness: Harness | None = None
    approvals: ApprovalGate | None = None
    researcher: Any | None = None
    research_budget: Any | None = None
    flash_store: Any | None = None
    on_event: Callable[[str, str], None] | None = None
    cancel_event: threading.Event | None = None
    budget: Any | None = None
    max_stdout_chars: int = 12000
    save_to_flash: bool = True
    catalog: CapabilityCatalog | None = None
    datasets: Sequence[DatasetRef] = ()
    roots: Mapping[str, Sequence[Path]] = field(default_factory=dict)
    job_ledger: Any | None = None
    evidence: Any | None = None
    run_dir: Path | None = None
    session_id: str | None = None
    run_id: str | None = None
    telemetry_emit: TelemetryEmit | None = None


def _evidence_count(evidence: Mapping[str, Any]) -> int | None:
    items = evidence.get("items")
    if isinstance(items, list):
        return len(items)
    return None


def load_catalog(
    harness: Harness,
    *,
    config_dir: Path | None = None,
    ttl_s: float = 300.0,
    clock: Callable[[], float] | None = None,
) -> CatalogResult:
    """Load the capabilities catalog. A failed fetch fails closed."""
    from fx1.interactive.orchestration import load_catalog as _load_catalog

    return _load_catalog(harness, config_dir=config_dir, ttl_s=ttl_s, clock=clock)


def run_plan(plan: SuperpowerPlan, ctx: SuperpowerContext) -> SuperpowerResult:
    """Execute *plan* and compose one evidence-grounded answer."""
    from fx1.interactive.orchestration import execute_superpower

    return execute_superpower(plan, ctx)


def describe_capabilities() -> str:
    lines = ["available /superpower capabilities:", ""]
    for cap_id, meta in sorted(CAPABILITIES.items()):
        flags = []
        if meta.get("consequential"):
            flags.append("approval-required")
        flags.append(str(meta.get("cost", "?")))
        lines.append(
            f"- {cap_id} [{meta.get('kind')}; {', '.join(flags)}]: {meta.get('description', '')}"
        )
    return "\n".join(lines)
