"""Qualified /superpower wiring: catalog, run directory, evidence, and jobs.

The concierge stays a controller. This module builds the roots, the per-run
directory, and the operator text for evidence and cancellation. A missing
catalog is fail-closed; datasets are not invented.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fx1.interactive.chat_tools import ensure_output_roots, policy_roots
from fx1.interactive.evidence_ledger import EvidenceLedger
from fx1.interactive.jobs import JobState, ReconcileReport
from fx1.interactive.runctx import new_run_id, require_id
from fx1.interactive.superpower import (
    SuperpowerContext,
    SuperpowerPlan,
    SuperpowerResult,
    load_catalog,
    plan_goal,
)

Selector = Callable[[str], dict[str, Any]]
TelemetryEmit = Callable[[str, Mapping[str, Any]], None]


@dataclass
class PreparedRun:
    """One qualified superpower attempt. ``datasets`` stays empty."""

    catalog: Any
    roots: dict[str, tuple[Path, ...]]
    run_id: str
    run_dir: Path
    evidence: EvidenceLedger
    unavailable: str | None


@dataclass
class RunSeams:
    """Injectable neighbors for :func:`build_context`. No secrets."""

    model_fn: Callable[[str], str] | None
    harness: Any
    approvals: Any
    researcher: Any
    research_budget: Any
    flash_store: Any
    on_event: Callable[[str, str], None] | None
    cancel_event: Any
    budget: Any
    job_ledger: Any
    session_id: str
    telemetry_emit: TelemetryEmit | None


def qualified_roots(
    cwd: Path | None = None, config_dir: Path | None = None
) -> dict[str, tuple[Path, ...]]:
    """Map :func:`policy_roots` into catalog path roles.

    ``output_dir`` covers receipts and the config runs directory so a typed
    step can land inside either. Output directories are created here.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    raw = policy_roots(root, config_dir)
    ensure_output_roots(raw)
    output = raw["output"]
    runs = raw["runs"]
    # Catalog defaults such as ``configs/research.yaml`` are relative to the
    # working directory. ``<cwd>/configs`` remains a root for paths already
    # inside that directory.
    return {
        "config": (root, raw["config"]),
        "input": (raw["input"],),
        "output": (output,),
        "runs": (runs,),
        "output_dir": (output, runs),
    }


def prepare_run_dir(runs_root: Path, run_id: str) -> Path:
    """Create ``runs_root/<run_id>`` at mode 0700. The id is validated first."""
    path = runs_root / require_id(run_id, "r")
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)
    return path


def prepare_qualified_run(harness: Any, *, config_dir: Path | None = None) -> PreparedRun:
    """Load the catalog and allocate one evidence ledger plus a run directory."""
    loaded = load_catalog(harness, config_dir=config_dir)
    roots = qualified_roots(config_dir=config_dir)
    run_id = new_run_id()
    unavailable = None if loaded.catalog is not None else (loaded.reason or "catalog unavailable")
    return PreparedRun(
        catalog=loaded.catalog,
        roots=roots,
        run_id=run_id,
        run_dir=prepare_run_dir(roots["runs"][0], run_id),
        evidence=EvidenceLedger(),
        unavailable=unavailable,
    )


def make_plan(
    goal: str,
    prepared: PreparedRun,
    *,
    model_fn: Callable[[str], str] | None,
    family_selector: Selector | None,
    max_harness: int,
) -> SuperpowerPlan:
    """Plan *goal*. A missing catalog is passed as ``catalog_unavailable``."""
    return plan_goal(
        goal,
        model_fn=model_fn,
        family_selector=family_selector,
        max_harness=max_harness,
        catalog=prepared.catalog,
        datasets=(),
        roots=prepared.roots,
        catalog_unavailable=prepared.unavailable,
    )


def build_context(prepared: PreparedRun, seams: RunSeams) -> SuperpowerContext:
    """Context for :func:`run_plan`. Datasets stay an empty tuple."""
    return SuperpowerContext(
        model_fn=seams.model_fn,
        harness=seams.harness,
        approvals=seams.approvals,
        researcher=seams.researcher,
        research_budget=seams.research_budget,
        flash_store=seams.flash_store,
        on_event=seams.on_event,
        cancel_event=seams.cancel_event,
        budget=seams.budget,
        catalog=prepared.catalog,
        datasets=(),
        roots=prepared.roots,
        job_ledger=seams.job_ledger,
        evidence=prepared.evidence,
        run_dir=prepared.run_dir,
        session_id=seams.session_id,
        run_id=prepared.run_id,
        telemetry_emit=seams.telemetry_emit,
    )


def operator_status(result: SuperpowerResult) -> str:
    """Map a run onto ``withheld``, ``cancelled``, or ``ok``."""
    if result.status == "withheld" or result.honesty_violation:
        return "withheld"
    if result.status == "cancelled" or _any_cancelled(result):
        return "cancelled"
    return "ok"


def evidence_status_line(result: SuperpowerResult) -> str:
    """One line. This is not a second evidence block."""
    return f"evidence status: {result.status}"


def evidence_report(result: Any, ledger: EvidenceLedger | None, focus: str) -> str:
    """Ledger listing, one item, or the legacy artifact paths."""
    active = _active_ledger(result, ledger)
    if active is None:
        return _artifact_fallback(result)
    if focus:
        return _item_detail(active, focus)
    return active.render_block()


def reconcile_lines(report: ReconcileReport) -> list[str]:
    """Honest startup lines. Nothing here was restarted."""
    lines = [f"job {job_id} was interrupted and was not restarted" for job_id in report.interrupted]
    lines.extend(_process_lines(report.orphan_running, "orphan process"))
    lines.extend(_process_lines(report.unverifiable, "unverifiable process"))
    lines.extend(f"corrupt job file {name}; not restarted" for name in report.corrupt)
    return lines


def cancel_lines(label: str, state: str | None) -> list[str]:
    """``cancel requested``, plus ``cancellation pending`` until confirmed."""
    lines = [f"[{label}] cancel requested"]
    if state != JobState.CANCELLED.value:
        lines.append("cancellation pending")
    return lines


def _any_cancelled(result: SuperpowerResult) -> bool:
    return any(run.status == "cancelled" for run in result.runs)


def _active_ledger(result: Any, ledger: EvidenceLedger | None) -> EvidenceLedger | None:
    if ledger is not None and ledger.items():
        return ledger
    payload = getattr(result, "evidence", None)
    if isinstance(payload, dict) and payload.get("items"):
        return EvidenceLedger.from_dict(payload)
    return None


def _artifact_fallback(result: Any) -> str:
    runs = getattr(result, "runs", None) or []
    paths = [artifact for run in runs for artifact in getattr(run, "artifacts", []) if artifact]
    if not paths:
        return "no receipt paths on the last job"
    return "\n".join(str(path) for path in paths)


def _item_detail(ledger: EvidenceLedger, focus: str) -> str:
    item = _find_item(ledger, focus)
    if item is None:
        return f"no evidence item {focus}"
    lines = [
        f"{item.item_id} {item.kind.value.lower()} {item.verdict.value} {item.data_label}",
        f"ref {item.ref}",
    ]
    if item.sha256:
        lines.append(f"sha256 {item.sha256}")
    if item.schema:
        lines.append(f"schema {item.schema}")
    if item.step:
        lines.append(f"step {item.step}")
    if item.detail:
        lines.append(item.detail)
    return "\n".join(lines)


def _find_item(ledger: EvidenceLedger, focus: str) -> Any:
    token = focus.strip()
    for item in ledger.items():
        if item.item_id == token:
            return item
    return None


def _process_lines(notes: Sequence[Any], label: str) -> list[str]:
    return [
        f"{label} pid {note.pid} step {note.step_id} job {note.job_id}; not restarted"
        for note in notes
    ]
