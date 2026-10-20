"""Startup checks and the /diagnostics report.

Failures become warning lines. The report and the zip are redacted by the
diagnostics module; this layer does not add credential text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from fx1.interactive.diagnostics import collect_diagnostics, format_report, write_bundle
from fx1.interactive.jobs import ReconcileReport
from fx1.interactive.superpower_glue import reconcile_lines
from fx1.interactive.telemetry import Telemetry


def startup_lines(
    health: Callable[[], list[str]],
    reconcile: Callable[[], ReconcileReport],
) -> list[str]:
    """Store-health warnings and a reconcile report. Never raises."""
    lines = _health_lines(health)
    lines.extend(_reconcile_lines(reconcile))
    return lines


def render_diagnostics(
    *,
    config_dir: Path,
    session_id: str | None,
    model: str,
    endpoint_rows: Callable[[], Sequence[Mapping[str, str]]],
    job_counts: Callable[[], Mapping[str, int]],
    session_counts: Callable[[], Mapping[str, int]],
    telemetry: Telemetry | None,
) -> tuple[dict[str, Any], str]:
    """Presence-only report text plus the mapping used for a bundle."""
    diag = collect_diagnostics(
        config_dir=config_dir,
        session_id=session_id,
        model=model,
        endpoint_rows=endpoint_rows,
        job_counts=job_counts,
        session_counts=session_counts,
        telemetry=telemetry,
    )
    return diag, format_report(diag).rstrip("\n")


def write_diagnostics_bundle(
    diag: Mapping[str, Any],
    path: Path,
    *,
    telemetry: Telemetry | None,
) -> str:
    """Write the zip or return a refusal. An existing path is not replaced."""
    if path.exists():
        return f"refusing to overwrite {path}"
    files: tuple[Path, ...] = ()
    if telemetry is not None:
        files = tuple(telemetry.log_files())
    try:
        written = write_bundle(diag, path, telemetry_files=files)
    except FileExistsError:
        return f"refusing to overwrite {path}"
    return f"diagnostics bundle {written}"


def telemetry_owner(emit: Any) -> Telemetry | None:
    """The Telemetry instance behind a bound ``emit``, if that is what was passed."""
    owner = getattr(emit, "__self__", None)
    if isinstance(owner, Telemetry):
        return owner
    return None


def _health_lines(health: Callable[[], list[str]]) -> list[str]:
    try:
        return [f"warning: {item}" for item in health()]
    except Exception as exc:  # noqa: BLE001 — startup checks must not raise
        return [f"warning: store health failed ({type(exc).__name__})"]


def _reconcile_lines(reconcile: Callable[[], ReconcileReport]) -> list[str]:
    try:
        return reconcile_lines(reconcile())
    except Exception as exc:  # noqa: BLE001 — a bad ledger is one warning line
        return [f"warning: job ledger reconcile failed ({type(exc).__name__}); not restarted"]
