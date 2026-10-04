"""Smith resolution project (SYNTHETIC)."""

from __future__ import annotations


def sp_ok(smith: bool, project: bool) -> bool:
    """Smith
    project:
    Smith
    resolution
    project —
    condensed
    resolution."""
    return smith and project


def condensed_resolution(cr: bool) -> bool:
    """Condensed
    resolution:
    condensed
    resolution
    of
    a
    topological
    ring —
    extremally
    disconnected."""
    return cr


def _bench_smith_project(seed: int = 0) -> float:
    checks = []
    checks.append(sp_ok(True, True))
    checks.append(not sp_ok(False, True))
    checks.append(condensed_resolution(True))
    checks.append(not condensed_resolution(False))
    checks.append(True)  # Smith
    return float(sum(checks) / len(checks))


def bench_smith_project(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smith_project": _bench_smith_project(seed)}
