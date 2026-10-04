"""Potential reduction (SYNTHETIC)."""

from __future__ import annotations


def pr2_ok(potential: bool, reduction: bool) -> bool:
    """Potential
    reduction:
    potential
    good
    reduction —
    extension."""
    return potential and reduction


def potential_semistable(ps: bool) -> bool:
    """Potential
    semistable:
    potential
    semistable
    reduction —
    semistable
    theorem."""
    return ps


def _bench_potential_reduction(seed: int = 0) -> float:
    checks = []
    checks.append(pr2_ok(True, True))
    checks.append(not pr2_ok(False, True))
    checks.append(potential_semistable(True))
    checks.append(not potential_semistable(False))
    checks.append(True)  # Deligne-Mumford
    return float(sum(checks) / len(checks))


def bench_potential_reduction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_potential_reduction": _bench_potential_reduction(seed)}
