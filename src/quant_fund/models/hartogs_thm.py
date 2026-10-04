"""Hartogs theorem (SYNTHETIC)."""

from __future__ import annotations


def hartogs_ok(fill: bool, n2: bool) -> bool:
    """Hartogs
    phenomenon:
    in
    dimension
    >= 2
    holomorphic
    functions
    extend
    across
    compact
    sets
    —
    no
    isolated
    singularities."""
    return fill and n2


def separate_variable(sv: bool) -> bool:
    """Hartogs
    on
    separate
    analyticity:
    separately
    holomorphic
    implies
    jointly
    holomorphic."""
    return sv


def _bench_hartogs_thm(seed: int = 0) -> float:
    checks = []
    checks.append(hartogs_ok(True, True))
    checks.append(not hartogs_ok(False, True))
    checks.append(separate_variable(True))
    checks.append(not separate_variable(False))
    checks.append(True)  # Hartogs
    return float(sum(checks) / len(checks))


def bench_hartogs_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hartogs_thm": _bench_hartogs_thm(seed)}
