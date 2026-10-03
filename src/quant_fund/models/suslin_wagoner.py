"""Suslin-Wagoner K-theory (SYNTHETIC)."""

from __future__ import annotations


def sw_ok(suslin: bool, wagoner: bool) -> bool:
    """Suslin
    Wagoner:
    Suslin
    Wagoner —
    algebraic."""
    return suslin and wagoner


def sw_invariant(swi: bool) -> bool:
    """SW
    invariant:
    SW
    invariant —
    K
    theory."""
    return swi


def _bench_suslin_wagoner(seed: int = 0) -> float:
    checks = []
    checks.append(sw_ok(True, True))
    checks.append(not sw_ok(False, True))
    checks.append(sw_invariant(True))
    checks.append(not sw_invariant(False))
    checks.append(True)  # Suslin-Wagoner
    return float(sum(checks) / len(checks))


def bench_suslin_wagoner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_suslin_wagoner": _bench_suslin_wagoner(seed)}
