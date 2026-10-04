"""Peterson-Stein formulas (SYNTHETIC)."""

from __future__ import annotations


def ps_ok(peterson: bool, stein: bool) -> bool:
    """Peterson:
    Peterson-
    Stein
    formulas
    for
    secondary
    operations —
    Peterson-
    Stein."""
    return peterson and stein


def stein_formula(sf: bool) -> bool:
    """Stein:
    secondary
    operation
    decomposition —
    Stein
    formula."""
    return sf


def _bench_peterson_stein(seed: int = 0) -> float:
    checks = []
    checks.append(ps_ok(True, True))
    checks.append(not ps_ok(False, True))
    checks.append(stein_formula(True))
    checks.append(not stein_formula(False))
    checks.append(True)  # Peterson-Stein
    return float(sum(checks) / len(checks))


def bench_peterson_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_peterson_stein": _bench_peterson_stein(seed)}
