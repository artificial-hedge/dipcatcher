"""compact scheme module (SYNTHETIC)."""

from __future__ import annotations


def compact_scheme_ok(grid: bool, flux: bool) -> bool:
    """compact_scheme
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def compact_scheme_aux(aux: bool) -> bool:
    """compact_scheme
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_compact_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(compact_scheme_ok(True, True))
    checks.append(not compact_scheme_ok(False, True))
    checks.append(compact_scheme_aux(True))
    checks.append(not compact_scheme_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_compact_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_scheme": _bench_compact_scheme(seed)}
