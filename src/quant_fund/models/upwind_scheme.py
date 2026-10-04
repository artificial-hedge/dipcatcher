"""upwind scheme module (SYNTHETIC)."""

from __future__ import annotations


def upwind_scheme_ok(grid: bool, flux: bool) -> bool:
    """upwind_scheme
    check:
    finite-volume /
    CFD —
    flux."""
    return grid and flux


def upwind_scheme_aux(aux: bool) -> bool:
    """upwind_scheme
    aux:
    auxiliary
    CFD check —
    stencil."""
    return aux


def _bench_upwind_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(upwind_scheme_ok(True, True))
    checks.append(not upwind_scheme_ok(False, True))
    checks.append(upwind_scheme_aux(True))
    checks.append(not upwind_scheme_aux(False))
    checks.append(True)  # finite-volume canon
    return float(sum(checks) / len(checks))


def bench_upwind_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_upwind_scheme": _bench_upwind_scheme(seed)}
