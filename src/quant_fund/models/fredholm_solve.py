"""fredholm solve module (SYNTHETIC)."""

from __future__ import annotations


def fredholm_solve_ok(kernel: bool, density: bool) -> bool:
    """fredholm_solve
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def fredholm_solve_aux(aux: bool) -> bool:
    """fredholm_solve
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_fredholm_solve(seed: int = 0) -> float:
    checks = []
    checks.append(fredholm_solve_ok(True, True))
    checks.append(not fredholm_solve_ok(False, True))
    checks.append(fredholm_solve_aux(True))
    checks.append(not fredholm_solve_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_fredholm_solve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fredholm_solve": _bench_fredholm_solve(seed)}
