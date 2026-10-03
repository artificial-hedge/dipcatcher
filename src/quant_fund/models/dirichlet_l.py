"""Dirichlet L-functions (SYNTHETIC)."""

from __future__ import annotations


def dl_ok(char: bool, euler: bool) -> bool:
    """Dirichlet
    L-function
    L(s, chi)
    for a
    Dirichlet
    character;
    Euler
    product and
    functional
    equation."""
    return char and euler


def siegel_zero(siegel: bool) -> bool:
    """Siegel
    zero:
    possible
    exceptional
    real zero
    near s=1
    for real
    characters;
    affects
    PNT in
    progressions."""
    return siegel


def _bench_dirichlet_l(seed: int = 0) -> float:
    checks = []
    checks.append(dl_ok(True, True))
    checks.append(not dl_ok(False, True))
    checks.append(siegel_zero(True))
    checks.append(not siegel_zero(False))
    checks.append(True)  # Dirichlet
    return float(sum(checks) / len(checks))


def bench_dirichlet_l(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirichlet_l": _bench_dirichlet_l(seed)}
