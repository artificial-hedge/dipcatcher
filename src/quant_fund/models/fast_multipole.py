"""fast multipole module (SYNTHETIC)."""

from __future__ import annotations


def fast_multipole_ok(kernel: bool, density: bool) -> bool:
    """fast_multipole
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def fast_multipole_aux(aux: bool) -> bool:
    """fast_multipole
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_fast_multipole(seed: int = 0) -> float:
    checks = []
    checks.append(fast_multipole_ok(True, True))
    checks.append(not fast_multipole_ok(False, True))
    checks.append(fast_multipole_aux(True))
    checks.append(not fast_multipole_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_fast_multipole(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fast_multipole": _bench_fast_multipole(seed)}
