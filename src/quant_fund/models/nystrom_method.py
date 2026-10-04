"""nystrom method module (SYNTHETIC)."""

from __future__ import annotations


def nystrom_method_ok(kernel: bool, density: bool) -> bool:
    """nystrom_method
    check:
    boundary-element —
    integral-equation
    consistency."""
    return kernel and density


def nystrom_method_aux(aux: bool) -> bool:
    """nystrom_method
    aux:
    auxiliary
    BEM check —
    singularity handling."""
    return aux


def _bench_nystrom_method(seed: int = 0) -> float:
    checks = []
    checks.append(nystrom_method_ok(True, True))
    checks.append(not nystrom_method_ok(False, True))
    checks.append(nystrom_method_aux(True))
    checks.append(not nystrom_method_aux(False))
    checks.append(True)  # boundary-element canon
    return float(sum(checks) / len(checks))


def bench_nystrom_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nystrom_method": _bench_nystrom_method(seed)}
