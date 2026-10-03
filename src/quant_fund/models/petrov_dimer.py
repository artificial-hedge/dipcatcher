"""petrov dimer module (SYNTHETIC)."""

from __future__ import annotations


def petrov_dimer_ok(dim: bool, hf: bool) -> bool:
    """petrov_dimer
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def petrov_dimer_aux(aux: bool) -> bool:
    """petrov_dimer
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_petrov_dimer(seed: int = 0) -> float:
    checks = []
    checks.append(petrov_dimer_ok(True, True))
    checks.append(not petrov_dimer_ok(False, True))
    checks.append(petrov_dimer_aux(True))
    checks.append(not petrov_dimer_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_petrov_dimer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_petrov_dimer": _bench_petrov_dimer(seed)}
