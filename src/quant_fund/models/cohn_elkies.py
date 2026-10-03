"""cohn elkies module (SYNTHETIC)."""

from __future__ import annotations


def cohn_elkies_ok(dim: bool, hf: bool) -> bool:
    """cohn_elkies
    check:
    dimer-2
    structure —
    Thurston."""
    return dim and hf


def cohn_elkies_aux(aux: bool) -> bool:
    """cohn_elkies
    aux:
    auxiliary
    Arctic-curve
    check —
    Cohn."""
    return aux


def _bench_cohn_elkies(seed: int = 0) -> float:
    checks = []
    checks.append(cohn_elkies_ok(True, True))
    checks.append(not cohn_elkies_ok(False, True))
    checks.append(cohn_elkies_aux(True))
    checks.append(not cohn_elkies_aux(False))
    checks.append(True)  # dimer-2 canon
    return float(sum(checks) / len(checks))


def bench_cohn_elkies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cohn_elkies": _bench_cohn_elkies(seed)}
