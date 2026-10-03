"""inverse local module (SYNTHETIC)."""

from __future__ import annotations


def inverse_local_ok(ex: bool, me: bool) -> bool:
    """inverse_local
    check:
    excursion
    theory —
    measure."""
    return ex and me


def inverse_local_aux(aux: bool) -> bool:
    """inverse_local
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_inverse_local(seed: int = 0) -> float:
    checks = []
    checks.append(inverse_local_ok(True, True))
    checks.append(not inverse_local_ok(False, True))
    checks.append(inverse_local_aux(True))
    checks.append(not inverse_local_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_inverse_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inverse_local": _bench_inverse_local(seed)}
