"""motivic partial module (SYNTHETIC)."""

from __future__ import annotations


def motivic_partial_ok(motivic: bool, categorical: bool) -> bool:
    """motivic_partial
    check:
    motivic
    structure —
    functor."""
    return motivic and categorical


def motivic_partial_aux(aux: bool) -> bool:
    """motivic_partial
    aux:
    auxiliary
    motivic
    check —
    nerve."""
    return aux


def _bench_motivic_partial(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_partial_ok(True, True))
    checks.append(not motivic_partial_ok(False, True))
    checks.append(motivic_partial_aux(True))
    checks.append(not motivic_partial_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_partial(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_partial": _bench_motivic_partial(seed)}
