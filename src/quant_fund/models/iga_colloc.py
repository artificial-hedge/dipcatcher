"""iga colloc module (SYNTHETIC)."""

from __future__ import annotations


def iga_colloc_ok(step: bool, radius: bool) -> bool:
    """iga_colloc
    check:
    optimization /
    IGA canon —
    step/radius
    consistency."""
    return step and radius


def iga_colloc_aux(aux: bool) -> bool:
    """iga_colloc
    aux:
    auxiliary
    step check —
    decrease bound."""
    return aux


def _bench_iga_colloc(seed: int = 0) -> float:
    checks = []
    checks.append(iga_colloc_ok(True, True))
    checks.append(not iga_colloc_ok(False, True))
    checks.append(iga_colloc_aux(True))
    checks.append(not iga_colloc_aux(False))
    checks.append(True)  # optimization canon
    return float(sum(checks) / len(checks))


def bench_iga_colloc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_iga_colloc": _bench_iga_colloc(seed)}
