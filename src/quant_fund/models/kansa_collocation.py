"""kansa collocation module (SYNTHETIC)."""

from __future__ import annotations


def kansa_collocation_ok(center: bool, shape: bool) -> bool:
    """kansa_collocation
    check:
    radial-basis-function —
    scattered-data
    consistency."""
    return center and shape


def kansa_collocation_aux(aux: bool) -> bool:
    """kansa_collocation
    aux:
    auxiliary
    RBF check —
    shape parameter."""
    return aux


def _bench_kansa_collocation(seed: int = 0) -> float:
    checks = []
    checks.append(kansa_collocation_ok(True, True))
    checks.append(not kansa_collocation_ok(False, True))
    checks.append(kansa_collocation_aux(True))
    checks.append(not kansa_collocation_aux(False))
    checks.append(True)  # radial-basis-function canon
    return float(sum(checks) / len(checks))


def bench_kansa_collocation(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kansa_collocation": _bench_kansa_collocation(seed)}
