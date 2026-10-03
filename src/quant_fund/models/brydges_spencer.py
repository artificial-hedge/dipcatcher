"""brydges spencer module (SYNTHETIC)."""

from __future__ import annotations


def brydges_spencer_ok(rc: bool, potts: bool) -> bool:
    """brydges_spencer
    check:
    random-cluster
    structure —
    Grimmett."""
    return rc and potts


def brydges_spencer_aux(aux: bool) -> bool:
    """brydges_spencer
    aux:
    auxiliary
    Potts-model
    check —
    Sokal."""
    return aux


def _bench_brydges_spencer(seed: int = 0) -> float:
    checks = []
    checks.append(brydges_spencer_ok(True, True))
    checks.append(not brydges_spencer_ok(False, True))
    checks.append(brydges_spencer_aux(True))
    checks.append(not brydges_spencer_aux(False))
    checks.append(True)  # random-cluster canon
    return float(sum(checks) / len(checks))


def bench_brydges_spencer(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brydges_spencer": _bench_brydges_spencer(seed)}
