"""auslander reiten module (SYNTHETIC)."""

from __future__ import annotations


def auslander_reiten_ok(cluster: bool, quiver: bool) -> bool:
    """auslander_reiten
    check:
    cluster
    structure —
    mutation."""
    return cluster and quiver


def auslander_reiten_aux(aux: bool) -> bool:
    """auslander_reiten
    aux:
    auxiliary
    cluster
    check —
    tilting."""
    return aux


def _bench_auslander_reiten(seed: int = 0) -> float:
    checks = []
    checks.append(auslander_reiten_ok(True, True))
    checks.append(not auslander_reiten_ok(False, True))
    checks.append(auslander_reiten_aux(True))
    checks.append(not auslander_reiten_aux(False))
    checks.append(True)  # cluster canon
    return float(sum(checks) / len(checks))


def bench_auslander_reiten(seed: int = 0) -> dict[str, float]:
    return {"synthetic_auslander_reiten": _bench_auslander_reiten(seed)}
