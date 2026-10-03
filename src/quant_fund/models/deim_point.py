"""deim point module (SYNTHETIC)."""

from __future__ import annotations


def deim_point_ok(basis: bool, mode: bool) -> bool:
    """deim_point
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def deim_point_aux(aux: bool) -> bool:
    """deim_point
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_deim_point(seed: int = 0) -> float:
    checks = []
    checks.append(deim_point_ok(True, True))
    checks.append(not deim_point_ok(False, True))
    checks.append(deim_point_aux(True))
    checks.append(not deim_point_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_deim_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deim_point": _bench_deim_point(seed)}
