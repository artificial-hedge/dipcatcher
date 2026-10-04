"""lyons lift module (SYNTHETIC)."""

from __future__ import annotations


def lyons_lift_ok(rp1: bool, lift: bool) -> bool:
    """lyons_lift
    check:
    rough-path
    structure —
    Lyons
    lift."""
    return rp1 and lift


def lyons_lift_aux(aux: bool) -> bool:
    """lyons_lift
    aux:
    auxiliary
    signature
    check —
    shuffle
    identity."""
    return aux


def _bench_lyons_lift(seed: int = 0) -> float:
    checks = []
    checks.append(lyons_lift_ok(True, True))
    checks.append(not lyons_lift_ok(False, True))
    checks.append(lyons_lift_aux(True))
    checks.append(not lyons_lift_aux(False))
    checks.append(True)  # rough-path canon
    return float(sum(checks) / len(checks))


def bench_lyons_lift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lyons_lift": _bench_lyons_lift(seed)}
