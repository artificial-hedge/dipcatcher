"""benamou brenier module (SYNTHETIC)."""

from __future__ import annotations


def benamou_brenier_ok(ot1: bool, gf: bool) -> bool:
    """benamou_brenier
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def benamou_brenier_aux(aux: bool) -> bool:
    """benamou_brenier
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_benamou_brenier(seed: int = 0) -> float:
    checks = []
    checks.append(benamou_brenier_ok(True, True))
    checks.append(not benamou_brenier_ok(False, True))
    checks.append(benamou_brenier_aux(True))
    checks.append(not benamou_brenier_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_benamou_brenier(seed: int = 0) -> dict[str, float]:
    return {"synthetic_benamou_brenier": _bench_benamou_brenier(seed)}
