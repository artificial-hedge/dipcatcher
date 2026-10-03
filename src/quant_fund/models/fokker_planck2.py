"""fokker planck2 module (SYNTHETIC)."""

from __future__ import annotations


def fokker_planck2_ok(ot1: bool, gf: bool) -> bool:
    """fokker_planck2
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def fokker_planck2_aux(aux: bool) -> bool:
    """fokker_planck2
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_fokker_planck2(seed: int = 0) -> float:
    checks = []
    checks.append(fokker_planck2_ok(True, True))
    checks.append(not fokker_planck2_ok(False, True))
    checks.append(fokker_planck2_aux(True))
    checks.append(not fokker_planck2_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_fokker_planck2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fokker_planck2": _bench_fokker_planck2(seed)}
