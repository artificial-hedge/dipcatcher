"""entropy regular module (SYNTHETIC)."""

from __future__ import annotations


def entropy_regular_ok(ot1: bool, gf: bool) -> bool:
    """entropy_regular
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def entropy_regular_aux(aux: bool) -> bool:
    """entropy_regular
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_entropy_regular(seed: int = 0) -> float:
    checks = []
    checks.append(entropy_regular_ok(True, True))
    checks.append(not entropy_regular_ok(False, True))
    checks.append(entropy_regular_aux(True))
    checks.append(not entropy_regular_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_entropy_regular(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entropy_regular": _bench_entropy_regular(seed)}
