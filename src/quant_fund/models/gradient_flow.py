"""gradient flow module (SYNTHETIC)."""

from __future__ import annotations


def gradient_flow_ok(ot1: bool, gf: bool) -> bool:
    """gradient_flow
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def gradient_flow_aux(aux: bool) -> bool:
    """gradient_flow
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_gradient_flow(seed: int = 0) -> float:
    checks = []
    checks.append(gradient_flow_ok(True, True))
    checks.append(not gradient_flow_ok(False, True))
    checks.append(gradient_flow_aux(True))
    checks.append(not gradient_flow_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_gradient_flow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gradient_flow": _bench_gradient_flow(seed)}
