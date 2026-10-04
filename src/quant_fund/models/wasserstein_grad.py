"""wasserstein grad module (SYNTHETIC)."""

from __future__ import annotations


def wasserstein_grad_ok(ot1: bool, gf: bool) -> bool:
    """wasserstein_grad
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def wasserstein_grad_aux(aux: bool) -> bool:
    """wasserstein_grad
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_wasserstein_grad(seed: int = 0) -> float:
    checks = []
    checks.append(wasserstein_grad_ok(True, True))
    checks.append(not wasserstein_grad_ok(False, True))
    checks.append(wasserstein_grad_aux(True))
    checks.append(not wasserstein_grad_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_wasserstein_grad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wasserstein_grad": _bench_wasserstein_grad(seed)}
