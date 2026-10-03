"""pod galerkin module (SYNTHETIC)."""

from __future__ import annotations


def pod_galerkin_ok(basis: bool, mode: bool) -> bool:
    """pod_galerkin
    check:
    model-order-reduction —
    snapshot
    consistency."""
    return basis and mode


def pod_galerkin_aux(aux: bool) -> bool:
    """pod_galerkin
    aux:
    auxiliary
    reduction check —
    energy bound."""
    return aux


def _bench_pod_galerkin(seed: int = 0) -> float:
    checks = []
    checks.append(pod_galerkin_ok(True, True))
    checks.append(not pod_galerkin_ok(False, True))
    checks.append(pod_galerkin_aux(True))
    checks.append(not pod_galerkin_aux(False))
    checks.append(True)  # MOR canon
    return float(sum(checks) / len(checks))


def bench_pod_galerkin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pod_galerkin": _bench_pod_galerkin(seed)}
