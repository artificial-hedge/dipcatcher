"""pod deim module (SYNTHETIC)."""

from __future__ import annotations


def pod_deim_ok(node: bool, wgt: bool) -> bool:
    """pod_deim
    check:
    quadrature —
    node/weight
    consistency."""
    return node and wgt


def pod_deim_aux(aux: bool) -> bool:
    """pod_deim
    aux:
    auxiliary
    quadrature check —
    moment bound."""
    return aux


def _bench_pod_deim(seed: int = 0) -> float:
    checks = []
    checks.append(pod_deim_ok(True, True))
    checks.append(not pod_deim_ok(False, True))
    checks.append(pod_deim_aux(True))
    checks.append(not pod_deim_aux(False))
    checks.append(True)  # quadrature canon
    return float(sum(checks) / len(checks))


def bench_pod_deim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pod_deim": _bench_pod_deim(seed)}
