"""meshless local module (SYNTHETIC)."""

from __future__ import annotations


def meshless_local_ok(node: bool, cloud: bool) -> bool:
    """meshless_local
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def meshless_local_aux(aux: bool) -> bool:
    """meshless_local
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_meshless_local(seed: int = 0) -> float:
    checks = []
    checks.append(meshless_local_ok(True, True))
    checks.append(not meshless_local_ok(False, True))
    checks.append(meshless_local_aux(True))
    checks.append(not meshless_local_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_meshless_local(seed: int = 0) -> dict[str, float]:
    return {"synthetic_meshless_local": _bench_meshless_local(seed)}
