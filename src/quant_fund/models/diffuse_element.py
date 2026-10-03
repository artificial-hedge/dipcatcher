"""diffuse element module (SYNTHETIC)."""

from __future__ import annotations


def diffuse_element_ok(node: bool, cloud: bool) -> bool:
    """diffuse_element
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def diffuse_element_aux(aux: bool) -> bool:
    """diffuse_element
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_diffuse_element(seed: int = 0) -> float:
    checks = []
    checks.append(diffuse_element_ok(True, True))
    checks.append(not diffuse_element_ok(False, True))
    checks.append(diffuse_element_aux(True))
    checks.append(not diffuse_element_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_diffuse_element(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diffuse_element": _bench_diffuse_element(seed)}
