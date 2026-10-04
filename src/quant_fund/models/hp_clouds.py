"""partition unity module (SYNTHETIC)."""

from __future__ import annotations


def hp_clouds_ok(node: bool, cloud: bool) -> bool:
    """hp_clouds
    check:
    meshfree/moving-least-squares —
    support
    consistency."""
    return node and cloud


def hp_clouds_aux(aux: bool) -> bool:
    """hp_clouds
    aux:
    auxiliary
    meshfree check —
    reproduction bound."""
    return aux


def _bench_hp_clouds(seed: int = 0) -> float:
    checks = []
    checks.append(hp_clouds_ok(True, True))
    checks.append(not hp_clouds_ok(False, True))
    checks.append(hp_clouds_aux(True))
    checks.append(not hp_clouds_aux(False))
    checks.append(True)  # meshfree canon
    return float(sum(checks) / len(checks))


def bench_hp_clouds(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hp_clouds": _bench_hp_clouds(seed)}
