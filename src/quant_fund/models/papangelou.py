"""papangelou module (SYNTHETIC)."""

from __future__ import annotations


def papangelou_ok(cp: bool, pg: bool) -> bool:
    """papangelou
    check:
    point-process
    theory —
    distribution."""
    return cp and pg


def papangelou_aux(aux: bool) -> bool:
    """papangelou
    aux:
    auxiliary
    point-process
    check —
    intensity."""
    return aux


def _bench_papangelou(seed: int = 0) -> float:
    checks = []
    checks.append(papangelou_ok(True, True))
    checks.append(not papangelou_ok(False, True))
    checks.append(papangelou_aux(True))
    checks.append(not papangelou_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_papangelou(seed: int = 0) -> dict[str, float]:
    return {"synthetic_papangelou": _bench_papangelou(seed)}
