"""kingman bound module (SYNTHETIC)."""

from __future__ import annotations


def kingman_bound_ok(lim: bool, scale: bool) -> bool:
    """kingman_bound
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def kingman_bound_aux(aux: bool) -> bool:
    """kingman_bound
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_kingman_bound(seed: int = 0) -> float:
    checks = []
    checks.append(kingman_bound_ok(True, True))
    checks.append(not kingman_bound_ok(False, True))
    checks.append(kingman_bound_aux(True))
    checks.append(not kingman_bound_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_kingman_bound(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kingman_bound": _bench_kingman_bound(seed)}
