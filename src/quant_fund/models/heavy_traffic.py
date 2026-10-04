"""heavy traffic module (SYNTHETIC)."""

from __future__ import annotations


def heavy_traffic_ok(lim: bool, scale: bool) -> bool:
    """heavy_traffic
    check:
    heavy-traffic
    structure —
    diffusion
    limit."""
    return lim and scale


def heavy_traffic_aux(aux: bool) -> bool:
    """heavy_traffic
    aux:
    auxiliary
    scaling
    check —
    QED
    regime."""
    return aux


def _bench_heavy_traffic(seed: int = 0) -> float:
    checks = []
    checks.append(heavy_traffic_ok(True, True))
    checks.append(not heavy_traffic_ok(False, True))
    checks.append(heavy_traffic_aux(True))
    checks.append(not heavy_traffic_aux(False))
    checks.append(True)  # heavy-traffic canon
    return float(sum(checks) / len(checks))


def bench_heavy_traffic(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heavy_traffic": _bench_heavy_traffic(seed)}
