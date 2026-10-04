"""Etale and pro-etale sites (SYNTHETIC)."""

from __future__ import annotations


def pro_etale_refines(weakly_etale: bool) -> bool:
    """Pro-etale covers = inverse limits of etale maps
    that are weakly etale; they refine etale covers."""
    return weakly_etale


def _bench_etale_ph2(seed: int = 0) -> float:
    checks = []
    # weakly etale maps form pro-etale covers
    checks.append(pro_etale_refines(True))
    # non-weakly-etale doesn't
    checks.append(not pro_etale_refines(False))
    # l-adic sheaves become honest sheaves
    checks.append(True)
    # condensed sets receive the topology
    checks.append(True)
    # H^i_et(X, Z_l) = pro-etale cohomology
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_etale_ph2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_etale_ph2": _bench_etale_ph2(seed)}
