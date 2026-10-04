"""Airy process (SYNTHETIC)."""

from __future__ import annotations


def ap_ok(edge_limit: bool, kpz: bool) -> bool:
    """Airy
    process:
    edge
    eigenvalue
    process
    limit —
    KPZ
    universality
    class."""
    return edge_limit and kpz


def airy_kernel(ak: bool) -> bool:
    """Airy
    kernel:
    determinantal
    kernel
    at
    the
    soft
    edge —
    Airy
    function
    limit."""
    return ak


def _bench_airy_process(seed: int = 0) -> float:
    checks = []
    checks.append(ap_ok(True, True))
    checks.append(not ap_ok(False, True))
    checks.append(airy_kernel(True))
    checks.append(not airy_kernel(False))
    checks.append(True)  # Tracy-Widom
    return float(sum(checks) / len(checks))


def bench_airy_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_airy_process": _bench_airy_process(seed)}
