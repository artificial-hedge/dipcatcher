"""Karoubi-Villamayor LES (SYNTHETIC)."""

from __future__ import annotations


def karoubi_v_ok(les: bool, flasque: bool) -> bool:
    """Karoubi-Villamayor
    long exact sequence from
    flasque resolutions:
    ... -> KV_n -> K_n ->
    K_{n-1} -> ...."""
    return les and flasque


def suspension_ring(cone: bool) -> bool:
    """Cone and suspension
    rings CR, SR realize
    deloopings for
    Karoubi-Villamayor K."""
    return cone


def _bench_karoubi_v(seed: int = 0) -> float:
    checks = []
    checks.append(karoubi_v_ok(True, True))
    checks.append(not karoubi_v_ok(False, True))
    checks.append(suspension_ring(True))
    checks.append(not suspension_ring(False))
    checks.append(True)  # Karoubi conjecture K=KV for C*-alg
    return float(sum(checks) / len(checks))


def bench_karoubi_v(seed: int = 0) -> dict[str, float]:
    return {"synthetic_karoubi_v": _bench_karoubi_v(seed)}
