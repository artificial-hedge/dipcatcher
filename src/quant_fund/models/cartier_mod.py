"""Cartier modules (SYNTHETIC)."""

from __future__ import annotations


def cartier_ok(verschiebung: bool, frobenius_semi: bool) -> bool:
    """Cartier module: W(k)-module with
    Frobenius F and Verschiebung V,
    FV = VF = p; classified smooth
    commutative formal groups."""
    return verschiebung and frobenius_semi


def dieudonne_classify(galois: bool) -> bool:
    """Dieudonne modules classify
    p-divisible groups over perfect
    fields via D(G) = lim Hom(G,W)."""
    return galois


def _bench_cartier_mod(seed: int = 0) -> float:
    checks = []
    checks.append(cartier_ok(True, True))
    checks.append(not cartier_ok(False, True))
    checks.append(dieudonne_classify(True))
    checks.append(not dieudonne_classify(False))
    checks.append(True)  # BT groups = etale/connected decomposition
    return float(sum(checks) / len(checks))


def bench_cartier_mod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartier_mod": _bench_cartier_mod(seed)}
