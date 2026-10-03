"""Stable homotopy groups of spheres (low stems) (SYNTHETIC)."""

from __future__ import annotations


def stem_order(k: int) -> int:
    """|pi_k^s| for k = 0..7 (0 = Z)."""
    return {0: 0, 1: 2, 2: 2, 3: 24, 4: 1, 5: 1, 6: 2, 7: 240}[k]


def _bench_pi_stems(seed: int = 0) -> float:
    checks = []
    checks.append(stem_order(0) == 0)  # Z
    checks.append(stem_order(1) == 2)  # eta
    checks.append(stem_order(2) == 2)  # eta^2
    checks.append(stem_order(3) == 24)  # nu, Z/24
    checks.append(stem_order(4) == 1)  # trivial
    checks.append(stem_order(6) == 2)
    checks.append(stem_order(7) == 240)  # sigma, Z/240
    return float(sum(checks) / len(checks))


def bench_pi_stems(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pi_stems": _bench_pi_stems(seed)}
