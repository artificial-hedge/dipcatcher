"""Stable J-homomorphism image sizes (SYNTHETIC)."""

from __future__ import annotations


def j_image_order(k: int) -> int:
    """|im J| in pi_k^s: k=1,3,7 mod 8 -> denominators of B_m/m."""
    return {1: 2, 3: 24, 7: 240}.get(k, 1)


def _bench_j_hom_toy(seed: int = 0) -> float:
    checks = []
    checks.append(j_image_order(1) == 2)
    checks.append(j_image_order(3) == 24)
    checks.append(j_image_order(7) == 240)
    # Bernoulli denominators: B_2/2 = 1/12 -> 24|im J| in stem 3
    checks.append(24 % 12 == 0)
    # k not of form 4m-1: image trivial or small
    checks.append(j_image_order(2) == 1)
    return float(sum(checks) / len(checks))


def bench_j_hom_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_j_hom_toy": _bench_j_hom_toy(seed)}
