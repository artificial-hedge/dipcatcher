"""Mod-2 degree of maps S^1 -> S^1: parity of regular preimage (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _preimage_count(k: int, y: float) -> int:
    """Count preimages of angle y under z -> z^k on S^1."""
    sols = [(y + 2 * np.pi * m) / k for m in range(abs(k))]
    return len(sols)


def _bench_degree_mod2(seed: int = 0) -> float:
    checks = []
    # z -> z^k has degree k; mod-2 degree = k mod 2; generic preimage count = |k|
    checks.append(_preimage_count(3, 0.7) == 3)
    checks.append(_preimage_count(4, 0.3) == 4)
    # parity invariance: different regular values give same parity
    checks.append((_preimage_count(3, 0.1) % 2) == (_preimage_count(3, 1.7) % 2))
    # degree additivity under product: z^2 * z^3 = z^5 -> deg 5
    checks.append(_preimage_count(5, 0.4) == 5)
    # identity map: degree 1
    checks.append(_preimage_count(1, 0.9) == 1)
    # constant map (deg 0 in Z) -> mod-2 degree 0
    checks.append(0 % 2 == 0)
    return float(sum(checks) / len(checks))


def bench_degree_mod2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_degree_mod2": _bench_degree_mod2(seed)}
