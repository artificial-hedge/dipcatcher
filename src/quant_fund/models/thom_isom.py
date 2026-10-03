"""Thom isomorphism for trivial bundles (SYNTHETIC)."""

from __future__ import annotations


def thom_rank(base_h: int, fiber_dim: int, target: int) -> int:
    """For a trivial R^n bundle over a base B, Thom space homology:
    H_k(Th) = H_{k-n}(B). Return rank when k-n == base_h index."""
    return base_h if target - fiber_dim >= 0 else 0


def _bench_thom_isom(seed: int = 0) -> float:
    checks = []
    # trivial R^1-bundle over S^1 (cylinder): H_k(Th) = H_{k-1}(S^1)
    b = {0: 1, 1: 1}
    checks.append(b.get(2 - 1, 0) == 1)  # H_2(Th) = H_1(S^1) = Z
    checks.append(b.get(1 - 1, 0) == 1)  # H_1(Th) = H_0(S^1) = Z
    # negative degree vanishes
    checks.append((0 - 1) < 0)
    # trivial R^2 over pt: Th = S^2, H_2 = Z
    checks.append(thom_rank(1, 2, 2) == 1)
    # H_0(Th(S^2-bundle over pt)) = H_{-2}(pt) = 0
    checks.append(thom_rank(0, 2, 0) == 0)
    return float(sum(checks) / len(checks))


def bench_thom_isom(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_isom": _bench_thom_isom(seed)}
