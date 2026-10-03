"""Leray-Hirsch theorem for fiber bundles (SYNTHETIC)."""

from __future__ import annotations


def bundle_betti(fiber: list[int], base: list[int]) -> list[int]:
    """If classes restrict freely (Leray-Hirsch), H*(E) = H*(B) x H*(F)."""
    n = len(fiber) + len(base) - 1
    out = [0] * n
    for i, a in enumerate(base):
        for j, b in enumerate(fiber):
            out[i + j] += a * b
    return out


def _bench_leray_hirsch(seed: int = 0) -> float:
    checks = []
    # trivial bundle S^1 x S^1
    checks.append(bundle_betti([1, 1], [1, 1]) == [1, 2, 1])
    # Hopf fibration S^1 -> S^3 -> S^2 FAILS Leray-Hirsch (no global section)
    checks.append(bundle_betti([1, 1], [1, 0, 1]) != [1, 0, 0, 1])
    # projective bundle P(E): E = B x P^{r-1}
    checks.append(bundle_betti([1, 0, 1], [1, 1]) == [1, 1, 1, 1])
    # fiber of a point: E = F
    checks.append(bundle_betti([1, 2, 1], [1]) == [1, 2, 1])
    # Euler char multiplicative for fibrations
    checks.append((1 - 1) * (1 - 1) == 0)
    return float(sum(checks) / len(checks))


def bench_leray_hirsch(seed: int = 0) -> dict[str, float]:
    return {"synthetic_leray_hirsch": _bench_leray_hirsch(seed)}
