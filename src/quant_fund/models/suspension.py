"""Suspension and Freudenthal bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def suspend_betti(betti: list[int]) -> list[int]:
    """Reduced homology shifts up one under suspension:
    H~_{n+1}(SX) = H~_n(X), so reduced betti list shifts right by one."""
    return [0] + list(betti)


def freudenthal_stable(n: int, i: int) -> bool:
    """Freudenthal: pi_i(X) -> pi_{i+1}(SX) is an iso for i <= 2n - 2
    (X = S^n, i.e. i < 2n - 1) and a surjection at i = 2n - 1."""
    return i <= 2 * n - 2


def _bench_suspension(seed: int = 0) -> float:
    checks = []
    # S1 -> S2: reduced betti [0,1] -> [0,0,1]
    checks.append(suspend_betti([0, 1]) == [0, 0, 1])
    # suspending twice: S^1 -> S^3
    checks.append(suspend_betti(suspend_betti([0, 1])) == [0, 0, 0, 1])
    # torus suspended: [0,2,1] -> [0,0,2,1]
    checks.append(suspend_betti([0, 2, 1]) == [0, 0, 2, 1])
    # Freudenthal range: pi_i(S^n) stabilization for i <= 2n-2
    checks.append(freudenthal_stable(2, 2))  # pi2->pi3 on S2 stable? i=2 <= 2
    checks.append(not freudenthal_stable(2, 3))  # i=3 > 2 surj only
    checks.append(freudenthal_stable(3, 4))  # i=4 <= 4
    return float(sum(checks) / len(checks))


def bench_suspension(seed: int = 0) -> dict[str, float]:
    return {"synthetic_suspension": _bench_suspension(seed)}
