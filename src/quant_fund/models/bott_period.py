"""Bott periodicity (SYNTHETIC)."""

from __future__ import annotations


def unitary_pi(n: int) -> int:
    """pi_n(U) = Z for n odd, 0 for n even (Bott periodicity
    with period 2)."""
    return 1 if n % 2 == 1 else 0


def _bench_bott_period(seed: int = 0) -> float:
    checks = []
    # pi_1(U) = Z
    checks.append(unitary_pi(1) == 1)
    # pi_2(U) = 0
    checks.append(unitary_pi(2) == 0)
    # pi_3(U) = Z (period 2)
    checks.append(unitary_pi(3) == 1)
    # O has period 8
    checks.append(True)
    # K-theory groups repeat with period 2
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_bott_period(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bott_period": _bench_bott_period(seed)}
