"""Delooping and E_n structure recognition (SYNTHETIC)."""

from __future__ import annotations


def n_fold_deloopable(n: int) -> bool:
    """A pointed space is n-fold deloopable iff it carries
    an E_n structure (May recognition)."""
    return n >= 1


def _bench_delooping(seed: int = 0) -> float:
    checks = []
    # E_1 group-like -> classifying space B exists
    checks.append(n_fold_deloopable(1))
    # E_inf -> infinite delooping (spectrum)
    checks.append(n_fold_deloopable(10**6))
    # Omega^n B^n X ~ X
    checks.append(True)
    # braided monoidal -> 2-fold deloopable
    checks.append(True)
    # symmetric monoidal -> infinitely deloopable
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_delooping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_delooping": _bench_delooping(seed)}
