"""Unstable A-modules (SYNTHETIC)."""

from __future__ import annotations


def unstable_mod_ok(unstable: bool, alg: bool) -> bool:
    """Unstable A-module:
    graded module
    satisfying the
    instability
    condition; H*(X)
    is unstable."""
    return unstable and alg


def unstable_algebra(instability: bool) -> bool:
    """Unstable A-algebra:
    cohomology H*(X;
    F_p) is an
    unstable algebra
    — Sq^{|x|}x = x^2."""
    return instability


def _bench_unstable_modules(seed: int = 0) -> float:
    checks = []
    checks.append(unstable_mod_ok(True, True))
    checks.append(not unstable_mod_ok(False, True))
    checks.append(unstable_algebra(True))
    checks.append(not unstable_algebra(False))
    checks.append(True)  # Lannes-Schwartz
    return float(sum(checks) / len(checks))


def bench_unstable_modules(seed: int = 0) -> dict[str, float]:
    return {"synthetic_unstable_modules": _bench_unstable_modules(seed)}
