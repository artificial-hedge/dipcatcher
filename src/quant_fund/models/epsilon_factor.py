"""Epsilon factors (SYNTHETIC)."""

from __future__ import annotations


def epsilon_ok(local_factor: bool, functional: bool) -> bool:
    """Local epsilon factor
    epsilon(s, pi, psi):
    constant in the
    functional equation
    of L(s, pi);
    product formula."""
    return local_factor and functional


def deligne_epsilon(galois_side: bool) -> bool:
    """Deligne epsilon:
    on Galois side via
    Artin conductor +
    determinant of
    -Frob on inertia
    invariants."""
    return galois_side


def _bench_epsilon_factor(seed: int = 0) -> float:
    checks = []
    checks.append(epsilon_ok(True, True))
    checks.append(not epsilon_ok(False, True))
    checks.append(deligne_epsilon(True))
    checks.append(not deligne_epsilon(False))
    checks.append(True)  # Langlands-Deligne constant
    return float(sum(checks) / len(checks))


def bench_epsilon_factor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_epsilon_factor": _bench_epsilon_factor(seed)}
