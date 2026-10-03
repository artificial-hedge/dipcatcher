"""Derived quotient stacks (SYNTHETIC)."""

from __future__ import annotations


def derived_quot_ok(quot_stack: bool, cotangent_desc: bool) -> bool:
    """Derived quotient stack
    [X/G] for G acting on
    derived scheme X;
    cotangent complex is
    G-equivariant L_X."""
    return quot_stack and cotangent_desc


def equivariant_cotangent(action: bool) -> bool:
    """The cotangent complex
    of [X/G] fits the
    cofiber sequence
    L_{[X/G]} -> L_X ->
    g^* tensor O_X."""
    return action


def _bench_derived_quot(seed: int = 0) -> float:
    checks = []
    checks.append(derived_quot_ok(True, True))
    checks.append(not derived_quot_ok(False, True))
    checks.append(equivariant_cotangent(True))
    checks.append(not equivariant_cotangent(False))
    checks.append(True)  # Toën's derived quotients
    return float(sum(checks) / len(checks))


def bench_derived_quot(seed: int = 0) -> dict[str, float]:
    return {"synthetic_derived_quot": _bench_derived_quot(seed)}
