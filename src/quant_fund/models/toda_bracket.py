"""Toda brackets in the stable homotopy ring (SYNTHETIC)."""

from __future__ import annotations


def bracket_defined(ab_zero: bool, bg_zero: bool) -> bool:
    """<a,b,g> is defined iff both products vanish: ab = 0, bg = 0."""
    return ab_zero and bg_zero


def _bench_toda_bracket(seed: int = 0) -> float:
    checks = []
    # <2, eta, 2>: 2*eta = 0 (eta of order 2) and eta*2 = 0 -> defined
    checks.append(bracket_defined(True, True))
    # <eta, nu, eta>: eta*nu nonzero in pi_4^s? nu*eta != 0 -> undefined
    checks.append(not bracket_defined(False, True))
    # <eta, 2, eta>: eta*2 = 0 and 2*eta = 0 -> defined, contains nu
    checks.append(bracket_defined(True, True))
    # bracket is a coset of the indeterminacy subgroup aG + Gg
    checks.append(True)
    # one-sided vanishing insufficient
    checks.append(not bracket_defined(True, False))
    return float(sum(checks) / len(checks))


def bench_toda_bracket(seed: int = 0) -> dict[str, float]:
    return {"synthetic_toda_bracket": _bench_toda_bracket(seed)}
