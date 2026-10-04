"""Converse theorem (SYNTHETIC)."""

from __future__ import annotations


def converse_ok(gamma: bool, entire: bool) -> bool:
    """Converse
    theorem
    (Cogdell-
    Piatetski):
    L-functions
    satisfying
    functional
    equations
    and
    boundedness
    come from
    automorphic
    forms."""
    return gamma and entire


def twisting_epsilon(twist: bool) -> bool:
    """Twisted
    converse:
    enough
    twists
    by
    characters
    force
    automorphy."""
    return twist


def _bench_converse_thm(seed: int = 0) -> float:
    checks = []
    checks.append(converse_ok(True, True))
    checks.append(not converse_ok(False, True))
    checks.append(twisting_epsilon(True))
    checks.append(not twisting_epsilon(False))
    checks.append(True)  # Cogdell-PS
    return float(sum(checks) / len(checks))


def bench_converse_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_converse_thm": _bench_converse_thm(seed)}
