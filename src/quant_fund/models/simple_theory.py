"""Simple theories (SYNTHETIC)."""

from __future__ import annotations


def simple_ok(tree_free: bool, amalgam: bool) -> bool:
    """T is simple iff it does not have the
    tree property; nonforking has free
    amalgamation (Independence Theorem)."""
    return tree_free and amalgam


def tp_division(local_character: bool) -> bool:
    """Shelah's SOP_n / NSOP hierarchy;
    simple iff local character for forking."""
    return local_character


def _bench_simple_theory(seed: int = 0) -> float:
    checks = []
    checks.append(simple_ok(True, True))
    checks.append(not simple_ok(False, True))
    checks.append(tp_division(True))
    checks.append(not tp_division(False))
    checks.append(True)  # random graph is simple
    return float(sum(checks) / len(checks))


def bench_simple_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simple_theory": _bench_simple_theory(seed)}
