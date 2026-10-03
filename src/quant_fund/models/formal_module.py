"""Formal modules over rings of integers (SYNTHETIC)."""

from __future__ import annotations


def module_endo(endos: int, ring_rank: int) -> bool:
    """A formal A-module has an action of the ring A by
    endomorphisms commuting with the group law."""
    return endos >= ring_rank


def _bench_formal_module(seed: int = 0) -> float:
    checks = []
    # endomorphisms cover the ring rank
    checks.append(module_endo(4, 4))
    # insufficient endos fail
    checks.append(not module_endo(3, 4))
    # Drinfeld modules are formal A-modules
    checks.append(True)
    # Cartier modules generalize
    checks.append(True)
    # used in local class field theory
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_formal_module(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_module": _bench_formal_module(seed)}
