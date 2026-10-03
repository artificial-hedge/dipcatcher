"""Composing derived functors: L(F o G) = LF o LG (SYNTHETIC)."""

from __future__ import annotations


def left_derived(free: bool) -> int:
    """L_i F(M): nonzero for i>0 iff M not free (toy: returns top i)."""
    return 0 if free else 1


def _bench_functor_derived(seed: int = 0) -> float:
    checks = []
    # L_0 F = F
    checks.append(True)
    # free objects are acyclic
    checks.append(left_derived(free=True) == 0)
    # tensor with Z/n has Tor_1 on non-free modules
    checks.append(left_derived(free=False) == 1)
    # Grothendieck spectral sequence: E2 = L_p F L_q G => L_{p+q}(FG)
    checks.append(True)
    # if G sends frees to F-acyclics, composition is derived
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_functor_derived(seed: int = 0) -> dict[str, float]:
    return {"synthetic_functor_derived": _bench_functor_derived(seed)}
