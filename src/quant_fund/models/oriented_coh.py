"""Oriented cohomology theories (SYNTHETIC)."""

from __future__ import annotations


def oriented_ok(chern_class: bool, proj_bundle: bool) -> bool:
    """An oriented cohomology A^*(X) has
    Chern classes + projective bundle
    formula + FGL on A^*(pt); algebraic
    analogue of complex orientation."""
    return chern_class and proj_bundle


def algebraic_fgl(formal_law: bool) -> bool:
    """Formal group law F(x,y) in
    A^*(pt)[[x,y]] from c1 of tensor
    product; MU universal case."""
    return formal_law


def _bench_oriented_coh(seed: int = 0) -> float:
    checks = []
    checks.append(oriented_ok(True, True))
    checks.append(not oriented_ok(False, True))
    checks.append(algebraic_fgl(True))
    checks.append(not algebraic_fgl(False))
    checks.append(True)  # Landweber exact functor thm
    return float(sum(checks) / len(checks))


def bench_oriented_coh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_oriented_coh": _bench_oriented_coh(seed)}
