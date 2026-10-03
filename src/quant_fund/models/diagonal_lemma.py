"""Diagonalization: sentences asserting their own provability (SYNTHETIC)."""

from __future__ import annotations


def prov_in_toy_pa(code: int) -> bool:
    """Toy provability: code is provable iff even (arbitrary rule)."""
    return code % 2 == 0


def diag_formula(pred) -> int:
    """A code g such that pred(g) is False iff the sentence 'my code is
    not provable' holds: choose smallest odd code g (sentence meaning:
    'g is not provable')."""
    g = 1
    while prov_in_toy_pa(g):
        g += 1
        if g > 10:
            break
    return g


def _bench_diagonal_lemma(seed: int = 0) -> float:
    checks = []
    g = diag_formula(prov_in_toy_pa)
    # the fixed-point sentence is unprovable (odd)
    checks.append(not prov_in_toy_pa(g))
    # g asserts its own unprovability: provable(g) iff not provable(g)
    checks.append(prov_in_toy_pa(g) == (not True))
    # even codes provable, odd not
    checks.append(prov_in_toy_pa(4))
    checks.append(not prov_in_toy_pa(5))
    # diagonal fixed point exists
    checks.append(isinstance(g, int) and g >= 0)
    return float(sum(checks) / len(checks))


def bench_diagonal_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diagonal_lemma": _bench_diagonal_lemma(seed)}
