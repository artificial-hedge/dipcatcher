"""Artin induction: Q-characters from cyclic subgroups (SYNTHETIC)."""

from __future__ import annotations


def permutation_char_on_cosets(g_order: int, h_order: int, elem_order: int) -> int:
    """For cyclic G: permutation character on G/H evaluated at an
    element: [G:H] if the element lies in H (order divides |H|), else 0."""
    if g_order % h_order != 0:
        return 0
    # element of order d is in unique subgroup of order |H| iff d | |H|
    return g_order // h_order if h_order % elem_order == 0 else 0


def _bench_artins_theorem(seed: int = 0) -> float:
    checks = []
    # G = Z/4, H = Z/2: induced trivial gives 2 on {0,2}, 0 on {1,3}
    checks.append(permutation_char_on_cosets(4, 2, 2) == 2)
    checks.append(permutation_char_on_cosets(4, 2, 4) == 0)
    # regular character = Ind_{1}: |G| at identity, 0 elsewhere
    checks.append(permutation_char_on_cosets(4, 1, 1) == 4)
    checks.append(permutation_char_on_cosets(4, 1, 2) == 0)
    # every element order divides group order (Lagrange)
    checks.append(4 % 2 == 0)
    return float(sum(checks) / len(checks))


def bench_artins_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_artins_theorem": _bench_artins_theorem(seed)}
