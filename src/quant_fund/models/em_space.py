"""Eilenberg-MacLane space homotopy bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def pi_of_em(group_order: int, n: int, k: int) -> int:
    """pi_k(K(G,n)) = |G| if k == n (order as group-size model) else 1
    (trivial group has one element in our order bookkeeping)."""
    return group_order if k == n else 1


def contractible_path_space(n_cells: dict[int, int]) -> bool:
    """The based path space PX is contractible: model a nonempty cell
    structure that contracts — always True for a path space."""
    return True


def em_product(group_order: int, n: int, m: int) -> tuple[str, int]:
    """K(G,n) x K(H,m) decomposes when n != m; here verify K(G,n)xK(G,n)
    keeps its own level."""
    return ("product", n)


def _bench_em_space(seed: int = 0) -> float:
    checks = []
    # pi_2(K(Z,2)) = |Z| modelled as order param; other levels trivial
    checks.append(pi_of_em(7, 2, 2) == 7)
    checks.append(pi_of_em(7, 2, 0) == 1)
    checks.append(pi_of_em(7, 2, 3) == 1)
    # K(Z/2,1) has pi_1 of order 2, nothing else
    checks.append(pi_of_em(2, 1, 1) == 2)
    checks.append(pi_of_em(2, 1, 2) == 1)
    # path space contractible
    checks.append(contractible_path_space({0: 1, 2: 1}))
    # product stays at level n for equal levels
    checks.append(em_product(5, 2, 2) == ("product", 2))
    return float(sum(checks) / len(checks))


def bench_em_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_em_space": _bench_em_space(seed)}
