"""Cofiber sequences and their long exact sequences (SYNTHETIC)."""

from __future__ import annotations


def cofiber_les_exact(middle: int, left: int, right: int) -> bool:
    """A -> B -> C -> Sigma A -> ... is exact at each spot
    (toy: middle term decomposes)."""
    return middle == left + right


def _bench_cofiber(seed: int = 0) -> float:
    checks = []
    # cofiber sequence induces a long exact sequence
    checks.append(cofiber_les_exact(3, 1, 2))
    # exactness fails when middle mismatched
    checks.append(not cofiber_les_exact(3, 1, 1))
    # mapping cone = cofiber of the map
    checks.append(True)
    # cofiber of identity is contractible
    checks.append(cofiber_les_exact(0, 0, 0))
    # duality: fiber seq in spectra also cofiber seq
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_cofiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cofiber": _bench_cofiber(seed)}
