"""Zariski topology on the affine line over GF(p): closed sets = finite ∪ all (SYNTHETIC)."""

from __future__ import annotations


def vanishing_set(polys: list[list[int]], p: int) -> frozenset[int]:
    """V(I) over GF(p): points where all polys vanish (modulus-free eval)."""
    pts = set(range(p))
    for f in polys:
        pts = {a for a in pts if _eval_plain(f, a, p) == 0}
    return frozenset(pts)


def _eval_plain(f: list[int], a: int, p: int) -> int:
    acc = 0
    for i, c in enumerate(f):
        acc = (acc + c * pow(a, i, p)) % p
    return acc


def is_zariski_closed(univ_size: int, subset: frozenset[int]) -> bool:
    """On A^1 over GF(p): closed iff finite or the whole space."""
    return len(subset) < univ_size or len(subset) == univ_size


def ideal_of(subset: frozenset[int], p: int) -> list[int]:
    """Vanishing ideal of a finite point set: product of (x - a)."""
    from quant_fund.models.field_ext import pmul

    acc = [1]
    for a in subset:
        acc = pmul(acc, [(-a) % p, 1], p)
    return acc


def _bench_zariski_topo(seed: int = 0) -> float:
    checks = []
    p = 5
    checks.append(vanishing_set([[0, 0, 1]], p) == frozenset({0}))  # x^2 vanishes at 0
    checks.append(vanishing_set([[4, 1]], p) == frozenset({1}))
    checks.append(vanishing_set([[4, 1]], p) == frozenset({1}))  # x - 1
    f = ideal_of(frozenset({1, 3}), p)
    checks.append(vanishing_set([f], p) == frozenset({1, 3}))
    checks.append(all(_eval_plain(f, a, p) == 0 for a in (1, 3)))
    checks.append(is_zariski_closed(p, frozenset({0, 2, 4})))
    checks.append(is_zariski_closed(p, frozenset(range(p))))
    return float(sum(checks) / len(checks))


def bench_zariski_topo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zariski_topo": _bench_zariski_topo(seed)}
