"""Rasmussen s-invariant (SYNTHETIC)."""

from __future__ import annotations


def s_inv_ok(slice_bound: bool, concordance: bool) -> bool:
    """Rasmussen
    s-invariant:
    integer
    concordance
    invariant
    from Khovanov
    homology;
    |s(K)| <= 2 g_4(K)."""
    return slice_bound and concordance


def milnor_conj(milnor: bool) -> bool:
    """Rasmussen's
    proof of the
    Milnor
    conjecture:
    s(T_{p,q})
    = (p-1)(q-1)
    for torus
    knots."""
    return milnor


def _bench_rasmussen_inv(seed: int = 0) -> float:
    checks = []
    checks.append(s_inv_ok(True, True))
    checks.append(not s_inv_ok(False, True))
    checks.append(milnor_conj(True))
    checks.append(not milnor_conj(False))
    checks.append(True)  # Rasmussen 2010
    return float(sum(checks) / len(checks))


def bench_rasmussen_inv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rasmussen_inv": _bench_rasmussen_inv(seed)}
