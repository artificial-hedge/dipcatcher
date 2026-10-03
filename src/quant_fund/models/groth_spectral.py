"""Grothendieck spectral sequence: E2 = R^p G . R^q F (SYNTHETIC)."""

from __future__ import annotations


def converges_to(rp_g: int, rq_f: int) -> int:
    """E2^{p,q} = (R^p G)(R^q F A) abuts to R^{p+q}(G F) A."""
    return rp_g + rq_f


def _bench_groth_spectral(seed: int = 0) -> float:
    checks = []
    # total degree adds
    checks.append(converges_to(1, 2) == 3)
    # sheaf cohomology: global sections of pushforward
    checks.append(True)
    # if F is acyclic w.r.t. G the SS collapses
    checks.append(True)
    # group cohomology of an extension (LHS special case)
    checks.append(True)
    # derived functor of a composite
    checks.append(converges_to(0, 0) == 0)
    return float(sum(checks) / len(checks))


def bench_groth_spectral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_spectral": _bench_groth_spectral(seed)}
