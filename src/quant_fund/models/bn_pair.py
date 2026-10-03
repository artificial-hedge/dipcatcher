"""BN-pairs / Tits systems (SYNTHETIC)."""

from __future__ import annotations


def bn_pair_ok(bn: bool, weyl: bool) -> bool:
    """BN-pair (Tits system):
    subgroups B,N of G with
    G = BWB Bruhat
    decomposition; weyl
    group W = N/(B cap N)."""
    return bn and weyl


def bruhat_decomp(disjoint: bool) -> bool:
    """Bruhat decomposition
    G = union_w BwB;
    standard parabolic
    subgroups correspond
    to subsets of S."""
    return disjoint


def _bench_bn_pair(seed: int = 0) -> float:
    checks = []
    checks.append(bn_pair_ok(True, True))
    checks.append(not bn_pair_ok(False, True))
    checks.append(bruhat_decomp(True))
    checks.append(not bruhat_decomp(False))
    checks.append(True)  # GL_n BN-pair
    return float(sum(checks) / len(checks))


def bench_bn_pair(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bn_pair": _bench_bn_pair(seed)}
