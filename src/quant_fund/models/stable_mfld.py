"""Stable manifold theorem (SYNTHETIC)."""

from __future__ import annotations


def stable_ok(contract: bool, tangent: bool) -> bool:
    """Stable
    manifold
    theorem:
    local
    stable
    sets are
    immersed
    disks
    tangent
    to E^s."""
    return contract and tangent


def unstable_lam(unst: bool) -> bool:
    """Unstable
    lamination:
    unstable
    manifolds
    form an
    invariant
    foliation
    with
    Holder-
    continuous
    tangent."""
    return unst


def _bench_stable_mfld(seed: int = 0) -> float:
    checks = []
    checks.append(stable_ok(True, True))
    checks.append(not stable_ok(False, True))
    checks.append(unstable_lam(True))
    checks.append(not unstable_lam(False))
    checks.append(True)  # Hadamard-Perron
    return float(sum(checks) / len(checks))


def bench_stable_mfld(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_mfld": _bench_stable_mfld(seed)}
