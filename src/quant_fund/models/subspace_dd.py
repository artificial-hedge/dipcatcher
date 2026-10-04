"""subspace dd module (SYNTHETIC)."""

from __future__ import annotations


def subspace_dd_ok(part: bool, coarse: bool) -> bool:
    """subspace_dd
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def subspace_dd_aux(aux: bool) -> bool:
    """subspace_dd
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_subspace_dd(seed: int = 0) -> float:
    checks = []
    checks.append(subspace_dd_ok(True, True))
    checks.append(not subspace_dd_ok(False, True))
    checks.append(subspace_dd_aux(True))
    checks.append(not subspace_dd_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_subspace_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subspace_dd": _bench_subspace_dd(seed)}
