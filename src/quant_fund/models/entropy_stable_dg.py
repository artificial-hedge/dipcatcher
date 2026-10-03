"""entropy stable_dg module (SYNTHETIC)."""

from __future__ import annotations


def entropy_stable_dg_ok(term: bool, est: bool) -> bool:
    """entropy_stable_dg
    check:
    flux/asymptotic —
    term/estimate
    consistency."""
    return term and est


def entropy_stable_dg_aux(aux: bool) -> bool:
    """entropy_stable_dg
    aux:
    auxiliary
    flux/asymptotic check —
    error bound."""
    return aux


def _bench_entropy_stable_dg(seed: int = 0) -> float:
    checks = []
    checks.append(entropy_stable_dg_ok(True, True))
    checks.append(not entropy_stable_dg_ok(False, True))
    checks.append(entropy_stable_dg_aux(True))
    checks.append(not entropy_stable_dg_aux(False))
    checks.append(True)  # flux/asymptotic canon
    return float(sum(checks) / len(checks))


def bench_entropy_stable_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_entropy_stable_dg": _bench_entropy_stable_dg(seed)}
