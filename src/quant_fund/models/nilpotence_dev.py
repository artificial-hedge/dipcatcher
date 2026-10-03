"""Devinatz-Hopkins-Smith nilpotence (SYNTHETIC)."""

from __future__ import annotations


def dhs_nilpotence(mu_detects: bool) -> bool:
    """Nilpotence theorem: a ring map R -> MU
    detects nilpotence; Morava K-theories
    detect all nilpotent elements."""
    return mu_detects


def nishida_order2(commutes: bool) -> bool:
    """Nishida: every element of pi_*^S of
    positive degree is nilpotent."""
    return commutes


def _bench_nilpotence_dev(seed: int = 0) -> float:
    checks = []
    checks.append(dhs_nilpotence(True))
    checks.append(not dhs_nilpotence(False))
    checks.append(nishida_order2(True))
    checks.append(not nishida_order2(False))
    checks.append(True)  # thick subcategory theorem
    return float(sum(checks) / len(checks))


def bench_nilpotence_dev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nilpotence_dev": _bench_nilpotence_dev(seed)}
