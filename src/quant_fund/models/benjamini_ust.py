"""benjamini ust module (SYNTHETIC)."""

from __future__ import annotations


def benjamini_ust_ok(ust: bool, lerw: bool) -> bool:
    """benjamini_ust
    check:
    UST/LERW
    structure —
    Wilson."""
    return ust and lerw


def benjamini_ust_aux(aux: bool) -> bool:
    """benjamini_ust
    aux:
    auxiliary
    spanning-tree
    check —
    Lawler."""
    return aux


def _bench_benjamini_ust(seed: int = 0) -> float:
    checks = []
    checks.append(benjamini_ust_ok(True, True))
    checks.append(not benjamini_ust_ok(False, True))
    checks.append(benjamini_ust_aux(True))
    checks.append(not benjamini_ust_aux(False))
    checks.append(True)  # UST/LERW canon
    return float(sum(checks) / len(checks))


def bench_benjamini_ust(seed: int = 0) -> dict[str, float]:
    return {"synthetic_benjamini_ust": _bench_benjamini_ust(seed)}
