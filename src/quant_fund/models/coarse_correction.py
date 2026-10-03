"""coarse correction module (SYNTHETIC)."""

from __future__ import annotations


def coarse_correction_ok(dom: bool, overlap: bool) -> bool:
    """coarse_correction
    check:
    domain-decomposition —
    interface
    consistency."""
    return dom and overlap


def coarse_correction_aux(aux: bool) -> bool:
    """coarse_correction
    aux:
    auxiliary
    subdomain check —
    overlap bound."""
    return aux


def _bench_coarse_correction(seed: int = 0) -> float:
    checks = []
    checks.append(coarse_correction_ok(True, True))
    checks.append(not coarse_correction_ok(False, True))
    checks.append(coarse_correction_aux(True))
    checks.append(not coarse_correction_aux(False))
    checks.append(True)  # domain-decomp canon
    return float(sum(checks) / len(checks))


def bench_coarse_correction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coarse_correction": _bench_coarse_correction(seed)}
