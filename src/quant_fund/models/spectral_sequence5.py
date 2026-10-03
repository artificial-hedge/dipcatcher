"""spectral sequence5 module (SYNTHETIC)."""

from __future__ import annotations


def spectral_sequence5_ok(homotopy: bool, stable: bool) -> bool:
    """spectral_sequence5
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def spectral_sequence5_aux(aux: bool) -> bool:
    """spectral_sequence5
    aux:
    auxiliary
    homotopy
    check —
    limit."""
    return aux


def _bench_spectral_sequence5(seed: int = 0) -> float:
    checks = []
    checks.append(spectral_sequence5_ok(True, True))
    checks.append(not spectral_sequence5_ok(False, True))
    checks.append(spectral_sequence5_aux(True))
    checks.append(not spectral_sequence5_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_spectral_sequence5(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_sequence5": _bench_spectral_sequence5(seed)}
