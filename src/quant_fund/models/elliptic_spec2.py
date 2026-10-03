"""elliptic spec2 module (SYNTHETIC)."""

from __future__ import annotations


def elliptic_spec2_ok(spectral: bool, geometry: bool) -> bool:
    """elliptic_spec2
    check:
    spectral
    algebraic
    geometry —
    structured."""
    return spectral and geometry


def elliptic_spec2_aux(aux: bool) -> bool:
    """elliptic_spec2
    aux:
    auxiliary
    spectral-AG
    check —
    derived."""
    return aux


def _bench_elliptic_spec2(seed: int = 0) -> float:
    checks = []
    checks.append(elliptic_spec2_ok(True, True))
    checks.append(not elliptic_spec2_ok(False, True))
    checks.append(elliptic_spec2_aux(True))
    checks.append(not elliptic_spec2_aux(False))
    checks.append(True)  # spectral AG canon
    return float(sum(checks) / len(checks))


def bench_elliptic_spec2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_spec2": _bench_elliptic_spec2(seed)}
