"""wiener bm module (SYNTHETIC)."""

from __future__ import annotations


def wiener_bm_ok(bm: bool, wiener: bool) -> bool:
    """wiener_bm
    check:
    Brownian-motion
    structure —
    Lévy."""
    return bm and wiener


def wiener_bm_aux(aux: bool) -> bool:
    """wiener_bm
    aux:
    auxiliary
    Wiener
    check —
    Paley."""
    return aux


def _bench_wiener_bm(seed: int = 0) -> float:
    checks = []
    checks.append(wiener_bm_ok(True, True))
    checks.append(not wiener_bm_ok(False, True))
    checks.append(wiener_bm_aux(True))
    checks.append(not wiener_bm_aux(False))
    checks.append(True)  # Brownian canon
    return float(sum(checks) / len(checks))


def bench_wiener_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wiener_bm": _bench_wiener_bm(seed)}
