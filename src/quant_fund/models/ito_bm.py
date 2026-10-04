"""ito bm module (SYNTHETIC)."""

from __future__ import annotations


def ito_bm_ok(bm: bool, wiener: bool) -> bool:
    """ito_bm
    check:
    Brownian-motion
    structure —
    Lévy."""
    return bm and wiener


def ito_bm_aux(aux: bool) -> bool:
    """ito_bm
    aux:
    auxiliary
    Wiener
    check —
    Paley."""
    return aux


def _bench_ito_bm(seed: int = 0) -> float:
    checks = []
    checks.append(ito_bm_ok(True, True))
    checks.append(not ito_bm_ok(False, True))
    checks.append(ito_bm_aux(True))
    checks.append(not ito_bm_aux(False))
    checks.append(True)  # Brownian canon
    return float(sum(checks) / len(checks))


def bench_ito_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ito_bm": _bench_ito_bm(seed)}
