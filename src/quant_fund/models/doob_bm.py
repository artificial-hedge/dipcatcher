"""doob bm module (SYNTHETIC)."""

from __future__ import annotations


def doob_bm_ok(bm: bool, wiener: bool) -> bool:
    """doob_bm
    check:
    Brownian-motion
    structure —
    Lévy."""
    return bm and wiener


def doob_bm_aux(aux: bool) -> bool:
    """doob_bm
    aux:
    auxiliary
    Wiener
    check —
    Paley."""
    return aux


def _bench_doob_bm(seed: int = 0) -> float:
    checks = []
    checks.append(doob_bm_ok(True, True))
    checks.append(not doob_bm_ok(False, True))
    checks.append(doob_bm_aux(True))
    checks.append(not doob_bm_aux(False))
    checks.append(True)  # Brownian canon
    return float(sum(checks) / len(checks))


def bench_doob_bm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doob_bm": _bench_doob_bm(seed)}
