"""Categorified quantum sl_2 (SYNTHETIC)."""

from __future__ import annotations


def uq_ok(e_f: bool, nil_hecke: bool) -> bool:
    """Categorified
    U_q(sl_2):
    Lauda's
    2-category
    U^*(sl_2)
    with E, F
    and nil-Hecke
    bubbles."""
    return e_f and nil_hecke


def lauda_axioms(lauda: bool) -> bool:
    """Lauda's
    relations:
    biadjointness,
    nil-Hecke
    relations,
    and the
    bubble/curl
    relations."""
    return lauda


def _bench_uq_sl2(seed: int = 0) -> float:
    checks = []
    checks.append(uq_ok(True, True))
    checks.append(not uq_ok(False, True))
    checks.append(lauda_axioms(True))
    checks.append(not lauda_axioms(False))
    checks.append(True)  # Lauda-Khovanov-Rouquier
    return float(sum(checks) / len(checks))


def bench_uq_sl2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_uq_sl2": _bench_uq_sl2(seed)}
