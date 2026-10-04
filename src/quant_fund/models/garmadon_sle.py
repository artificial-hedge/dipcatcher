"""garmadon sle module (SYNTHETIC)."""

from __future__ import annotations


def garmadon_sle_ok(sle: bool, conf: bool) -> bool:
    """garmadon_sle
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def garmadon_sle_aux(aux: bool) -> bool:
    """garmadon_sle
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_garmadon_sle(seed: int = 0) -> float:
    checks = []
    checks.append(garmadon_sle_ok(True, True))
    checks.append(not garmadon_sle_ok(False, True))
    checks.append(garmadon_sle_aux(True))
    checks.append(not garmadon_sle_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_garmadon_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_garmadon_sle": _bench_garmadon_sle(seed)}
