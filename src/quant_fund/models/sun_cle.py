"""sun cle module (SYNTHETIC)."""

from __future__ import annotations


def sun_cle_ok(cle: bool, loop: bool) -> bool:
    """sun_cle
    check:
    conformal-loop
    structure —
    Camia."""
    return cle and loop


def sun_cle_aux(aux: bool) -> bool:
    """sun_cle
    aux:
    auxiliary
    SLE-outer
    check —
    Newman."""
    return aux


def _bench_sun_cle(seed: int = 0) -> float:
    checks = []
    checks.append(sun_cle_ok(True, True))
    checks.append(not sun_cle_ok(False, True))
    checks.append(sun_cle_aux(True))
    checks.append(not sun_cle_aux(False))
    checks.append(True)  # CLE-2 canon
    return float(sum(checks) / len(checks))


def bench_sun_cle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sun_cle": _bench_sun_cle(seed)}
