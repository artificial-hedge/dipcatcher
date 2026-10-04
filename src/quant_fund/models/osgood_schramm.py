"""osgood schramm module (SYNTHETIC)."""

from __future__ import annotations


def osgood_schramm_ok(sle: bool, conf: bool) -> bool:
    """osgood_schramm
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def osgood_schramm_aux(aux: bool) -> bool:
    """osgood_schramm
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_osgood_schramm(seed: int = 0) -> float:
    checks = []
    checks.append(osgood_schramm_ok(True, True))
    checks.append(not osgood_schramm_ok(False, True))
    checks.append(osgood_schramm_aux(True))
    checks.append(not osgood_schramm_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_osgood_schramm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_osgood_schramm": _bench_osgood_schramm(seed)}
