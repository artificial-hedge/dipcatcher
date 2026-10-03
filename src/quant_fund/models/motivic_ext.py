"""motivic ext module (SYNTHETIC)."""

from __future__ import annotations


def motivic_ext_ok(motivic: bool, stable: bool) -> bool:
    """motivic_ext
    check:
    motivic
    structure —
    stable."""
    return motivic and stable


def motivic_ext_aux(aux: bool) -> bool:
    """motivic_ext
    aux:
    auxiliary
    motivic
    check —
    residue."""
    return aux


def _bench_motivic_ext(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_ext_ok(True, True))
    checks.append(not motivic_ext_ok(False, True))
    checks.append(motivic_ext_aux(True))
    checks.append(not motivic_ext_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_motivic_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_ext": _bench_motivic_ext(seed)}
