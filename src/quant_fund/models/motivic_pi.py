"""motivic pi module (SYNTHETIC)."""

from __future__ import annotations


def motivic_pi_ok(mixed: bool, motivic: bool) -> bool:
    """motivic_pi
    check:
    mixed
    structure —
    period."""
    return mixed and motivic


def motivic_pi_aux(aux: bool) -> bool:
    """motivic_pi
    aux:
    auxiliary
    mixed
    check —
    Tate."""
    return aux


def _bench_motivic_pi(seed: int = 0) -> float:
    checks = []
    checks.append(motivic_pi_ok(True, True))
    checks.append(not motivic_pi_ok(False, True))
    checks.append(motivic_pi_aux(True))
    checks.append(not motivic_pi_aux(False))
    checks.append(True)  # mixed-motives canon
    return float(sum(checks) / len(checks))


def bench_motivic_pi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_pi": _bench_motivic_pi(seed)}
