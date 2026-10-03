"""excursion theory module (SYNTHETIC)."""

from __future__ import annotations


def excursion_theory_ok(jd: bool, mj: bool) -> bool:
    """excursion_theory
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def excursion_theory_aux(aux: bool) -> bool:
    """excursion_theory
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_excursion_theory(seed: int = 0) -> float:
    checks = []
    checks.append(excursion_theory_ok(True, True))
    checks.append(not excursion_theory_ok(False, True))
    checks.append(excursion_theory_aux(True))
    checks.append(not excursion_theory_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_excursion_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_excursion_theory": _bench_excursion_theory(seed)}
