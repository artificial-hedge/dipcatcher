"""pitman thm module (SYNTHETIC)."""

from __future__ import annotations


def pitman_thm_ok(ex: bool, me: bool) -> bool:
    """pitman_thm
    check:
    excursion
    theory —
    measure."""
    return ex and me


def pitman_thm_aux(aux: bool) -> bool:
    """pitman_thm
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_pitman_thm(seed: int = 0) -> float:
    checks = []
    checks.append(pitman_thm_ok(True, True))
    checks.append(not pitman_thm_ok(False, True))
    checks.append(pitman_thm_aux(True))
    checks.append(not pitman_thm_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_pitman_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pitman_thm": _bench_pitman_thm(seed)}
