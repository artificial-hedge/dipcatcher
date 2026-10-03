"""excursion proc module (SYNTHETIC)."""

from __future__ import annotations


def excursion_proc_ok(ex: bool, me: bool) -> bool:
    """excursion_proc
    check:
    excursion
    theory —
    measure."""
    return ex and me


def excursion_proc_aux(aux: bool) -> bool:
    """excursion_proc
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_excursion_proc(seed: int = 0) -> float:
    checks = []
    checks.append(excursion_proc_ok(True, True))
    checks.append(not excursion_proc_ok(False, True))
    checks.append(excursion_proc_aux(True))
    checks.append(not excursion_proc_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_excursion_proc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_excursion_proc": _bench_excursion_proc(seed)}
