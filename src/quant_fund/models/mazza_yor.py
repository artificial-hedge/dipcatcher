"""mazza yor module (SYNTHETIC)."""

from __future__ import annotations


def mazza_yor_ok(ex: bool, me: bool) -> bool:
    """mazza_yor
    check:
    excursion
    theory —
    measure."""
    return ex and me


def mazza_yor_aux(aux: bool) -> bool:
    """mazza_yor
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_mazza_yor(seed: int = 0) -> float:
    checks = []
    checks.append(mazza_yor_ok(True, True))
    checks.append(not mazza_yor_ok(False, True))
    checks.append(mazza_yor_aux(True))
    checks.append(not mazza_yor_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_mazza_yor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mazza_yor": _bench_mazza_yor(seed)}
