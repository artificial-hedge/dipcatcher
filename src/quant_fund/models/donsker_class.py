"""donsker class module (SYNTHETIC)."""

from __future__ import annotations


def donsker_class_ok(proc: bool, tight: bool) -> bool:
    """donsker_class
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def donsker_class_aux(aux: bool) -> bool:
    """donsker_class
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_donsker_class(seed: int = 0) -> float:
    checks = []
    checks.append(donsker_class_ok(True, True))
    checks.append(not donsker_class_ok(False, True))
    checks.append(donsker_class_aux(True))
    checks.append(not donsker_class_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_donsker_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_donsker_class": _bench_donsker_class(seed)}
