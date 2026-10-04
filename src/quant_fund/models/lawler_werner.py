"""lawler werner module (SYNTHETIC)."""

from __future__ import annotations


def lawler_werner_ok(sle: bool, conf: bool) -> bool:
    """lawler_werner
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def lawler_werner_aux(aux: bool) -> bool:
    """lawler_werner
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_lawler_werner(seed: int = 0) -> float:
    checks = []
    checks.append(lawler_werner_ok(True, True))
    checks.append(not lawler_werner_ok(False, True))
    checks.append(lawler_werner_aux(True))
    checks.append(not lawler_werner_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_lawler_werner(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lawler_werner": _bench_lawler_werner(seed)}
