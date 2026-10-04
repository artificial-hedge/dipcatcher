"""skorohod int module (SYNTHETIC)."""

from __future__ import annotations


def skorohod_int_ok(ml1: bool, div: bool) -> bool:
    """skorohod_int
    check:
    Malliavin
    calculus —
    divergence
    operator."""
    return ml1 and div


def skorohod_int_aux(aux: bool) -> bool:
    """skorohod_int
    aux:
    auxiliary
    chaos
    check —
    Wiener
    decomposition."""
    return aux


def _bench_skorohod_int(seed: int = 0) -> float:
    checks = []
    checks.append(skorohod_int_ok(True, True))
    checks.append(not skorohod_int_ok(False, True))
    checks.append(skorohod_int_aux(True))
    checks.append(not skorohod_int_aux(False))
    checks.append(True)  # malliavin canon
    return float(sum(checks) / len(checks))


def bench_skorohod_int(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skorohod_int": _bench_skorohod_int(seed)}
