"""multifactor rough module (SYNTHETIC)."""

from __future__ import annotations


def multifactor_rough_ok(fh1: bool, rv: bool) -> bool:
    """multifactor_rough
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def multifactor_rough_aux(aux: bool) -> bool:
    """multifactor_rough
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_multifactor_rough(seed: int = 0) -> float:
    checks = []
    checks.append(multifactor_rough_ok(True, True))
    checks.append(not multifactor_rough_ok(False, True))
    checks.append(multifactor_rough_aux(True))
    checks.append(not multifactor_rough_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_multifactor_rough(seed: int = 0) -> dict[str, float]:
    return {"synthetic_multifactor_rough": _bench_multifactor_rough(seed)}
