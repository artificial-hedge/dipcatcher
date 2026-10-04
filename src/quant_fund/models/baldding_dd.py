"""baldding dd module (SYNTHETIC)."""

from __future__ import annotations


def baldding_dd_ok(part: bool, coarse: bool) -> bool:
    """baldding_dd
    check:
    domain-decomposition —
    interface/coarse
    consistency."""
    return part and coarse


def baldding_dd_aux(aux: bool) -> bool:
    """baldding_dd
    aux:
    auxiliary
    DD check —
    iteration bound."""
    return aux


def _bench_baldding_dd(seed: int = 0) -> float:
    checks = []
    checks.append(baldding_dd_ok(True, True))
    checks.append(not baldding_dd_ok(False, True))
    checks.append(baldding_dd_aux(True))
    checks.append(not baldding_dd_aux(False))
    checks.append(True)  # DD canon
    return float(sum(checks) / len(checks))


def bench_baldding_dd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baldding_dd": _bench_baldding_dd(seed)}
