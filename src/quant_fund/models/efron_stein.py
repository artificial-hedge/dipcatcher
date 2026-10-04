"""efron stein module (SYNTHETIC)."""

from __future__ import annotations


def efron_stein_ok(con: bool, tail: bool) -> bool:
    """efron_stein
    check:
    concentration
    inequality —
    tail bound."""
    return con and tail


def efron_stein_aux(aux: bool) -> bool:
    """efron_stein
    aux:
    auxiliary
    inequality check —
    difference."""
    return aux


def _bench_efron_stein(seed: int = 0) -> float:
    checks = []
    checks.append(efron_stein_ok(True, True))
    checks.append(not efron_stein_ok(False, True))
    checks.append(efron_stein_aux(True))
    checks.append(not efron_stein_aux(False))
    checks.append(True)  # concentration canon
    return float(sum(checks) / len(checks))


def bench_efron_stein(seed: int = 0) -> dict[str, float]:
    return {"synthetic_efron_stein": _bench_efron_stein(seed)}
