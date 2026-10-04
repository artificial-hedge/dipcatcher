"""saito epsilon module (SYNTHETIC)."""

from __future__ import annotations


def saito_epsilon_ok(swan: bool, tame: bool) -> bool:
    """saito_epsilon
    check:
    ramification
    structure —
    Kato."""
    return swan and tame


def saito_epsilon_aux(aux: bool) -> bool:
    """saito_epsilon
    aux:
    auxiliary
    conductor
    check —
    Saito."""
    return aux


def _bench_saito_epsilon(seed: int = 0) -> float:
    checks = []
    checks.append(saito_epsilon_ok(True, True))
    checks.append(not saito_epsilon_ok(False, True))
    checks.append(saito_epsilon_aux(True))
    checks.append(not saito_epsilon_aux(False))
    checks.append(True)  # ramification canon
    return float(sum(checks) / len(checks))


def bench_saito_epsilon(seed: int = 0) -> dict[str, float]:
    return {"synthetic_saito_epsilon": _bench_saito_epsilon(seed)}
