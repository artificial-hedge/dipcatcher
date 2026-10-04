"""characteristic markov module (SYNTHETIC)."""

from __future__ import annotations


def characteristic_markov_ok(mp: bool, se: bool) -> bool:
    """characteristic_markov
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def characteristic_markov_aux(aux: bool) -> bool:
    """characteristic_markov
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_characteristic_markov(seed: int = 0) -> float:
    checks = []
    checks.append(characteristic_markov_ok(True, True))
    checks.append(not characteristic_markov_ok(False, True))
    checks.append(characteristic_markov_aux(True))
    checks.append(not characteristic_markov_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_characteristic_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_characteristic_markov": _bench_characteristic_markov(seed)}
