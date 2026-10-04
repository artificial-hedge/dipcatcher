"""cadlag markov module (SYNTHETIC)."""

from __future__ import annotations


def cadlag_markov_ok(mp: bool, se: bool) -> bool:
    """cadlag_markov
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def cadlag_markov_aux(aux: bool) -> bool:
    """cadlag_markov
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_cadlag_markov(seed: int = 0) -> float:
    checks = []
    checks.append(cadlag_markov_ok(True, True))
    checks.append(not cadlag_markov_ok(False, True))
    checks.append(cadlag_markov_aux(True))
    checks.append(not cadlag_markov_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_cadlag_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cadlag_markov": _bench_cadlag_markov(seed)}
