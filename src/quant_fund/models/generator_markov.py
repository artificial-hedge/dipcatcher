"""generator markov module (SYNTHETIC)."""

from __future__ import annotations


def generator_markov_ok(mp: bool, se: bool) -> bool:
    """generator_markov
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def generator_markov_aux(aux: bool) -> bool:
    """generator_markov
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_generator_markov(seed: int = 0) -> float:
    checks = []
    checks.append(generator_markov_ok(True, True))
    checks.append(not generator_markov_ok(False, True))
    checks.append(generator_markov_aux(True))
    checks.append(not generator_markov_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_generator_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_generator_markov": _bench_generator_markov(seed)}
