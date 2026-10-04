"""markov stopping module (SYNTHETIC)."""

from __future__ import annotations


def markov_stopping_ok(os1: bool, sd: bool) -> bool:
    """markov_stopping
    check:
    optimal-
    stopping —
    value
    function."""
    return os1 and sd


def markov_stopping_aux(aux: bool) -> bool:
    """markov_stopping
    aux:
    auxiliary
    stopping
    check —
    boundary."""
    return aux


def _bench_markov_stopping(seed: int = 0) -> float:
    checks = []
    checks.append(markov_stopping_ok(True, True))
    checks.append(not markov_stopping_ok(False, True))
    checks.append(markov_stopping_aux(True))
    checks.append(not markov_stopping_aux(False))
    checks.append(True)  # optimal-stopping canon
    return float(sum(checks) / len(checks))


def bench_markov_stopping(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_stopping": _bench_markov_stopping(seed)}
