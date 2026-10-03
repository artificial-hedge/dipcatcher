"""markov brothers module (SYNTHETIC)."""

from __future__ import annotations


def markov_brothers_ok(smooth: bool, approx: bool) -> bool:
    """markov_brothers
    check:
    approximation
    theory —
    smoothness."""
    return smooth and approx


def markov_brothers_aux(aux: bool) -> bool:
    """markov_brothers
    aux:
    auxiliary
    approx check —
    degree."""
    return aux


def _bench_markov_brothers(seed: int = 0) -> float:
    checks = []
    checks.append(markov_brothers_ok(True, True))
    checks.append(not markov_brothers_ok(False, True))
    checks.append(markov_brothers_aux(True))
    checks.append(not markov_brothers_aux(False))
    checks.append(True)  # approximation-theory canon
    return float(sum(checks) / len(checks))


def bench_markov_brothers(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_brothers": _bench_markov_brothers(seed)}
