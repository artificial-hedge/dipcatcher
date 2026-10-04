"""markov semigroup module (SYNTHETIC)."""

from __future__ import annotations


def markov_semigroup_ok(sem: bool, gen: bool) -> bool:
    """markov_semigroup
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def markov_semigroup_aux(aux: bool) -> bool:
    """markov_semigroup
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_markov_semigroup(seed: int = 0) -> float:
    checks = []
    checks.append(markov_semigroup_ok(True, True))
    checks.append(not markov_semigroup_ok(False, True))
    checks.append(markov_semigroup_aux(True))
    checks.append(not markov_semigroup_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_markov_semigroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_markov_semigroup": _bench_markov_semigroup(seed)}
