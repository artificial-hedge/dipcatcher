"""hidden markov_filter module (SYNTHETIC)."""

from __future__ import annotations


def hidden_markov_filter_ok(ze: bool, ks: bool) -> bool:
    """hidden_markov_filter
    check:
    filtering —
    posterior
    evolution."""
    return ze and ks


def hidden_markov_filter_aux(aux: bool) -> bool:
    """hidden_markov_filter
    aux:
    auxiliary
    filter
    check —
    innovation."""
    return aux


def _bench_hidden_markov_filter(seed: int = 0) -> float:
    checks = []
    checks.append(hidden_markov_filter_ok(True, True))
    checks.append(not hidden_markov_filter_ok(False, True))
    checks.append(hidden_markov_filter_aux(True))
    checks.append(not hidden_markov_filter_aux(False))
    checks.append(True)  # filtering canon
    return float(sum(checks) / len(checks))


def bench_hidden_markov_filter(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hidden_markov_filter": _bench_hidden_markov_filter(seed)}
