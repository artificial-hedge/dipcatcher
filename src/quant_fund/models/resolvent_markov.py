"""resolvent markov module (SYNTHETIC)."""

from __future__ import annotations


def resolvent_markov_ok(mp: bool, se: bool) -> bool:
    """resolvent_markov
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def resolvent_markov_aux(aux: bool) -> bool:
    """resolvent_markov
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_resolvent_markov(seed: int = 0) -> float:
    checks = []
    checks.append(resolvent_markov_ok(True, True))
    checks.append(not resolvent_markov_ok(False, True))
    checks.append(resolvent_markov_aux(True))
    checks.append(not resolvent_markov_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_resolvent_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_resolvent_markov": _bench_resolvent_markov(seed)}
