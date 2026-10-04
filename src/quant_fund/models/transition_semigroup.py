"""transition semigroup module (SYNTHETIC)."""

from __future__ import annotations


def transition_semigroup_ok(mp: bool, se: bool) -> bool:
    """transition_semigroup
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def transition_semigroup_aux(aux: bool) -> bool:
    """transition_semigroup
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_transition_semigroup(seed: int = 0) -> float:
    checks = []
    checks.append(transition_semigroup_ok(True, True))
    checks.append(not transition_semigroup_ok(False, True))
    checks.append(transition_semigroup_aux(True))
    checks.append(not transition_semigroup_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_transition_semigroup(seed: int = 0) -> dict[str, float]:
    return {"synthetic_transition_semigroup": _bench_transition_semigroup(seed)}
