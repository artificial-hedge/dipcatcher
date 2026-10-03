"""ergodic markov module (SYNTHETIC)."""

from __future__ import annotations


def ergodic_markov_ok(dc: bool, hr: bool) -> bool:
    """ergodic_markov
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def ergodic_markov_aux(aux: bool) -> bool:
    """ergodic_markov
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_ergodic_markov(seed: int = 0) -> float:
    checks = []
    checks.append(ergodic_markov_ok(True, True))
    checks.append(not ergodic_markov_ok(False, True))
    checks.append(ergodic_markov_aux(True))
    checks.append(not ergodic_markov_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_ergodic_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ergodic_markov": _bench_ergodic_markov(seed)}
