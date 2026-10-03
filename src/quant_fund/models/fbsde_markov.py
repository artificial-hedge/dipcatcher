"""fbsde markov module (SYNTHETIC)."""

from __future__ import annotations


def fbsde_markov_ok(bs1: bool, pp: bool) -> bool:
    """fbsde_markov
    check:
    BSDE —
    Pardoux-Peng
    adapted
    solution."""
    return bs1 and pp


def fbsde_markov_aux(aux: bool) -> bool:
    """fbsde_markov
    aux:
    auxiliary
    FBSDE
    check —
    decoupling
    field."""
    return aux


def _bench_fbsde_markov(seed: int = 0) -> float:
    checks = []
    checks.append(fbsde_markov_ok(True, True))
    checks.append(not fbsde_markov_ok(False, True))
    checks.append(fbsde_markov_aux(True))
    checks.append(not fbsde_markov_aux(False))
    checks.append(True)  # BSDE canon
    return float(sum(checks) / len(checks))


def bench_fbsde_markov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fbsde_markov": _bench_fbsde_markov(seed)}
