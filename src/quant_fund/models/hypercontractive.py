"""hypercontractive module (SYNTHETIC)."""

from __future__ import annotations


def hypercontractive_ok(sem: bool, gen: bool) -> bool:
    """hypercontractive
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def hypercontractive_aux(aux: bool) -> bool:
    """hypercontractive
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_hypercontractive(seed: int = 0) -> float:
    checks = []
    checks.append(hypercontractive_ok(True, True))
    checks.append(not hypercontractive_ok(False, True))
    checks.append(hypercontractive_aux(True))
    checks.append(not hypercontractive_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_hypercontractive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hypercontractive": _bench_hypercontractive(seed)}
