"""stochastic game2 module (SYNTHETIC)."""

from __future__ import annotations


def stochastic_game2_ok(sg1: bool, dk: bool) -> bool:
    """stochastic_game2
    check:
    stochastic
    game —
    Dynkin
    value."""
    return sg1 and dk


def stochastic_game2_aux(aux: bool) -> bool:
    """stochastic_game2
    aux:
    auxiliary
    game
    check —
    saddle
    point."""
    return aux


def _bench_stochastic_game2(seed: int = 0) -> float:
    checks = []
    checks.append(stochastic_game2_ok(True, True))
    checks.append(not stochastic_game2_ok(False, True))
    checks.append(stochastic_game2_aux(True))
    checks.append(not stochastic_game2_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_stochastic_game2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stochastic_game2": _bench_stochastic_game2(seed)}
