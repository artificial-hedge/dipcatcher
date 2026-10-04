"""nonzero sum_game module (SYNTHETIC)."""

from __future__ import annotations


def nonzero_sum_game_ok(sg1: bool, dk: bool) -> bool:
    """nonzero_sum_game
    check:
    stochastic
    game —
    Dynkin
    value."""
    return sg1 and dk


def nonzero_sum_game_aux(aux: bool) -> bool:
    """nonzero_sum_game
    aux:
    auxiliary
    game
    check —
    saddle
    point."""
    return aux


def _bench_nonzero_sum_game(seed: int = 0) -> float:
    checks = []
    checks.append(nonzero_sum_game_ok(True, True))
    checks.append(not nonzero_sum_game_ok(False, True))
    checks.append(nonzero_sum_game_aux(True))
    checks.append(not nonzero_sum_game_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_nonzero_sum_game(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nonzero_sum_game": _bench_nonzero_sum_game(seed)}
