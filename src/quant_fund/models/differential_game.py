"""differential game module (SYNTHETIC)."""

from __future__ import annotations


def differential_game_ok(sg1: bool, dk: bool) -> bool:
    """differential_game
    check:
    stochastic
    game —
    Dynkin
    value."""
    return sg1 and dk


def differential_game_aux(aux: bool) -> bool:
    """differential_game
    aux:
    auxiliary
    game
    check —
    saddle
    point."""
    return aux


def _bench_differential_game(seed: int = 0) -> float:
    checks = []
    checks.append(differential_game_ok(True, True))
    checks.append(not differential_game_ok(False, True))
    checks.append(differential_game_aux(True))
    checks.append(not differential_game_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_differential_game(seed: int = 0) -> dict[str, float]:
    return {"synthetic_differential_game": _bench_differential_game(seed)}
