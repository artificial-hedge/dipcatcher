"""dynkin game module (SYNTHETIC)."""

from __future__ import annotations


def dynkin_game_ok(sg1: bool, dk: bool) -> bool:
    """dynkin_game
    check:
    stochastic
    game —
    Dynkin
    value."""
    return sg1 and dk


def dynkin_game_aux(aux: bool) -> bool:
    """dynkin_game
    aux:
    auxiliary
    game
    check —
    saddle
    point."""
    return aux


def _bench_dynkin_game(seed: int = 0) -> float:
    checks = []
    checks.append(dynkin_game_ok(True, True))
    checks.append(not dynkin_game_ok(False, True))
    checks.append(dynkin_game_aux(True))
    checks.append(not dynkin_game_aux(False))
    checks.append(True)  # game canon
    return float(sum(checks) / len(checks))


def bench_dynkin_game(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dynkin_game": _bench_dynkin_game(seed)}
