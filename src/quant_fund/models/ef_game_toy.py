"""Ehrenfeucht-Fraisse games for elementary equivalence (SYNTHETIC)."""

from __future__ import annotations


def ef_winner_linord(a: list[int], b: list[int], rounds: int) -> bool:
    """Duplicator wins k rounds between finite linear orders iff
    min(|A|, |B|) < 2^k - 1 fails... standard result: orders of
    length >= 2^k - 1 are k-equivalent; otherwise need |A| = |B|."""
    if len(a) == len(b):
        return True
    return min(len(a), len(b)) >= (1 << rounds) - 1


def _bench_ef_game_toy(seed: int = 0) -> float:
    checks = []
    # orders 3 vs 5 with 2 rounds: min=3 >= 3 -> duplicator wins
    checks.append(ef_winner_linord(list(range(3)), list(range(5)), 2))
    # orders 2 vs 5 with 2 rounds: 2 < 3 -> spoiler wins
    checks.append(not ef_winner_linord(list(range(2)), list(range(5)), 2))
    # equal length always wins for duplicator
    checks.append(ef_winner_linord(list(range(7)), list(range(7)), 1))
    # with 3 rounds need >= 7
    checks.append(ef_winner_linord(list(range(7)), list(range(9)), 3))
    checks.append(not ef_winner_linord(list(range(6)), list(range(9)), 3))
    return float(sum(checks) / len(checks))


def bench_ef_game_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ef_game_toy": _bench_ef_game_toy(seed)}
