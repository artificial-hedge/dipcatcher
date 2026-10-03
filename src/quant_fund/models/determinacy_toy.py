"""Gale-Stewart determinacy on finite game trees (SYNTHETIC)."""

from __future__ import annotations


def minimax_winner(
    payoff: dict[tuple[int, ...], int],
    branching: int,
    depth: int,
) -> int:
    """Finite perfect-information game: player I wins iff minimax = 1.
    payoff maps terminal histories (tuples of length depth) to {0,1}."""

    def val(hist: tuple[int, ...], player: int) -> int:
        if len(hist) == depth:
            return payoff.get(hist, 0)
        nxt = [val(hist + (m,), 1 - player) for m in range(branching)]
        return max(nxt) if player == 1 else min(nxt)

    return val((), 1)


def winning_strategy_exists(
    payoff: dict[tuple[int, ...], int], branching: int, depth: int, player: int
) -> bool:
    return minimax_winner(payoff, branching, depth) == (1 if player == 1 else 0)


def _bench_determinacy_toy(seed: int = 0) -> float:
    checks = []
    # game: I wins iff the terminal contains a 0
    from itertools import product

    payoff = {h: (1 if 0 in h else 0) for h in product((0, 1), repeat=3)}
    checks.append(minimax_winner(payoff, 2, 3) == 1)  # I plays 0 immediately
    # I wins iff all moves are 1: I cannot force all-1s (II plays 0)
    payoff2 = {h: (1 if all(m == 1 for m in h) else 0) for h in product((0, 1), repeat=3)}
    checks.append(minimax_winner(payoff2, 2, 3) == 0)
    # parity game: I wins iff sum of moves is even; I controls the last
    # move at depth 3 (I moves at positions 0 and 2) -> I wins
    payoff3 = {h: (1 if sum(h) % 2 == 0 else 0) for h in product((0, 1), repeat=3)}
    checks.append(minimax_winner(payoff3, 2, 3) == 1)
    checks.append(winning_strategy_exists(payoff3, 2, 3, 1))
    # with depth 2, II moves last and can always flip parity -> II wins
    payoff3b = {h: (1 if sum(h) % 2 == 0 else 0) for h in product((0, 1), repeat=2)}
    checks.append(minimax_winner(payoff3b, 2, 2) == 0)
    # depth-1: I chooses a winning leaf if any exists
    payoff4: dict[tuple[int, ...], int] = {(0,): 1, (1,): 0}
    checks.append(minimax_winner(payoff4, 2, 1) == 1)
    # determinacy: value is 0 or 1, never intermediate
    checks.append(minimax_winner(payoff, 2, 3) in (0, 1))
    return float(sum(checks) / len(checks))


def bench_determinacy_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_determinacy_toy": _bench_determinacy_toy(seed)}
