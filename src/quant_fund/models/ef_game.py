"""Ehrenfeucht–Fraïssé games: duplicator wins => elementary equivalence (SYNTHETIC)."""

from __future__ import annotations


def is_partial_iso(
    a_map: dict[int, int],
    rel_a: dict[int, set[int]],
    rel_b: dict[int, set[int]],
    pred_a: set[int],
    pred_b: set[int],
) -> bool:
    """Preserve membership in unary pred + edge relation both ways."""
    for x, y in a_map.items():
        if (x in pred_a) != (y in pred_b):
            return False
    for x1, y1 in a_map.items():
        for x2, y2 in a_map.items():
            if (x2 in rel_a.get(x1, set())) != (y2 in rel_b.get(y1, set())):
                return False
    return True


def duplicator_wins(
    dom_a: list[int], dom_b: list[int], rel_a, rel_b, pred_a, pred_b, rounds: int
) -> bool:
    """Recursive game: spoiler picks element, duplicator responds preserving partial iso."""

    def rec(mapping: dict[int, int], used_b: set[int], r: int) -> bool:
        if r == 0:
            return True
        # spoiler moves on A side
        for x in dom_a:
            if x in mapping:
                continue
            ok = False
            for y in dom_b:
                if y in used_b:
                    continue
                if is_partial_iso({**mapping, x: y}, rel_a, rel_b, pred_a, pred_b) and rec(
                    {**mapping, x: y}, used_b | {y}, r - 1
                ):
                    ok = True
                    break
            if not ok:
                return False
        # spoiler moves on B side
        for y in dom_b:
            if y in used_b:
                continue
            ok = False
            for x in dom_a:
                if x in mapping:
                    continue
                if is_partial_iso({**mapping, x: y}, rel_a, rel_b, pred_a, pred_b) and rec(
                    {**mapping, x: y}, used_b | {y}, r - 1
                ):
                    ok = True
                    break
            if not ok:
                return False
        return True

    return rec({}, set(), rounds)


def _bench_ef_game(seed: int = 0) -> float:
    checks = []
    # two chains of length 2: isomorphic -> duplicator wins any rounds
    rel: dict[int, set[int]] = {0: {1}, 1: set()}
    rel2: dict[int, set[int]] = {0: {1}, 1: set()}
    checks.append(duplicator_wins([0, 1], [0, 1], rel, rel2, set(), set(), 3))
    # chain vs non-chain: spoiler wins in 2 rounds
    rel3: dict[int, set[int]] = {0: set(), 1: set()}
    checks.append(not duplicator_wins([0, 1], [0, 1], rel, rel3, set(), set(), 2))
    # different unary preds: spoiler wins 1 round
    checks.append(not duplicator_wins([0], [0], {0: set()}, {0: set()}, {0}, set(), 1))
    # same preds: duplicator wins
    checks.append(duplicator_wins([0], [0], {0: set()}, {0: set()}, {0}, {0}, 1))
    return float(sum(checks) / len(checks))


def bench_ef_game(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ef_game": _bench_ef_game(seed)}
