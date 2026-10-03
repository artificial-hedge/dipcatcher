"""SYNTHETIC cellular automata: elementary CA + Conway's Life.

Rule 90 on a single seed satisfies row popcount = 2^popcount(t) (Lucas
theorem); Life's block/beehive are still lifes, the blinker oscillates with
period 2, and the glider translates diagonally every 4 generations.
"""

from __future__ import annotations

import random


def eca_step(row: list[int], rule: int) -> list[int]:
    n = len(row)
    out = [0] * n
    for i in range(n):
        nb = (row[(i - 1) % n] << 2) | (row[i] << 1) | row[(i + 1) % n]
        out[i] = (rule >> nb) & 1
    return out


def life_step(grid: set[tuple[int, int]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for x, y in grid:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    counts[(x + dx, y + dy)] = counts.get((x + dx, y + dy), 0) + 1
    return {c for c, k in counts.items() if k == 3 or (k == 2 and c in grid)}


def _popcount(x: int) -> int:
    return bin(x).count("1")


def bench_cellular_automata(seed: int = 20261231 + 505) -> dict[str, float]:
    _ = random.Random(seed)
    # Rule 90 popcount = 2^{popcount(t)} on ring large enough no wrap
    r90 = True
    t_max = 64
    row = [0] * (2 * t_max + 3)
    row[t_max + 1] = 1
    for t in range(1, t_max):
        row = eca_step(row, 90)
        if sum(row) != 2 ** _popcount(t):
            r90 = False
            break
    # Life still lifes
    block = {(0, 0), (0, 1), (1, 0), (1, 1)}
    beehive = {(1, 0), (2, 0), (0, 1), (3, 1), (1, 2), (2, 2)}
    still = life_step(block) == block and life_step(beehive) == beehive
    # blinker period 2
    blinker = {(0, 0), (1, 0), (2, 0)}
    osc = life_step(life_step(blinker)) == blinker and life_step(blinker) != blinker
    # glider translates by (1,1) after 4 gens
    glider = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
    g = glider
    for _ in range(4):
        g = life_step(g)
    glid = g == {(x + 1, y + 1) for x, y in glider}
    return {
        "synthetic_rule90_lucas": float(r90),
        "synthetic_life_still_invariant": float(still),
        "synthetic_life_oscillator_glider": float(osc and glid),
    }
