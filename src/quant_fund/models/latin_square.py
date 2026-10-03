"""Latin-square validation and backtracking completion (wave 282).

A Latin square has each symbol exactly once per row and column. Cyclic
squares are Latin; a partial square is completed by constraint propagation
+ backtracking and validated against the Latin property.
"""

import numpy as np

_SEED = 20261231 + 781


def is_latin(grid: list[list[int]], n: int) -> bool:
    want = set(range(1, n + 1))
    for r in grid:
        if set(r) != want:
            return False
    return all({grid[r][c] for r in range(n)} == want for c in range(n))


def complete(grid: list[list[int]], n: int) -> list[list[int]] | None:
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                for v in range(1, n + 1):
                    if v not in grid[r] and all(grid[k][c] != v for k in range(n)):
                        grid[r][c] = v
                        if complete(grid, n):
                            return grid
                        grid[r][c] = 0
                return None
    return grid


def bench_latin_square(seed: int = _SEED) -> dict[str, float]:
    n = 5
    cyc = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]
    ok = int(is_latin(cyc, n))
    rng = np.random.RandomState(seed)
    partial = [row[:] for row in cyc]
    for _ in range(12):
        r, c = int(rng.randint(0, n)), int(rng.randint(0, n))
        partial[r][c] = 0
    done = complete([row[:] for row in partial], n)
    ok += int(done is not None and is_latin(done, n))
    bad = [row[:] for row in cyc]
    bad[0][0], bad[0][1] = bad[0][1], bad[0][0]
    ok += int(not is_latin(bad, n))
    return {"synthetic_latin": float(ok == 3)}
