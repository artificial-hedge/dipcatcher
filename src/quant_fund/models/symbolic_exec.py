"""Symbolic execution: path-constraint exploration of a toy program (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 673


def sym_exec_paths(x0_range: tuple[int, int]) -> list[tuple[int, ...]]:
    """Program: if x>0: if x>5: y=1 else y=2 else: y=3. Enumerate concrete paths."""
    paths = []
    for x in range(x0_range[0], x0_range[1] + 1):
        if x > 0:
            y = 1 if x > 5 else 2
        else:
            y = 3
        paths.append((x, y))
    # merge equivalent outcomes
    seen: dict[int, list[int]] = {}
    for x, y in paths:
        seen.setdefault(y, []).append(x)
    return sorted((tuple(v) for v in seen.values()), key=len)


def bench_symbolic_exec(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    for _ in range(trials):
        lo, hi = sorted(rng.randint(-10, 15, 2))
        paths = sym_exec_paths((int(lo), int(hi)))
        # distinct outcomes = regions covered: {x<=0}, {0<x<=5}, {x>5}
        expected = int(lo <= 0 and True) + int(hi > 0 and lo <= 5 or (0 < lo <= 5)) + int(hi > 5)
        expected = int(lo <= 0) + int(hi >= 1 and lo <= 5) + int(hi >= 6)
        ok += float(len(paths) == expected)
    return {"synthetic_sym_paths": ok / trials}
