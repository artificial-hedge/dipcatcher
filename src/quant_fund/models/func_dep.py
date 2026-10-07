"""Functional dependency discovery (TANE-lite): A→B iff partition by A (SYNTHETIC)
refines B."""

import itertools

import numpy as np

_SEED = 20261231 + 603


def _holds(rows: np.ndarray, lhs: tuple[int, ...], rhs: int) -> bool:
    seen: dict[tuple, object] = {}
    for row in rows:
        key = tuple(row[i] for i in lhs)
        val = row[rhs]
        if key in seen and seen[key] != val:
            return False
        seen[key] = val
    return True


def discover_fds(rows: np.ndarray) -> set[tuple[tuple[int, ...], int]]:
    ncols = rows.shape[1]
    fds: set[tuple[tuple[int, ...], int]] = set()
    for rhs in range(ncols):
        for k in range(1, 3):
            for lhs in itertools.combinations([c for c in range(ncols) if c != rhs], k):
                minimal = not any(
                    set(sub) < set(lhs) and (sub, rhs) in fds
                    for sub in itertools.combinations(lhs, k - 1)
                )
                if _holds(rows, lhs, rhs) and minimal:
                    fds.add((lhs, rhs))
    return fds


def bench_func_dep(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(25):
        k_rows = 80
        # plant: col1 -> col2 (map via dict), col0,col1 -> col3
        m12 = {v: v % 3 for v in range(10)}
        c0 = rng.randint(0, 5, k_rows)
        c1 = rng.randint(0, 10, k_rows)
        c2 = np.array([m12[v] for v in c1])
        c3 = (c0 * 2 + c1) % 7
        rows = np.stack([c0, c1, c2, c3], axis=1)
        fds = discover_fds(rows)
        if ((1,), 2) in fds and ((0, 1), 3) in fds:
            ok += 1
    return {"synthetic_fd_found": ok / 25}
