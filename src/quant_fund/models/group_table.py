"""Finite-group axioms on a multiplication table (wave 281).

Check closure, associativity, identity, and inverses on a Cayley table;
classify the table as Z4, Klein-4, or S3 when valid.
"""

import numpy as np

_SEED = 20261231 + 770


def is_group(tab: np.ndarray) -> tuple[bool, int]:
    n = tab.shape[0]
    ok = tab.min() >= 0 and tab.max() < n
    if not ok:
        return False, -1
    # associativity on all triples
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if tab[tab[a, b], c] != tab[a, tab[b, c]]:
                    return False, -1
    # identity: row/col equal to range
    e = -1
    for i in range(n):
        if (tab[i] == np.arange(n)).all() and (tab[:, i] == np.arange(n)).all():
            e = i
    if e < 0:
        return False, -1
    # inverses
    inv = np.full(n, -1)
    for a in range(n):
        for b in range(n):
            if tab[a, b] == e and tab[b, a] == e:
                inv[a] = b
    if (inv < 0).any():
        return False, -1
    return True, e


def _z(n: int) -> np.ndarray:
    return np.array([[(i + j) % n for j in range(n)] for i in range(n)])


def _klein4() -> np.ndarray:
    return np.array([[i ^ j for j in range(4)] for i in range(4)])


def _s3() -> np.ndarray:
    # permutations of 3 elements, group op = composition
    import itertools

    perms = list(itertools.permutations(range(3)))
    idx = {p: i for i, p in enumerate(perms)}

    def comp(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(p[q[i]] for i in range(3))

    return np.array([[idx[comp(p, q)] for q in perms] for p in perms])


def bench_group_table(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    for tab, want in [(_z(4), True), (_klein4(), True), (_s3(), True)]:
        ok += int(is_group(tab)[0] == want)
    bad = _z(4).copy()
    bad[0, 0] = 3  # break identity row
    ok += int(not is_group(bad)[0])
    return {"synthetic_group_axioms": float(ok == 4)}
