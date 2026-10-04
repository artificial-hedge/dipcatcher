"""Associative operad: Assoc(n) = S_n, composition = shuffle (SYNTHETIC)."""

from __future__ import annotations

from itertools import permutations


def assoc_n(n: int) -> list[tuple[int, ...]]:
    """Assoc(n) is the symmetric group: all orderings of n inputs."""
    return list(permutations(range(n)))


def assoc_compose(outer: tuple[int, ...], inners: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Operad composition: plug the i-th inner permutation's block into the
    position marked i in the outer permutation, then flatten to a single
    ordering by renaming blocks in outer order."""
    blocks: dict[int, tuple[int, ...]] = {i: inners[i] for i in range(len(inners))}
    out: list[int] = []
    offsets: dict[int, int] = {}
    total = 0
    for i in range(len(inners)):
        offsets[i] = total
        total += len(inners[i])
    for slot in outer:
        for j in blocks[slot]:
            out.append(offsets[slot] + j)
    return tuple(out)


def _bench_operad_assoc(seed: int = 0) -> float:
    checks = []
    # |Assoc(n)| = n!
    checks.append(len(assoc_n(3)) == 6 and len(assoc_n(4)) == 24)
    # identity: 1 in Assoc(1) acts as unit
    checks.append(assoc_compose((0,), [(0, 1, 2)]) == (0, 1, 2))
    # plug two blocks into the 2-ary operation ordered (0,1):
    # outer (0,1), inners [(1,0), (0,1)] -> block0 reversed then block1
    checks.append(assoc_compose((0, 1), [(1, 0), (0, 1)]) == (1, 0, 2, 3))
    # reversed outer (1,0): block1 first
    checks.append(assoc_compose((1, 0), [(1, 0), (0, 1)]) == (2, 3, 1, 0))
    # associativity of composition on small case: m o (m, m) o structures
    # coincide: (m o_1 m) o_2 m == m o_1 (m o_2 m)
    m = (0, 1)
    lhs = assoc_compose(assoc_compose(m, [m, (0,)]), [m, m, (0,)])
    rhs = assoc_compose(m, [assoc_compose(m, [m, m]), (0,)])
    checks.append(lhs == rhs)
    # symmetric group action: relabeling commutes with composition sanity
    checks.append(sorted(assoc_compose((0, 1, 2), [(0, 1), (0, 1), (0, 1)])) == [0, 1, 2, 3, 4, 5])
    return float(sum(checks) / len(checks))


def bench_operad_assoc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_assoc": _bench_operad_assoc(seed)}
