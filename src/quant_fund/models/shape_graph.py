"""Canonical shape graphs: concrete heap -> abstract shape graph.

Concrete heap cells carry "nxt" pointer; canonical embedding merges cells
with identical abstraction predicate vectors into summary nodes; the
resulting structure is checked for list-ness, garbage freedom, and merge
soundness (every concrete cell maps to exactly one abstract node).
"""

from __future__ import annotations

_SEED = 20261231 + 1012

Cell = dict[str, int | None]
Heap = dict[int, Cell]


def canonical_embed(heap: Heap, roots: list[int]) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Group concrete cells by unary-predicate vector (reachability from each
    root + sharing); returns (abstract nodes -> members, cell -> abstract id)."""
    n = len(heap)
    # unary abstraction predicates: r_i = reachable-from-root_i, shared = 2+ preds
    preds: dict[int, tuple[bool, ...]] = {}
    for c in heap:
        vec = tuple(c in _reach_set(heap, r) for r in roots)
        indeg = sum(1 for d in heap.values() if d.get("nxt") == c)
        preds[c] = (*vec, indeg > 1)
    groups: dict[tuple[bool, ...], list[int]] = {}
    for c, v in preds.items():
        groups.setdefault(v, []).append(c)
    abs_nodes = {i: mem for i, mem in enumerate(groups.values())}
    cell_to = {c: i for i, mem in abs_nodes.items() for c in mem}
    _ = n
    return abs_nodes, cell_to


def _reach_set(heap: Heap, root: int | None) -> set[int]:
    seen: set[int] = set()
    cur = root
    while cur is not None and cur in heap and cur not in seen:
        seen.add(cur)
        cur = heap[cur].get("nxt")
    return seen


def garbage(heap: Heap, roots: list[int]) -> set[int]:
    live: set[int] = set()
    for r in roots:
        live |= _reach_set(heap, r)
    return set(heap) - live


def is_list(heap: Heap, root: int) -> bool:
    seen: set[int] = set()
    cur: int | None = root
    while cur is not None and cur in heap:
        if cur in seen:
            return False
        seen.add(cur)
        cur = heap[cur].get("nxt")
    return True


def bench_shape_graph(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # concrete list 0->1->2->None, garbage cell 9
    heap: Heap = {0: {"nxt": 1}, 1: {"nxt": 2}, 2: {"nxt": None}, 9: {"nxt": None}}
    checks.append(is_list(heap, 0))
    checks.append(garbage(heap, [0]) == {9})
    # shared tail: 0->1, 3->1 (two roots share cell 1 only)
    heap2: Heap = {0: {"nxt": 1}, 1: {"nxt": 2}, 2: {"nxt": None}, 3: {"nxt": 1}}
    abs_nodes, cell_to = canonical_embed(heap2, [0, 3])
    # cell 1 is shared (indeg>1); cell 2 is not -> distinct abstraction vectors
    checks.append(cell_to[1] != cell_to[2])
    # partition covers the whole heap exactly
    checks.append(set().union(*map(set, abs_nodes.values())) == set(heap2))
    # single-root list: every cell has the identical vector -> all merge
    heap_u: Heap = {0: {"nxt": 1}, 1: {"nxt": 2}, 2: {"nxt": None}}
    abs_u, _ = canonical_embed(heap_u, [0])
    checks.append(len(abs_u) == 1)
    # cycle: 0->1->2->1
    heap3: Heap = {0: {"nxt": 1}, 1: {"nxt": 2}, 2: {"nxt": 1}}
    checks.append(not is_list(heap3, 0))
    checks.append(garbage(heap3, [0]) == set())
    return {"synthetic_shape_graph": float(sum(checks)) / len(checks)}
