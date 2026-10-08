"""Mini separation-logic satisfiability + entailment over finite heaps (SYNTHETIC).

Formulas: ("emp",), ("mapsto", x, y-term), ("star", f, g), ("list", x),
("ptsto-chain", x, [y0,...,null]). Satisfaction is checked by heap
partitioning; entailment on small formulas is verified by enumerating
heaps over a finite address space. The frame rule is exercised by running
a mutating program restricted to a formula's footprint.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1013

Heap = dict[int, int | None]
Env = dict[str, int | None]


def _holds(f: Any, env: Env, heap: Heap) -> bool:
    tag = f[0]
    if tag == "emp":
        return not heap
    if tag == "mapsto":
        x, y = env.get(f[1]), f[2]
        yv = None if y == "null" else env.get(y, y if isinstance(y, int) else None)
        return len(heap) == 1 and heap.get(x) == yv  # type: ignore[arg-type]
    if tag == "star":
        # exists disjoint split H1,H2 with H1|=f1, H2|=f2
        keys = list(heap)
        for mask in range(1 << len(keys)):
            h1 = {k: heap[k] for i, k in enumerate(keys) if mask >> i & 1}
            h2 = {k: heap[k] for i, k in enumerate(keys) if not (mask >> i & 1)}
            if _holds(f[1], env, h1) and _holds(f[2], env, h2):
                return True
        return False
    if tag == "list":
        # list(x): x=null -> emp ; else x↦n * list(n)
        return _list_holds(env.get(f[1]), heap)
    raise ValueError(f"bad formula {f}")


def _list_holds(x: int | None, heap: Heap) -> bool:
    seen: set[int] = set()
    cur = x
    used: list[int] = []
    while cur is not None:
        if cur in seen or cur not in heap:
            return False
        seen.add(cur)
        used.append(cur)
        cur = heap[cur]
    return set(used) == set(heap)


def entails(f: Any, g: Any, addrs: list[int], env: Env) -> bool:
    """Enumerate all heaps over addrs (2-addr chain depth) and check f -> g."""
    vals: list[int | None] = [*addrs, None]
    heaps: list[Heap] = []
    for a in addrs:
        for v in vals:
            heaps.append({a: v})
    for a in addrs:
        for b in addrs:
            if a != b:
                for va in vals:
                    for vb in vals:
                        heaps.append({a: va, b: vb})
    heaps.append({})
    return all(not _holds(f, env, h) or _holds(g, env, h) for h in heaps)


def footprint(f: Any, env: Env) -> set[int]:
    tag = f[0]
    if tag == "mapsto":
        x = env.get(f[1])
        return {x} if x is not None else set()  # type: ignore[return-value]
    if tag == "star":
        return footprint(f[1], env) | footprint(f[2], env)
    return set()


def bench_separation_logic(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    env: Env = {"x": 1, "y": 2}
    f = ("star", ("mapsto", "x", "y"), ("mapsto", "y", "null"))
    checks.append(_holds(f, env, {1: 2, 2: None}))
    checks.append(not _holds(f, env, {1: 2}))  # needs both cells
    # list entailment: x->y->null |= list(x)
    checks.append(_holds(("list", "x"), env, {1: 2, 2: None}))
    # x↦y * y↦null |= list(x) by heap enumeration over {1,2}
    checks.append(entails(f, ("list", "x"), [1, 2], env))
    # frame rule: program mutating footprint of x↦y preserves y↦null
    heap = {1: 2, 2: None}
    fp = footprint(("mapsto", "x", "y"), env)
    heap[1] = 2  # mutation inside footprint only
    checks.append(
        _holds(("mapsto", "y", "null"), env, {k: v for k, v in heap.items() if k not in fp})
    )
    # disjointness: x↦y * y↦x is impossible on a 2-cell heap?  {1:2,2:1}
    checks.append(_holds(("star", ("mapsto", "x", "y"), ("mapsto", "y", "x")), env, {1: 2, 2: 1}))
    return {"synthetic_separation_logic": float(sum(checks)) / len(checks)}
