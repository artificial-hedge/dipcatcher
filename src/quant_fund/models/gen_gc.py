"""Generational garbage collector (nursery + old gen, promotion, write barrier)."""

import numpy as np

_SEED = 20261231 + 540


class _Obj:
    __slots__ = ("id", "refs", "gen")

    def __init__(self, i: int) -> None:
        self.id = i
        self.refs: list[int] = []
        self.gen = 0


class GenGC:
    def __init__(self) -> None:
        self.objs: dict[int, _Obj] = {}
        self.roots: set[int] = set()
        self.remembered: set[int] = set()  # old objects pointing to young
        self.promoted: set[int] = set()

    def alloc(self, i: int) -> _Obj:
        o = _Obj(i)
        self.objs[i] = o
        return o

    def set_ref(self, src: int, dst: int) -> None:
        s, d = self.objs[src], self.objs[dst]
        if dst not in s.refs:
            s.refs.append(dst)
        if s.gen == 1 and d.gen == 0:
            self.remembered.add(src)

    def _mark_from(self, starts: set[int], gen_max: int) -> set[int]:
        live: set[int] = set()
        stack = [i for i in starts if self.objs[i].gen <= gen_max]
        while stack:
            i = stack.pop()
            if i in live or self.objs[i].gen > gen_max:
                continue
            live.add(i)
            stack.extend(self.objs[i].refs)
        return live

    def minor_collect(self) -> None:
        # seed with young roots + young children of remembered old objects
        starts = {r for r in self.roots if self.objs[r].gen == 0}
        for r in self.remembered:
            if r in self.objs:
                starts.update(c for c in self.objs[r].refs if self.objs[c].gen == 0)
        live = self._mark_from(starts, 0)
        # survivors promote
        for i in live:
            self.objs[i].gen = 1
            self.promoted.add(i)
        for i in list(self.objs):
            if self.objs[i].gen == 0 and i not in live:
                del self.objs[i]
        self.remembered = {r for r in self.remembered if r in self.objs}

    def major_collect(self) -> None:
        live = self._mark_from(self.roots, 1)
        for i in list(self.objs):
            if i not in live:
                del self.objs[i]
        self.remembered &= set(self.objs)


def _build(gc: GenGC, rng: np.random.RandomState) -> tuple[list[int], list[int]]:
    """Returns (expected live young ids, expected live old ids)."""
    ids = iter(range(10_000))
    old_keep: list[int] = []
    # old gen: rooted chains
    for _ in range(6):
        chain = [next(ids) for _ in range(3)]
        for c in chain:
            gc.alloc(c)
            gc.objs[c].gen = 1
        for a, b in zip(chain, chain[1:], strict=False):
            gc.set_ref(a, b)
        gc.roots.add(chain[0])
        old_keep.extend(chain)
    # young reachable only via an old-gen object (write-barrier test)
    deep = [next(ids) for _ in range(4)]
    for d in deep:
        gc.alloc(d)
    for a, b in zip(deep, deep[1:], strict=False):
        gc.set_ref(a, b)
    gc.set_ref(old_keep[-1], deep[0])  # old → young edge
    # rooted young
    rooted = [next(ids) for _ in range(3)]
    for r in rooted:
        gc.alloc(r)
        gc.roots.add(r)
    # garbage young
    for _ in range(20):
        g = next(ids)
        gc.alloc(g)
        for _ in range(2):
            t = next(ids)
            gc.alloc(t)
            gc.set_ref(g, t)
    return deep + rooted, old_keep


def bench_gen_gc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    gc = GenGC()
    live_young, live_old = _build(gc, rng)
    n_before = len(gc.objs)
    gc.minor_collect()
    preserved = int(all(i in gc.objs for i in live_young))
    promoted_ok = int(all(gc.objs[i].gen == 1 for i in live_young if i in gc.objs))
    collected = n_before - len(gc.objs)
    # only the 60 garbage (20 heads + 40 targets) should be gone
    gc2 = GenGC()
    live_young2, live_old2 = _build(gc2, rng)
    gc2.roots -= set(live_young2[-3:])  # unroot the rooted-young block
    expected = gc2._mark_from(set(gc2.roots), 1)
    gc2.major_collect()
    major_ok = int(set(gc2.objs) == expected)
    return {
        "synthetic_preserved": float(preserved),
        "synthetic_promoted": float(promoted_ok),
        "synthetic_garbage_collected": float(collected == 60),
        "synthetic_major_preserves_reachable": float(major_ok),
    }
