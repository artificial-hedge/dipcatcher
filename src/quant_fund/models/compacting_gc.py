"""Semi-space copying collector (Cheney algorithm) with forwarding pointers (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 541


class CopyGC:
    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.from_space: dict[int, tuple[list[int], bool]] = {}  # id -> (refs, forwarded)
        self.to_space: dict[int, list[int]] = {}
        self.fwd: dict[int, int] = {}
        self.roots: list[int] = []

    def alloc(self, i: int) -> None:
        self.from_space[i] = ([], False)

    def collect(self) -> dict[int, list[int]]:
        self.to_space = {}
        self.fwd = {}
        scan: list[int] = []
        for r in self.roots:
            if r in self.from_space:
                new = self._copy(r)
                scan.append(new)
        self.roots = [self.fwd[r] for r in self.roots if r in self.fwd]
        # breadth-first: to-space objects are queued in copy order
        queue = list(self.to_space.keys())
        qi = 0
        while qi < len(queue):
            old_new = queue[qi]
            qi += 1
            orig_id = [k for k, v in self.fwd.items() if v == old_new][0]
            refs = self.from_space[orig_id][0]
            new_refs = [self._copy(rr) if rr in self.from_space else rr for rr in refs]
            self.to_space[old_new] = new_refs
            for rr in refs:
                if rr in self.fwd and self.fwd[rr] not in queue:
                    queue.append(self.fwd[rr])
        new_heap = {i: (refs, False) for i, refs in self.to_space.items()}
        self.from_space = new_heap
        self.to_space = {}
        return {i: list(refs) for i, (refs, _) in new_heap.items()}

    def _copy(self, i: int) -> int:
        if i in self.fwd:
            return self.fwd[i]
        new = len(self.to_space)
        self.fwd[i] = new
        self.to_space[new] = []
        return new


def _reachable(heap: dict[int, list[int]], roots: list[int]) -> set[int]:
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        i = stack.pop()
        if i in seen or i not in heap:
            continue
        seen.add(i)
        stack.extend(heap[i])
    return seen


def bench_compacting_gc(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = 60
    gc = CopyGC(n)
    edges: dict[int, list[int]] = {}
    for i in range(n):
        gc.alloc(i)
        edges[i] = []
    # random graph, roots = first 8 ids
    for i in range(n):
        for _ in range(rng.randint(0, 3)):
            j = rng.randint(n)
            if j not in edges[i]:
                edges[i].append(j)
                gc.from_space[i][0].append(j)
    gc.roots = list(range(8))
    live = _reachable({i: list(rr) for i, rr in edges.items()}, gc.roots)
    heap = gc.collect()
    # same reachable set under new ids
    new_live = _reachable(heap, gc.roots)
    size_ok = len(new_live) == len(live)
    # topology preserved: map old id -> new id, check edge images match
    topo_ok = True
    for old_i, new_i in gc.fwd.items():
        if new_i not in new_live:
            continue
        got = set(heap.get(new_i, []))
        want = {gc.fwd[j] for j in edges[old_i] if j in gc.fwd}
        if got != want:
            topo_ok = False
            break
    contig = int(sorted(new_live) == list(range(len(new_live))))
    return {
        "synthetic_live_preserved": float(size_ok),
        "synthetic_topology_exact": float(topo_ok),
        "synthetic_compacted_contiguous": float(contig),
    }
