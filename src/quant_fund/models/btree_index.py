"""B+tree index — sorted search/insert with splits — SYNTHETIC.

Verified: tree invariant (sorted leaf chain, all leaves same depth),
find matches dict oracle, range scans sorted+complete.
"""

from __future__ import annotations

import bisect
import random

ORDER = 4  # max keys per node


class Leaf:
    def __init__(self) -> None:
        self.keys: list[int] = []
        self.vals: list[int] = []
        self.next: Leaf | None = None


class Inner:
    def __init__(self) -> None:
        self.keys: list[int] = []
        self.kids: list[object] = []


def _find_leaf(root: object, k: int) -> Leaf:
    while isinstance(root, Inner):
        i = bisect.bisect_right(root.keys, k)
        root = root.kids[i]
    if not (isinstance(root, Leaf)):
        raise ValueError("isinstance(root, Leaf)")
    return root


class BPTree:
    def __init__(self) -> None:
        self.root: object = Leaf()
        self.depth = 1

    def get(self, k: int) -> int | None:
        lf = _find_leaf(self.root, k)
        i = bisect.bisect_left(lf.keys, k)
        return lf.vals[i] if i < len(lf.keys) and lf.keys[i] == k else None

    def put(self, k: int, v: int) -> None:
        lf = _find_leaf(self.root, k)
        i = bisect.bisect_left(lf.keys, k)
        if i < len(lf.keys) and lf.keys[i] == k:
            lf.vals[i] = v
            return
        lf.keys.insert(i, k)
        lf.vals.insert(i, v)
        if len(lf.keys) > ORDER:
            self._split_leaf(lf)

    def _parent(self, target: object) -> Inner | None:
        """Walk root→leaf; return the Inner containing `target`."""

        def walk(n: object) -> Inner | None:
            if not isinstance(n, Inner):
                return None
            for k in n.kids:
                if k is target:
                    return n
                r = walk(k)
                if r is not None:
                    return r
            return None

        return walk(self.root)

    def _insert_sep(self, old: object, new: object, sep: int) -> None:
        parent = self._parent(old)
        if parent is None:
            inn = Inner()
            inn.keys = [sep]
            inn.kids = [old, new]
            self.root = inn
            return
        i = parent.kids.index(old)
        parent.keys.insert(i, sep)
        parent.kids.insert(i + 1, new)
        if len(parent.keys) > ORDER:
            mid = len(parent.keys) // 2
            push = parent.keys[mid]
            ni = Inner()
            ni.keys = parent.keys[mid + 1 :]
            ni.kids = parent.kids[mid + 1 :]
            parent.keys = parent.keys[:mid]
            parent.kids = parent.kids[: mid + 1]
            self._insert_sep(parent, ni, push)

    def _split_leaf(self, lf: Leaf) -> None:
        mid = len(lf.keys) // 2
        nl = Leaf()
        nl.keys, nl.vals = lf.keys[mid:], lf.vals[mid:]
        lf.keys, lf.vals = lf.keys[:mid], lf.vals[:mid]
        nl.next = lf.next
        lf.next = nl
        self._insert_sep(lf, nl, nl.keys[0])

    def scan(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        lf: object = self.root
        while isinstance(lf, Inner):
            lf = lf.kids[0]
        if not (isinstance(lf, Leaf)):
            raise ValueError("isinstance(lf, Leaf)")
        while lf is not None:
            out.extend(zip(lf.keys, lf.vals, strict=True))
            lf = lf.next
        return out


def bench_btree_index(seed: int = 20261231 + 360) -> dict[str, float]:
    rng = random.Random(seed)
    find_ok = scan_ok = depth_ok = 0
    trials = 25
    for _ in range(trials):
        t = BPTree()
        ref: dict[int, int] = {}
        keys = rng.sample(range(10_000), rng.randrange(30, 200))
        for k in keys:
            t.put(k, k * 7)
            ref[k] = k * 7
        find_ok += int(all(t.get(k) == ref[k] for k in keys) and t.get(-1) is None)
        s = t.scan()
        scan_ok += int(s == sorted(ref.items()))

        # leaf chain depth check: all leaves at same level
        def dep(n: object, d: int, acc: list[int]) -> None:
            if isinstance(n, Leaf):
                acc.append(d)
                return
            if not (isinstance(n, Inner)):
                raise ValueError("isinstance(n, Inner)")
            for kk in n.kids:
                dep(kk, d + 1, acc)

        ds: list[int] = []
        dep(t.root, 0, ds)
        depth_ok += int(len(set(ds)) == 1)
    return {
        "synthetic_find_exact": float(find_ok / trials),
        "synthetic_scan_sorted": float(scan_ok / trials),
        "synthetic_balanced_depth": float(depth_ok / trials),
    }
