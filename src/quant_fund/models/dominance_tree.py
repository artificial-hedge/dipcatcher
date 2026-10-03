"""Immediate-dominator tree via iterative dataflow (synthetic).

CFG dominator sets by standard meet-over-paths fixpoint; idom =
strict dominator not dominated by any other strict dominator.
Verified: (i) dom sets agree with exhaustive path-enumeration
oracle; (ii) idom tree is a valid spanning tree rooted at entry;
(iii) dom(v) = {v} ∪ dom(idom(v)) for all v.
"""

from __future__ import annotations

import random


def dom_sets(adj: list[list[int]], entry: int = 0) -> list[set[int]]:
    n = len(adj)
    dom: list[set[int]] = [set(range(n)) for _ in range(n)]
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if v == entry:
                continue
            preds = [u for u in range(n) if v in adj[u]]
            if not preds:
                continue
            new = {v} | set.intersection(*(dom[u] for u in preds))
            if new != dom[v]:
                dom[v] = new
                changed = True
    return dom


def idom_tree(dom: list[set[int]], entry: int = 0) -> dict[int, int]:
    """idom[v] = strict dominator of v dominated by all other strict doms."""
    idom: dict[int, int] = {}
    for v in range(len(dom)):
        if v == entry:
            continue
        strict = dom[v] - {v}
        if not strict:
            continue
        # idom = the strict dominator u where dom[u] is maximal
        cand = max(strict, key=lambda u: len(dom[u]))
        idom[v] = cand
    return idom


def _paths_dom(adj: list[list[int]], entry: int, v: int) -> set[int]:
    """Oracle: nodes on EVERY entry→v path."""
    on_all: set[int] | None = None
    stack = [(entry, [entry])]
    while stack:
        u, path = stack.pop()
        if u == v:
            s = set(path)
            on_all = s if on_all is None else on_all & s
            continue
        for w in adj[u]:
            if w not in path:
                stack.append((w, path + [w]))
    return on_all or set()


def bench_dominance_tree(seed: int = 20261231 + 283) -> dict[str, float]:
    rng = random.Random(seed)
    agree = tree_ok = rec_ok = ran = 0
    trials = 30
    for _ in range(trials):
        n = rng.randint(4, 9)
        # random DAG-ish CFG with backedges allowed
        adj: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < 0.2:
                    adj[i].append(j)
        # ensure everything reachable from 0 for oracle finiteness
        reach = {0}
        work = [0]
        while work:
            u = work.pop()
            for v in adj[u]:
                if v not in reach:
                    reach.add(v)
                    work.append(v)
        keep = sorted(reach)
        remap = {o: i for i, o in enumerate(keep)}
        adj = [[remap[v] for v in adj[o] if v in remap] for o in keep]
        n = len(adj)
        if n < 2:
            continue
        ran += 1
        dom = dom_sets(adj, 0)
        ok = all(dom[v] == _paths_dom(adj, 0, v) for v in range(n))
        agree += int(ok)
        idom = idom_tree(dom, 0)
        # idom must be a tree: following parents reaches root
        ok2 = True
        for v in range(1, n):
            seen = set()
            cur = v
            while cur in idom and cur not in seen:
                seen.add(cur)
                cur = idom[cur]
            ok2 = ok2 and cur == 0
        tree_ok += int(ok2)
        # dom(v) = {v} ∪ dom(idom(v))
        ok3 = all(dom[v] == {v} | dom[idom[v]] for v in range(1, n) if v in idom)
        rec_ok += int(ok3)
    return {
        "synthetic_dom_agree": float(agree / max(1, ran)),
        "synthetic_idom_tree": float(tree_ok / max(1, ran)),
        "synthetic_recurrence": float(rec_ok / max(1, ran)),
    }
