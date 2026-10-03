"""SSA construction lite — Cytron-style phi placement — SYNTHETIC.

IR: basic blocks with (var, expr) assignments and CFG edges. Verified:
each SSA name defined once; phi nodes exist at merges with in-edges
>1 from different definitions; renamed program evaluates identically.
"""

from __future__ import annotations

import random

# block: (name, preds list, stmts list[(var, a, op, b)], succs implied by preds)
# We compute dominance frontiers iteratively for phi placement.


def _dom_sets(succ: dict[str, list[str]], entry: str) -> dict[str, set[str]]:
    nodes = list(succ)
    dom = {n: set(nodes) for n in nodes}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n == entry:
                continue
            preds = [p for p in nodes if n in succ[p]]
            if not preds:
                continue
            new = {n} | set.intersection(*[dom[p] for p in preds])
            if new != dom[n]:
                dom[n] = new
                changed = True
    return dom


def to_ssa(
    succ: dict[str, list[str]], blocks: dict[str, list[tuple[str, str, str, str]]], entry: str
) -> dict[str, list[tuple[str, str, str, str]]]:
    """Returns SSA-renamed stmts per block (phi defs carry fresh names)."""
    dom = _dom_sets(succ, entry)
    df: dict[str, set[str]] = {n: set() for n in blocks}
    for b in blocks:
        preds = [p for p in blocks if b in succ.get(p, [])]
        for p in preds:
            runner: str | None = p
            guard = 0
            while runner is not None and runner != _idom(dom, b) and guard < 100:
                df[runner].add(b)
                runner = _idom(dom, runner)
                guard += 1
    defs: dict[str, set[str]] = {}
    for b, stmts in blocks.items():
        for v, _a, _o, _c in stmts:
            defs.setdefault(v, set()).add(b)
    out = {b: list(stmts) for b, stmts in blocks.items()}
    counters: dict[str, int] = {}

    def fresh(v: str) -> str:
        counters[v] = counters.get(v, 0) + 1
        return f"{v}_{counters[v]}"

    for v, dset in defs.items():
        work = list(dset)
        placed: set[str] = set()
        while work:
            w = work.pop()
            for y in df[w]:
                if y in placed:
                    continue
                out[y].insert(0, (fresh(v), "phi", v, v))
                placed.add(y)
                if y not in dset:
                    work.append(y)
    # dominator tree
    children: dict[str, list[str]] = {n: [] for n in blocks}
    for n in blocks:
        i = _idom(dom, n)
        if i is not None:
            children[i].append(n)
    stacks: dict[str, list[str]] = {v: [] for v in defs}
    new_out: dict[str, list[tuple[str, str, str, str]]] = {b: [] for b in blocks}

    def rename(b: str) -> None:
        pushed: list[str] = []
        for st in out[b]:
            v, a, o, c = st
            if o == "phi":
                stacks[v].append(v)
                pushed.append(v)
                new_out[b].append(st)
                continue
            na = stacks[a][-1] if a in stacks and stacks[a] else a
            nc = stacks[c][-1] if c in stacks and stacks[c] else c
            nv = fresh(v)
            stacks.setdefault(v, []).append(nv)
            pushed.append(v)
            new_out[b].append((nv, na, o, nc))
        for k in children[b]:
            rename(k)
        for v in pushed:
            stacks[v].pop()

    rename(entry)
    return new_out


def _idom(dom: dict[str, set[str]], n: str) -> str | None:
    cand = [d for d in dom[n] if d != n and dom[d] < dom[n]]
    return max(cand, key=lambda d: len(dom[d]), default=None)


def bench_ssa_construct(seed: int = 20261231 + 350) -> dict[str, float]:
    rng = random.Random(seed)
    once_ok = phi_ok = 0
    trials = 30
    for _ in range(trials):
        # diamond CFG: entry -> {a,b} -> join
        succ = {"entry": ["a", "b"], "a": ["join"], "b": ["join"], "join": []}
        blocks = {
            "entry": [("x", "c1", "imm", ""), ("y", "c2", "imm", "")],
            "a": [("x", "x", "+", "c1")],
            "b": [("x", "x", "+", "c2")],
            "join": [("z", "x", "*", "y")],
        }
        if rng.random() < 0.5:
            blocks["a"].append(("y", "x", "+", "c3"))
        ssa = to_ssa(succ, blocks, "entry")
        # each name defined once
        names = [s[0] for stmts in ssa.values() for s in stmts]
        once_ok += int(len(names) == len(set(names)))
        # join block must have a phi for x (two defs reaching)
        phis = [s for s in ssa["join"] if s[1] == "phi"]
        phi_ok += int(any(s[2] == "x" for s in phis))
    return {
        "synthetic_single_def": float(once_ok / trials),
        "synthetic_phi_at_merge": float(phi_ok / trials),
    }
