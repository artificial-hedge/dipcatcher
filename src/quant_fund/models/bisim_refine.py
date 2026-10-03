"""Bisimulation minimization via partition refinement (Paige–Tarjan lite).

States with action labels; two states are bisimilar iff same labels and
successors land in same blocks. Refine by splitting on
(label, signature of target blocks) until stable — Hopcroft-style.
"""

from __future__ import annotations

_SEED = 20261231 + 1045


def minimize(
    states: set[int],
    labels: dict[int, str],
    succ: dict[int, list[int]],
    init_blocks: list[set[int]] | None = None,
) -> list[set[int]]:
    blocks = init_blocks or [set(states)]
    while True:
        sig_of: dict[int, tuple] = {}
        bid = {s: i for i, b in enumerate(blocks) for s in b}
        for s in states:
            tgts = tuple(sorted(bid.get(t, -1) for t in succ.get(s, [])))
            sig_of[s] = (labels.get(s, ""), tgts)
        new_blocks: list[set[int]] = []
        for b in blocks:
            parts: dict[tuple, set[int]] = {}
            for s in b:
                parts.setdefault(sig_of[s], set()).add(s)
            new_blocks.extend(parts.values())
        if len(new_blocks) == len(blocks) and all(
            sorted(nb) == sorted(ob) for nb, ob in zip(new_blocks, blocks, strict=True)
        ):
            return new_blocks
        blocks = new_blocks


def bisimilar(
    states: set[int], labels: dict[int, str], succ: dict[int, list[int]], a: int, b: int
) -> bool:
    blocks = minimize(states, labels, succ)
    return any(a in blk and b in blk for blk in blocks)


def bench_bisim_refine(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # s0,s1 identical label 'a' both -> s2 'b': s0~s1
    states = {0, 1, 2}
    labels = {0: "a", 1: "a", 2: "b"}
    succ = {0: [2], 1: [2], 2: []}
    bl = minimize(states, labels, succ)
    checks.append(len(bl) == 2 and bisimilar(states, labels, succ, 0, 1))
    # differ in targets: s0->'b'-state, s1->'c'-state -> not bisimilar
    states2 = {0, 1, 2, 3}
    labels2 = {0: "a", 1: "a", 2: "b", 3: "c"}
    succ2 = {0: [2], 1: [3], 2: [], 3: []}
    checks.append(not bisimilar(states2, labels2, succ2, 0, 1))
    # cyclic: s0<->s1 same label -> bisimilar
    checks.append(bisimilar({0, 1}, {0: "a", 1: "a"}, {0: [1], 1: [0]}, 0, 1))
    # minimized chain a->b->c stays 3 blocks
    bl2 = minimize({0, 1, 2}, {0: "a", 1: "b", 2: "c"}, {0: [1], 1: [2], 2: []})
    checks.append(len(bl2) == 3)
    # diamond: s0->{s1,s2}->{s3}: s1~s2
    bl3 = minimize(
        {0, 1, 2, 3}, {0: "a", 1: "b", 2: "b", 3: "c"}, {0: [1, 2], 1: [3], 2: [3], 3: []}
    )
    checks.append(len(bl3) == 3)
    return {"synthetic_bisim_refine": float(sum(checks)) / len(checks)}
