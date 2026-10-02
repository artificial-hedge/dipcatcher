"""RGA sequence CRDT: collaborative ordered list (synthetic).

Nodes carry (sid, sseq) unique ids; each insert references its
left neighbor; merge = causal-order weave (sort by (sid desc at
same parent) then seq). Simplified: verify convergence by
merging replicas in all orders → identical sequence.
"""

from __future__ import annotations

import random
from itertools import permutations

# node: (parent_id, node_id, char); parent_id -1 = list head
Node = tuple[int, int, str]


def rga_seq(nodes: list[Node]) -> list[str]:
    """Linearize: DFS from head, children sorted by node_id desc."""
    kids: dict[int, list[tuple[int, str]]] = {}
    for p, i, c in nodes:
        kids.setdefault(p, []).append((i, c))
    for k in kids:
        kids[k].sort(key=lambda t: -t[0])
    out: list[str] = []

    def dfs(p: int) -> None:
        for i, c in sorted(kids.get(p, []), key=lambda t: -t[0]):
            out.append(c)
            dfs(i)

    dfs(-1)
    return out


def bench_rga_sequence(seed: int = 20261231 + 305) -> dict[str, float]:
    rng = random.Random(seed)
    # two replicas insert chars at varying parents; merged node set
    # must linearize identically regardless of arrival order
    conv = 0
    trials = 40
    for _ in range(trials):
        nodes: list[Node] = []
        # sequential edits: insert after a random existing node
        ids = [-1]
        for nid, ch in enumerate(rng.choice(["abca", "hello", "xyyx"])):
            p = rng.choice(ids)
            nodes.append((p, nid, ch))
            ids.append(nid)
        # split into two replica views with different orderings
        order = rng.sample(nodes, len(nodes))
        seq1 = rga_seq(order)
        seq2 = rga_seq(rng.sample(nodes, len(nodes)))
        conv += int(seq1 == seq2)
    # deterministic tie-break: concurrent inserts at same parent
    # order by node_id desc — check explicit case
    nodes = [(-1, 1, "a"), (-1, 2, "b"), (-1, 3, "c")]
    seq = rga_seq(nodes)
    tie_ok = seq == ["c", "b", "a"]  # higher id first at head
    # different permutation orders converge
    perms_ok = all(rga_seq(list(p)) == seq for p in permutations(nodes))
    return {
        "synthetic_convergent": float(conv / trials),
        "synthetic_tiebreak": float(tie_ok),
        "synthetic_all_orders": float(perms_ok),
    }
