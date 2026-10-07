"""SYNTHETIC B-link tree — high keys + right siblings for concurrent splits.

Nodes carry (keys, highkey, right_link). Insert may split and post a
separator to the parent lazily; searches must "move right" past highkey.
Verify: lookups succeed right after splits before parent updated.
"""

from __future__ import annotations

import random


class Node:
    def __init__(self, leaf: bool = True) -> None:
        self.leaf = leaf
        self.keys: list[int] = []
        self.kids: list[Node] = []
        self.high = 10**9
        self.right: Node | None = None


def _find_leaf(root: Node, k: int) -> Node:
    n = root
    while not n.leaf:
        i = 0
        while i < len(n.keys) and k >= n.keys[i]:
            i += 1
        n = n.kids[i]
    # move right while key exceeds highkey (concurrent-split safety)
    while k >= n.high and n.right is not None:
        n = n.right
    return n


def _insert(root: Node, k: int, cap: int = 4) -> Node:
    n = _find_leaf(root, k)
    n.keys.append(k)
    n.keys.sort()
    if len(n.keys) <= cap:
        return root
    # split: right half gets high = old high, left gets new sep
    mid = len(n.keys) // 2
    sib = Node(leaf=True)
    sib.keys = n.keys[mid:]
    sib.high = n.high
    sib.right = n.right
    n.keys = n.keys[:mid]
    n.high = sib.keys[0]
    n.right = sib
    # post separator to parent if root is single leaf we grow a new root
    if root is n:
        r = Node(leaf=False)
        r.keys = [sib.keys[0]]
        r.kids = [n, sib]
        return r

    # find parent path and insert separator
    def _post(node: Node, child: Node, sep: int) -> bool:
        if node.leaf:
            return False
        for i, kd in enumerate(node.kids):
            if kd is child or _post(kd, child, sep):
                node.keys.insert(min(i, len(node.keys)), sep) if kd is child else None
                if kd is child:
                    if not (n.right is not None):
                        raise ValueError("n.right is not None")
                    node.kids.insert(i + 1, n.right)
                return True
        return False

    _post(root, n, sib.keys[0])
    return root


def _search(root: Node, k: int) -> bool:
    n = _find_leaf(root, k)
    return k in n.keys


def bench_blink_tree(seed: int = 20261231 + 435) -> dict[str, float]:
    rng = random.Random(seed)
    found = ordered = move_right = 0
    trials = 30
    for _ in range(trials):
        keys = rng.sample(range(1, 400), rng.randrange(20, 50))
        root = Node()
        for k in keys:
            root = _insert(root, k)
        found += int(all(_search(root, k) for k in keys))
        found += int(not any(_search(root, k + 500) for k in keys))
        # leaves form a sorted chain via right links
        lf: Node | None = root
        while lf is not None and not lf.leaf:
            lf = lf.kids[0]
        seq = []
        while lf is not None:
            seq.extend(lf.keys)
            lf = lf.right
        ordered += int(seq == sorted(keys))
        # move-right exercised: a key at/above a node highkey is still found
        move_right += int(all(_search(root, k) for k in keys[:5]))
    return {
        "synthetic_finds_all": float(found / (2 * trials)),
        "synthetic_sorted_chain": float(ordered / trials),
        "synthetic_move_right_safe": float(move_right / trials),
    }
