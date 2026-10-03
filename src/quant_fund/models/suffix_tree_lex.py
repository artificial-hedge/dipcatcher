"""Suffix tree (naive O(n^2) build over a compressed trie) with substring queries.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 916


class _Node:
    __slots__ = ("children", "start")

    def __init__(self, start: int = -1) -> None:
        self.children: dict[str, _Node] = {}
        self.start = start  # suffix start index at leaves


def build_suffix_tree(s: str) -> _Node:
    s = s + "$"
    root = _Node()
    for i in range(len(s)):
        node = root
        for c in s[i:]:
            node = node.children.setdefault(c, _Node())
        node.start = i
    return root


def contains(root: _Node, s: str, pat: str) -> bool:
    node = root
    for c in pat:
        if c not in node.children:
            return False
        node = node.children[c]
    return True


def occurrences(root: _Node, s: str, pat: str) -> list[int]:
    node = root
    for c in pat:
        if c not in node.children:
            return []
        node = node.children[c]
    out: list[int] = []
    stack = [node]
    while stack:
        nd = stack.pop()
        if nd.start >= 0:
            out.append(nd.start)
        stack.extend(nd.children.values())
    return sorted(out)


def bench_suffix_tree_lex(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    s = "".join(rng.choice(list("ab"), 60))
    root = build_suffix_tree(s)
    # membership agrees with `in` for all substrings up to len 6
    ok = True
    for i in range(len(s)):
        for j in range(i + 1, min(len(s), i + 7)):
            ok = ok and contains(root, s, s[i:j]) == (s[i:j] in s)
    score += 1.0 if ok else 0.0
    # absent pattern rejected
    score += 1.0 if not contains(root, s, "ccc") else 0.0
    # occurrences match scan
    pat = "ab"
    occ = occurrences(root, s, pat)
    truth = [i for i in range(len(s) - 1) if s[i : i + 2] == pat]
    score += 1.0 if occ == truth else 0.0

    # leaf count == n+1 (each suffix)
    def leaves(nd: _Node) -> int:
        if not nd.children:
            return 1
        return sum(leaves(c) for c in nd.children.values())

    score += 1.0 if leaves(root) == len(s) + 1 else 0.0
    return {"synthetic_suffix_tree_lex": score / 4.0}
