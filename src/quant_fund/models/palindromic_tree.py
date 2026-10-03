"""Eertree (palindromic tree): distinct palindromic substrings + counts in O(n).

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 919


class _PNode:
    __slots__ = ("length", "suffix_link", "edges", "occ")

    def __init__(self, length: int) -> None:
        self.length = length
        self.suffix_link: _PNode | None = None
        self.edges: dict[str, _PNode] = {}
        self.occ = 0


def build_eertree(s: str) -> tuple[list[_PNode], list[int]]:
    """Returns (nodes, last-node-index per prefix)."""
    root_odd = _PNode(-1)
    root_even = _PNode(0)
    root_odd.suffix_link = root_odd
    root_even.suffix_link = root_odd
    nodes = [root_odd, root_even]
    last = root_even
    lasts: list[int] = []
    for i, c in enumerate(s):
        cur = last
        while True:
            pos = i - 1 - cur.length
            if pos >= 0 and s[pos] == c:
                break
            cur = cur.suffix_link if cur.suffix_link else root_even
        if c in cur.edges:
            last = cur.edges[c]
            last.occ += 1
            lasts.append(nodes.index(last))
            continue
        newlen = cur.length + 2
        node = _PNode(newlen)
        node.occ = 1
        cur.edges[c] = node
        nodes.append(node)
        if newlen == 1:
            node.suffix_link = root_even
        else:
            tmp = cur.suffix_link if cur.suffix_link else root_even
            while True:
                pos = i - 1 - tmp.length
                if pos >= 0 and s[pos] == c:
                    node.suffix_link = tmp.edges[c]
                    break
                tmp = tmp.suffix_link if tmp.suffix_link else root_even
        last = node
        lasts.append(nodes.index(node))
    # propagate occurrence counts up suffix links (longest first)
    for nd in sorted(nodes[2:], key=lambda x: -x.length):
        if nd.suffix_link is not None and nd.suffix_link.length > 0:
            nd.suffix_link.occ += nd.occ
    return nodes, lasts


def distinct_palindromes(s: str) -> set[str]:
    out: set[str] = set()
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if s[i:j] == s[i:j][::-1]:
                out.add(s[i:j])
    return out


def bench_palindromic_tree(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    s = "".join(rng.choice(list("ab"), 40))
    nodes, _lasts = build_eertree(s)
    # node count - 2 (roots) == number of distinct palindromes
    n_distinct = len(nodes) - 2
    truth = distinct_palindromes(s)
    score += 1.0 if n_distinct == len(truth) else 0.0
    # node lengths match palindrome lengths
    lengths = sorted(nd.length for nd in nodes[2:])
    truth_lengths = sorted(len(p) for p in truth)
    score += 1.0 if lengths == truth_lengths else 0.0
    # classic example 'eertree-like': 'ababa' palindromes
    nodes2, _ = build_eertree("ababa")
    score += 1.0 if len(nodes2) - 2 == len({"a", "b", "aba", "bab", "ababa"}) else 0.0
    # occurrence counts propagated ≥1 per node; total palindrome count via suffix links
    nodes3, _ = build_eertree("aaa")
    cnt = sorted(nd.occ for nd in nodes3[2:])
    score += 1.0 if cnt == sorted([3, 2, 1]) else 0.0
    return {"synthetic_palindromic_tree": score / 4.0}
