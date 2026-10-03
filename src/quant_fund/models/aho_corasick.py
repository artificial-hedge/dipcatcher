"""Aho-Corasick multi-pattern string search (synthetic).

Trie + failure links; O(n + total pattern length + matches).
Verified: match set equals naive per-pattern find oracle on random
texts with planted + noise patterns.
"""

from __future__ import annotations

import random
from collections import deque
from typing import cast


def build(patterns: list[str]) -> list[dict[str, object]]:
    """Trie nodes: children dict, fail link, output set of pattern ids."""
    nodes: list[dict[str, object]] = [{"ch": {}, "fail": 0, "out": set()}]
    for pid, pat in enumerate(patterns):
        v = 0
        for c in pat:
            ch: dict[str, int] = nodes[v]["ch"]  # type: ignore[assignment]
            if c not in ch:
                ch[c] = len(nodes)
                nodes.append({"ch": {}, "fail": 0, "out": set()})
            v = ch[c]
        out: set[int] = nodes[v]["out"]  # type: ignore[assignment]
        out.add(pid)
    # BFS fail links
    q: deque[int] = deque()
    root_ch: dict[str, int] = nodes[0]["ch"]  # type: ignore[assignment]
    for u in root_ch.values():
        nodes[u]["fail"] = 0
        q.append(u)
    while q:
        v = q.popleft()
        vch: dict[str, int] = nodes[v]["ch"]  # type: ignore[assignment]
        for c, u in vch.items():
            f = cast("int", nodes[v]["fail"])
            fch: dict[str, int] = nodes[f]["ch"]  # type: ignore[assignment]
            while c not in fch and f != 0:
                f = cast("int", nodes[f]["fail"])
                fch = nodes[f]["ch"]  # type: ignore[assignment]
            nodes[u]["fail"] = fch.get(c, 0)
            uout: set[int] = nodes[u]["out"]  # type: ignore[assignment]
            fout: set[int] = nodes[cast("int", nodes[u]["fail"])]["out"]  # type: ignore[index,assignment]
            uout |= fout
            q.append(u)
    return nodes


def search(text: str, nodes: list[dict[str, object]]) -> list[tuple[int, int]]:
    """Return [(pos, pattern_id)] for every pattern ending at pos."""
    v = 0
    hits: list[tuple[int, int]] = []
    for i, c in enumerate(text):
        ch: dict[str, int] = nodes[v]["ch"]  # type: ignore[assignment]
        while c not in ch and v != 0:
            v = nodes[v]["fail"]  # type: ignore[assignment]
            ch = nodes[v]["ch"]  # type: ignore[assignment]
        v = ch.get(c, 0)
        out: set[int] = nodes[v]["out"]  # type: ignore[assignment]
        for pid in out:
            hits.append((i, pid))
    return hits


def bench_aho_corasick(seed: int = 20261231 + 260) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "abcd"
    agree = 0
    trials = 40
    for _ in range(trials):
        text = "".join(rng.choice(alpha) for _ in range(200))
        pats = []
        for _ in range(6):
            if rng.random() < 0.5:
                i = rng.randrange(len(text))
                pats.append(text[i : i + rng.randint(1, 8)])
            else:
                pats.append("".join(rng.choice(alpha) for _ in range(rng.randint(2, 6))))
        nodes = build(pats)
        got = {(pos, pid) for pos, pid in search(text, nodes)}
        want = {
            (j + len(p) - 1, pid)
            for pid, p in enumerate(pats)
            for j in range(len(text) - len(p) + 1)
            if text[j : j + len(p)] == p and p
        }
        agree += int(got == want)
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_mean_patterns": 6.0,
        "synthetic_text_len": 200.0,
    }
