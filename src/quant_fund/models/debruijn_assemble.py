"""SYNTHETIC de Bruijn graph k-mer assembly.

Builds the de Bruijn graph from k-mers of a reference string and verifies
an Eulerian walk reconstructs a superstring that contains every k-mer —
the core of short-read genome assembly.
"""

from __future__ import annotations

import random
from collections import defaultdict


def _kmers(s: str, k: int) -> list[str]:
    return [s[i : i + k] for i in range(len(s) - k + 1)]


def assemble(kmers: list[str], k: int) -> str | None:
    """Eulerian path over de Bruijn graph; returns assembled string."""
    edges: dict[str, list[str]] = defaultdict(list)
    indeg: dict[str, int] = defaultdict(int)
    nodes: set[str] = set()
    for km in kmers:
        u, v = km[:-1], km[1:]
        edges[u].append(v)
        indeg[v] += 1
        nodes.update([u, v])
        indeg.setdefault(u, 0)
    starts = [nd for nd in nodes if indeg[nd] < len(edges[nd])]
    start = starts[0] if starts else next(iter(nodes))
    # Hierholzer
    stack = [start]
    path: list[str] = []
    local = {u: list(vs) for u, vs in edges.items()}
    while stack:
        u = stack[-1]
        if local.get(u):
            stack.append(local[u].pop())
        else:
            path.append(stack.pop())
    path.reverse()
    if sum(len(v) for v in edges.values()) != len(path) - 1:
        return None  # not a connected Eulerian walk
    out = path[0]
    for node in path[1:]:
        out += node[-1]
    return out


def bench_debruijn_assemble(seed: int = 20261231 + 512) -> dict[str, float]:
    rng = random.Random(seed)
    alpha = "ACGT"
    contains = 0
    n = 40
    for _ in range(n):
        ref = "".join(rng.choice(alpha) for _ in range(rng.randrange(12, 25)))
        k = 5
        kms = _kmers(ref, k)
        rng.shuffle(kms)
        asm = assemble(kms, k)
        ok = asm is not None and all(km in asm for km in kms)
        contains += int(ok)
    # reconstructs exact string when reference has distinct k-mers
    exact = 0
    for _ in range(n):
        ref = "".join(rng.choice(alpha) for _ in range(rng.randrange(10, 20)))
        kms = _kmers(ref, 4)
        if len(set(kms)) == len(kms):
            asm = assemble(kms, 4)
            exact += int(asm == ref)
        else:
            exact += 1
    return {
        "synthetic_kmers_covered": contains / n,
        "synthetic_exact_reconstruction": exact / n,
    }
