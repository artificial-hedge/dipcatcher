"""Boundary-of-boundary is zero (wave 280) (SYNTHETIC).

For random flag complexes (clique complexes of random graphs), verify
∂_k ∘ ∂_{k+1} = 0 over GF(2) — the foundational chain-complex property.
"""

import numpy as np

_SEED = 20261231 + 765


def _boundary(faces: list[tuple[int, ...]], codom: list[tuple[int, ...]]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(codom)}
    m = np.zeros((len(codom), len(faces)), dtype=int)
    for j, s in enumerate(faces):
        for f in range(len(s)):
            m[idx[s[:f] + s[f + 1 :]], j] = 1
    return m


def bench_boundary_sq(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n, p = 8, 0.5
    adj = rng.rand(n, n) < p
    np.fill_diagonal(adj, False)
    edges: list[tuple[int, ...]] = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    tris: list[tuple[int, ...]] = [
        (i, j, k) for (i, j) in edges for k in range(j + 1, n) if adj[i, k] and adj[j, k]
    ]
    verts: list[tuple[int, ...]] = [(i,) for i in range(n)]
    ok = 0
    # the real chain-complex property: ∂_1 ∘ ∂_2 = 0 over GF(2) —
    # only verifiable on a sample that actually has triangles
    if tris:
        d2 = _boundary(tris, edges)
        d1 = _boundary(edges, verts)
        ok += int(((d1 @ d2) % 2 == 0).all())
        ok += int(d2.shape[1] > 0 and d1.shape[1] > 0)
    return {"synthetic_boundary_sq": float(ok >= 2)}
