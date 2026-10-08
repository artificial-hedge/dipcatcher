"""Loop subdivision on a triangular mesh (SYNTHETIC).

One Loop step: new edge vertices at 3/8(endpoints) + 1/8(opposite
vertices); updated vertices at (1-n*u)v + u*sum(neighbors) with
u = 3/(8n) for n>3 else 3/16. Verified on an icosahedron: refined
counts match V' = V+E, F' = 4F, every new edge is shared by exactly 2
faces (closed manifold), and the limit surface tightens (max vertex
displacement decays geometrically).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 971


def icosahedron() -> tuple[np.ndarray, list[list[int]]]:
    t = (1 + np.sqrt(5)) / 2
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    return v, f


def loop_step(verts: np.ndarray, faces: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    edge_map: dict[tuple[int, int], list[int]] = {}
    adj: list[set[int]] = [set() for _ in range(len(verts))]
    for fi, f in enumerate(faces):
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            e = (a, b) if a < b else (b, a)
            edge_map.setdefault(e, []).append(fi)
            adj[a].add(b)
            adj[b].add(a)
    # edge vertices
    ev: dict[tuple[int, int], np.ndarray] = {}
    for e, adj_f in edge_map.items():
        if len(adj_f) == 2:
            opp = [w for fi in adj_f for w in faces[fi] if w not in e]
            ev[e] = (3 * verts[e[0]] + 3 * verts[e[1]] + verts[opp[0]] + verts[opp[1]]) / 8
        else:
            ev[e] = 0.5 * (verts[e[0]] + verts[e[1]])
    # updated vertices
    new_v = np.zeros_like(verts)
    for i in range(len(verts)):
        n = len(adj[i])
        if n == 0:
            new_v[i] = verts[i]
            continue
        u = 3.0 / (8.0 * n) if n > 3 else 3.0 / 16.0
        nb = np.mean([verts[j] for j in adj[i]], axis=0)
        new_v[i] = (1 - n * u) * verts[i] + u * n * nb
    e_index = {e: len(verts) + k for k, e in enumerate(edge_map)}
    out_v = np.concatenate([new_v] + [ev[e][None] for e in edge_map], axis=0)
    out_f: list[list[int]] = []
    for f in faces:
        e01 = e_index[(f[0], f[1]) if f[0] < f[1] else (f[1], f[0])]
        e12 = e_index[(f[1], f[2]) if f[1] < f[2] else (f[2], f[1])]
        e20 = e_index[(f[2], f[0]) if f[2] < f[0] else (f[0], f[2])]
        out_f += [
            [f[0], e01, e20],
            [f[1], e12, e01],
            [f[2], e20, e12],
            [e01, e12, e20],
        ]
    return out_v, out_f


def bench_loop_subdiv(seed: int = _SEED) -> dict[str, float]:
    v, f = icosahedron()
    v1, f1 = loop_step(v, f)
    e_count = 30
    counts_ok = len(v1) == len(v) + e_count and len(f1) == 4 * len(f)
    em: dict[tuple[int, int], int] = {}
    for face in f1:
        for k in range(3):
            e = (
                (face[k], face[(k + 1) % 3])
                if face[k] < face[(k + 1) % 3]
                else (face[(k + 1) % 3], face[k])
            )
            em[e] = em.get(e, 0) + 1
    manifold_ok = all(c == 2 for c in em.values())
    # geometric contraction toward smooth limit: displacement decays
    rng = np.random.default_rng(seed)
    vv = v + rng.normal(scale=0.02, size=v.shape)
    ff = f
    disps = []
    prev = vv
    for _ in range(3):
        vv2, ff2 = loop_step(prev, ff)
        d = float(np.abs(vv2[: len(prev)] - prev).max())
        disps.append(d)
        prev, ff = vv2, ff2
        vv = vv2
    decay = all(disps[i + 1] < disps[i] * 0.9 for i in range(2))
    return {"synthetic_loop_subdiv": float(np.mean([counts_ok, manifold_ok, decay]))}
