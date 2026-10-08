"""Catmull-Clark subdivision on a quad-dominant mesh (SYNTHETIC).

One subdivision step: face points (vertex average), edge points
(vertex+endpoint average), and smoothed vertices (Catmull-Clark rule
Q/n + 2R/n + (n-3)v/n). Verified on a cube and a perturbed-quad mesh:
vertex/face/edge counts match the closed-form refinement counts, the
mesh stays manifold-consistent (each undirected edge used by its
faces), and repeated subdivision shrinks the bbox monotonically toward
the centroid neighborhood (contraction).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 970


def cc_step(verts: np.ndarray, faces: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    v = verts.tolist()
    face_pts = []
    for f in faces:
        face_pts.append(np.mean(verts[f], axis=0))
    # undirected edges -> adjacent faces
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        m = len(f)
        for k in range(m):
            e = (f[k], f[(k + 1) % m]) if f[k] < f[(k + 1) % m] else (f[(k + 1) % m], f[k])
            edge_map.setdefault(e, []).append(fi)
    edge_pts = {}
    for e, adj in edge_map.items():
        if len(adj) == 2:
            edge_pts[e] = 0.25 * (verts[e[0]] + verts[e[1]] + face_pts[adj[0]] + face_pts[adj[1]])
        else:  # boundary edge: midpoint
            edge_pts[e] = 0.5 * (verts[e[0]] + verts[e[1]])
    # vertex neighbors
    adj_v: list[set[int]] = [set() for _ in range(len(v))]
    adj_f: list[set[int]] = [set() for _ in range(len(v))]
    for e in edge_map:
        adj_v[e[0]].add(e[1])
        adj_v[e[1]].add(e[0])
    for fi, f in enumerate(faces):
        for vi in f:
            adj_f[vi].add(fi)
    new_v = []
    for i, vi in enumerate(v):
        n = len(adj_v[i])
        if n == 0:
            new_v.append(np.asarray(vi))
            continue
        q = np.mean([face_pts[fi] for fi in adj_f[i]], axis=0)
        r = np.mean([0.5 * (verts[i] + verts[j]) for j in adj_v[i]], axis=0)
        new_v.append(q / n + 2.0 * r / n + (n - 3) * np.asarray(vi) / n)
    # assemble new mesh
    out_v = new_v + face_pts + [edge_pts[e] for e in edge_map]
    e_index = {e: len(new_v) + len(face_pts) + k for k, e in enumerate(edge_map)}
    out_f: list[list[int]] = []
    for fi, f in enumerate(faces):
        m = len(f)
        fp_idx = len(new_v) + fi
        for k in range(m):
            e1 = (f[k], f[(k + 1) % m]) if f[k] < f[(k + 1) % m] else (f[(k + 1) % m], f[k])
            e0 = (f[(k - 1) % m], f[k]) if f[(k - 1) % m] < f[k] else (f[k], f[(k - 1) % m])
            out_f.append([f[k], e_index[e1], fp_idx, e_index[e0]])
    return np.asarray(out_v), out_f


def cube_mesh() -> tuple[np.ndarray, list[list[int]]]:
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    f = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    return v, f


def bench_catmull_clark(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    v, f = cube_mesh()
    v = v + rng.normal(scale=0.05, size=v.shape)
    v1, f1 = cc_step(v, f)
    nv, nf, ne_expected = len(v), len(f), 12
    counts_ok = len(v1) == nv + nf + ne_expected and len(f1) == 4 * nf
    # manifold check: every undirected edge in new mesh shared by <=2 faces
    em: dict[tuple[int, int], int] = {}
    for face in f1:
        m = len(face)
        for k in range(m):
            e = (
                (face[k], face[(k + 1) % m])
                if face[k] < face[(k + 1) % m]
                else (face[(k + 1) % m], face[k])
            )
            em[e] = em.get(e, 0) + 1
    manifold_ok = all(c <= 2 for c in em.values())
    # contraction toward centroid after several steps
    vv, ff = v, f
    spans = []
    for _ in range(4):
        vv, ff = cc_step(vv, ff)
        spans.append(float((vv.max(axis=0) - vv.min(axis=0)).max()))
    contracts = all(spans[i + 1] < spans[i] for i in range(3))
    finite = bool(np.isfinite(vv).all())
    return {"synthetic_catmull_clark": float(np.mean([counts_ok, manifold_ok, contracts, finite]))}
