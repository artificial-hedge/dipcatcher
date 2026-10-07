"""Umbrella-operator Laplacian mesh smoothing + cotangent Laplacian (SYNTHETIC).

Uniform smoothing v <- v + lambda*(mean(neighbors)-v) damps noise but
shrinks volume; cotangent-weighted Laplacian is the discrete mean
curvature normal. Verified: noise energy drops monotonically for the
first steps, uniform smoothing shrinks the mesh (documented pathology),
and the cotangent Laplacian of a planar region's interior vertices is
~0 (flatness recovery).
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 974


def umbrella_smooth(
    verts: np.ndarray, adj: list[set[int]], lam: float = 0.5, iters: int = 1
) -> np.ndarray:
    v = verts.copy()
    for _ in range(iters):
        nv = v.copy()
        for i in range(len(v)):
            if adj[i]:
                nb = np.mean([v[j] for j in adj[i]], axis=0)
                nv[i] = v[i] + lam * (nb - v[i])
        v = nv
    return v


def cotangent_laplacian(verts: np.ndarray, faces: list[list[int]], i: int) -> np.ndarray:
    """sum_j (cot a_ij + cot b_ij)(v_j - v_i) over neighbors."""
    adj: set[int] = set()
    tri_of: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        if i in f:
            tri_of.setdefault(i, []).append(fi)
            adj.update(w for w in f if w != i)
    out = np.zeros(3)
    vi = verts[i]
    for j in adj:
        wsum = 0.0
        for f in faces:
            if i in f and j in f:
                k = [w for w in f if w not in (i, j)][0]
                e1 = verts[i] - verts[k]
                e2 = verts[j] - verts[k]
                cr = np.cross(e1, e2)
                nn = np.linalg.norm(cr)
                if nn < 1e-14:
                    continue
                cot = float(e1 @ e2) / nn
                wsum += cot
        out += wsum * (verts[j] - vi)
    return out


def bench_laplacian_smooth(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # noisy sphere-ish mesh: icosphere adjacency via faces
    from quant_fund.models.loop_subdiv import icosahedron

    v0, faces = icosahedron()
    adj: list[set[int]] = [set() for _ in range(len(v0))]
    for f in faces:
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            adj[a].add(b)
            adj[b].add(a)
    noisy = v0 + rng.normal(scale=0.08, size=v0.shape)

    def noise_energy(x: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    np.linalg.norm(x[i] - np.mean([x[j] for j in adj[i]], axis=0))
                    for i in range(len(x))
                ]
            )
        )

    n0 = noise_energy(noisy)
    s1 = umbrella_smooth(noisy, adj, 0.5, 1)
    s3 = umbrella_smooth(noisy, adj, 0.5, 3)
    e1 = noise_energy(s1)
    e3 = noise_energy(s3)
    span0 = float((v0.max(0) - v0.min(0)).mean())
    span3 = float((s3.max(0) - s3.min(0)).mean())
    # planar patch cotangent Laplacian ~ 0 at an interior vertex
    sq_v = np.array([[x, y, 0.0] for x in range(3) for y in range(3)])
    sq_f = [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]]
    lap = cotangent_laplacian(sq_v, sq_f, 4)
    flat_ok = float(np.linalg.norm(lap)) < 1e-10
    checks = [
        e1 < n0,
        e3 < e1,
        span3 < span0,  # shrinkage pathology documented
        flat_ok,
    ]
    return {"synthetic_laplacian_smooth": float(np.mean(checks))}
