"""Simplicial homology over GF(2): Betti numbers via boundary ranks (wave 280) (SYNTHETIC).

For a complex K: b_k = dim ker ∂_k - rank ∂_{k+1}, computed by GF(2) Gaussian
elimination on incidence matrices. Oracles: sphere (1,0,1), torus (1,2,1),
circle (1,1), figure-8 (1,2).
"""

import numpy as np

_SEED = 20261231 + 764


def _gf2_rank(mat: np.ndarray) -> int:
    m = mat.copy() % 2
    rows, cols = m.shape
    r = 0
    for c in range(cols):
        piv = np.nonzero(m[r:, c])[0]
        if len(piv) == 0:
            continue
        m[[r, r + piv[0]]] = m[[r + piv[0], r]]
        for i in range(rows):
            if i != r and m[i, c]:
                m[i] ^= m[r]
        r += 1
        if r == rows:
            break
    return r


def betti(simplices_by_dim: list[list[tuple[int, ...]]]) -> list[int]:
    out = []
    for k in range(len(simplices_by_dim)):
        lower, upper = (
            simplices_by_dim[k],
            (simplices_by_dim[k + 1] if k + 1 < len(simplices_by_dim) else []),
        )
        dk = np.zeros(
            (max(len(simplices_by_dim[k - 1]), 1) if k else max(len(lower), 1), max(len(lower), 1)),
            dtype=int,
        )
        if k == 0:
            dk = np.zeros((1, len(lower)), dtype=int)
        else:
            idx = {s: i for i, s in enumerate(simplices_by_dim[k - 1])}
            for j, s in enumerate(lower):
                for f in range(k + 1):
                    dk[idx[tuple(sorted(s[:f] + s[f + 1 :]))], j] = 1
        dk1 = np.zeros((max(len(lower), 1), max(len(upper), 1)), dtype=int)
        if upper:
            idx2 = {s: i for i, s in enumerate(lower)}
            for j, s in enumerate(upper):
                for f in range(k + 2):
                    dk1[idx2[tuple(sorted(s[:f] + s[f + 1 :]))], j] = 1
        out.append(len(lower) - _gf2_rank(dk) - _gf2_rank(dk1))
    return out


def _complex(kind: str) -> list[list[tuple[int, ...]]]:
    if kind == "sphere":  # tetrahedron boundary
        v0: list[tuple[int, ...]] = [(0,), (1,), (2,), (3,)]
        e: list[tuple[int, ...]] = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        t: list[tuple[int, ...]] = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        return [v0, e, t]
    if kind == "torus":  # 3x3-grid minimal torus triangulation
        v: list[tuple[int, ...]] = [(i,) for i in range(9)]
        tris: list[tuple[int, ...]] = []
        for i in range(3):
            for j in range(3):
                a = i * 3 + j
                b = i * 3 + (j + 1) % 3
                c = ((i + 1) % 3) * 3 + j
                d = ((i + 1) % 3) * 3 + (j + 1) % 3
                tris += [(a, b, d), (a, d, c)]
        es: list[tuple[int, ...]] = sorted(
            {
                tuple(sorted(e2))
                for t2 in tris
                for e2 in [(t2[0], t2[1]), (t2[1], t2[2]), (t2[0], t2[2])]
            }
        )
        return [v, es, tris]
    if kind == "circle":
        circ: list[list[tuple[int, ...]]] = [[(0,), (1,), (2,)], [(0, 1), (1, 2), (0, 2)]]
        return circ
    # figure-8: two triangles sharing a vertex
    fig8: list[list[tuple[int, ...]]] = [
        [(0,), (1,), (2,), (3,), (4,)],
        [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)],
    ]
    return fig8


def bench_simp_betti(seed: int = _SEED) -> dict[str, float]:
    ok = 0
    for kind, want in [
        ("sphere", [1, 0, 1]),
        ("torus", [1, 2, 1]),
        ("circle", [1, 1]),
        ("fig8", [1, 2]),
    ]:
        ok += int(betti(_complex(kind)) == want)
    return {"synthetic_betti_exact": float(ok == 4), "synthetic_betti_partial": ok / 4}
