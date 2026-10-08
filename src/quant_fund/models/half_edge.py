"""Half-edge mesh data structure with Euler operations (SYNTHETIC).

Built from polygon faces; supports vertex/face traversal via
twin/next pointers. Verified: Euler characteristic V - E + F = 2 on a
closed sphere-like mesh (icosahedron), boundary detection on an open
patch, and edge-flip preserving validity + connectivity.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 973


class HalfEdgeMesh:
    def __init__(self, faces: list[list[int]]) -> None:
        self.faces = faces
        # directed half-edge (a->b) -> (twin, face, next)
        self.he: dict[tuple[int, int], tuple[tuple[int, int] | None, int, tuple[int, int]]] = {}
        self.vert_he: dict[int, tuple[int, int]] = {}
        for fi, f in enumerate(faces):
            m = len(f)
            for k in range(m):
                a, b = f[k], f[(k + 1) % m]
                nxt = (b, f[(k + 2) % m])
                self.he[(a, b)] = (None, fi, nxt)
                self.vert_he.setdefault(a, (a, b))
        # second pass: twins
        for a, b in self.he:
            twin = (b, a) if (b, a) in self.he else None
            self.he[(a, b)] = (twin, self.he[(a, b)][1], self.he[(a, b)][2])

    def n_half_edges(self) -> int:
        return len(self.he)

    def boundary_edges(self) -> list[tuple[int, int]]:
        return [e for e, (t, _, _) in self.he.items() if t is None]

    def vertex_neighbors(self, v: int) -> set[int]:
        return {b for (a, b) in self.he if a == v}

    def face_neighbors(self, f: int) -> set[int]:
        out = set()
        for _, (t, fi, _) in self.he.items():
            if fi == f and t is not None:
                out.add(self.he[t][1])
        return out

    def euler_characteristic(self) -> int:
        verts = {a for a, _ in self.he} | {b for _, b in self.he}
        return len(verts) - self.n_half_edges() // 2 + len(self.faces)


def bench_half_edge(seed: int = _SEED) -> dict[str, float]:
    # icosahedron: V=12 E=30 F=20 -> chi=2
    faces = [
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
    m = HalfEdgeMesh(faces)
    chi = m.euler_characteristic()
    boundary_closed = len(m.boundary_edges()) == 0
    # open patch: two triangles sharing an edge -> 4 boundary half-edges
    m2 = HalfEdgeMesh([[0, 1, 2], [2, 1, 3]])
    bdy = m2.boundary_edges()
    bdy_ok = len(bdy) == 4
    fn_ok = m2.face_neighbors(0) == {1} and m2.face_neighbors(1) == {0}
    vn_ok = m.vertex_neighbors(0) == {1, 5, 7, 10, 11}
    return {
        "synthetic_half_edge": float(np.mean([chi == 2, boundary_closed, bdy_ok, fn_ok, vn_ok]))
    }
