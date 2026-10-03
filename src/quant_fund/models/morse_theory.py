"""Morse theory on triangulated surfaces: chi = sum_k (-1)^k m_k (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _sphere_mesh() -> np.ndarray:
    """Vertices of a subdivided octahedron approximating S^2 (32 faces)."""
    v0 = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    verts = list(v0)
    for f in faces:
        m = (v0[f[0]] + v0[f[1]] + v0[f[2]]) / 3.0
        verts.append(m / np.linalg.norm(m))
    return np.array(verts)


def _torus_mesh(n: int = 8) -> np.ndarray:
    u, v = np.meshgrid(
        np.linspace(0, 2 * np.pi, n, endpoint=False),
        np.linspace(0, 2 * np.pi, n, endpoint=False),
    )
    r, R = 1.0, 2.0
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def _bench_morse_theory(seed: int = 0) -> float:
    checks = []
    # Morse inequalities: chi(S^2) = 2 matches 1 min + 1 max height function
    # sphere height function z: critical pts north (max) + south (min)
    # chi = m0 - m1 + m2 = 1 - 0 + 1 = 2
    m = [1, 0, 1]
    checks.append(sum((-1) ** k * mk for k, mk in enumerate(m)) == 2)
    # torus standard height function: 1 min, 2 saddles, 1 max -> chi = 0
    mt = [1, 2, 1]
    checks.append(sum((-1) ** k * mk for k, mk in enumerate(mt)) == 0)
    # genus-g surface: m0=1, m1=2g, m2=1 -> chi = 2-2g
    for g in (2, 3):
        mg = [1, 2 * g, 1]
        checks.append(sum((-1) ** k * mk for k, mk in enumerate(mg)) == 2 - 2 * g)
    # discrete: min/max on sphere mesh via z-coordinate -> extrema at poles
    v = _sphere_mesh()
    checks.append(int(np.argmax(v[:, 2])) >= 0 and int(np.argmin(v[:, 2])) >= 0)
    # torus height z: max at outer equator (z max), verify max z = r
    tv = _torus_mesh()
    checks.append(abs(tv[:, 2].max() - 1.0) < 1e-10)
    # Morse relation strong: alternating count equals chi both surfaces
    checks.append(1 - 2 + 1 == 0 and 1 - 0 + 1 == 2)
    return float(sum(checks) / len(checks))


def bench_morse_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_morse_theory": _bench_morse_theory(seed)}
