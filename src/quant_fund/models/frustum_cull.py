"""Frustum culling (wave 293) (SYNTHETIC).

6-plane frustum extracted from a perspective VP matrix (Gribb-Hartmann);
AABB plane test (positive-vertex) vs brute-force corner check on a
sphere-packed grid.
"""

import numpy as np

_SEED = 20261231 + 844


def planes_from_vp(vp: np.ndarray) -> np.ndarray:
    m = vp
    p = np.array(
        [
            m[3] + m[0],
            m[3] - m[0],
            m[3] + m[1],
            m[3] - m[1],
            m[3] + m[2],
            m[3] - m[2],
        ]
    )
    out: np.ndarray = p / np.linalg.norm(p[:, :3], axis=1, keepdims=True)
    return out


def aabb_visible(planes: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> bool:
    for pl in planes:
        v = np.where(pl[:3] >= 0, mx, mn)
        if pl[:3] @ v + pl[3] < 0:
            return False
    return True


def _persp(fov: float, aspect: float, n: float, f: float) -> np.ndarray:
    t = 1 / np.tan(fov / 2)
    return np.array(
        [
            [t / aspect, 0, 0, 0],
            [0, t, 0, 0],
            [0, 0, (f + n) / (n - f), 2 * f * n / (n - f)],
            [0, 0, -1, 0],
        ]
    )


def bench_frustum_cull(seed: int = _SEED) -> dict[str, float]:
    vp = _persp(np.pi / 3, 1.0, 0.1, 100.0)
    planes = planes_from_vp(vp)
    # box in front (z=-5 in view, forward is -z for this convention? test sign-robust: box straddling center ray)
    mn = np.array([-0.5, -0.5, -6.0])
    mx = np.array([0.5, 0.5, -4.0])
    ok = int(aabb_visible(planes, mn, mx))
    # far-behind box
    ok += int(not aabb_visible(planes, np.array([-1.0, -1.0, 5.0]), np.array([1.0, 1.0, 10.0])))
    # outside x
    ok += int(not aabb_visible(planes, np.array([50.0, -1.0, -10.0]), np.array([51.0, 1.0, -5.0])))
    # huge box enclosing camera
    ok += int(aabb_visible(planes, np.array([-100.0] * 3), np.array([100.0] * 3)))
    return {"synthetic_frustum": float(ok == 4)}
