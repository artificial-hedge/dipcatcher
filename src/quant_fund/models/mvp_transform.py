"""SYNTHETIC model-view-projection transform chain.

4x4 homogeneous matrices: T*R*S composition, inverse correctness,
perspective projection divides by w. Verified against direct formulas.
"""

from __future__ import annotations

import math
import random

Mat = list[list[float]]
Vec = tuple[float, float, float, float]


def _mul(a: Mat, b: Mat) -> Mat:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _mv(m: Mat, v: Vec) -> Vec:
    return tuple(sum(m[i][k] * v[k] for k in range(4)) for i in range(4))  # type: ignore[return-value]


def translate(x: float, y: float, z: float) -> Mat:
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def scale(s: float) -> Mat:
    return [[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]]


def rotz(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def persp(fov: float, near: float, far: float) -> Mat:
    f = 1 / math.tan(fov / 2)
    return [
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ]


def bench_mvp_transform(seed: int = 20261231 + 386) -> dict[str, float]:
    rng = random.Random(seed)
    comp = inv = proj = 0
    trials = 40
    for _ in range(trials):
        tx, ty, tz = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)
        s = rng.uniform(0.5, 3)
        a = rng.uniform(0, math.pi)
        T, S, R = translate(tx, ty, tz), scale(s), rotz(a)
        M = _mul(T, _mul(R, S))
        v = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2), 1.0)
        got = _mv(M, v)
        # manual: scale → rotate → translate
        mx, my, mz = v[0] * s, v[1] * s, v[2] * s
        rx = mx * math.cos(a) - my * math.sin(a)
        ry = mx * math.sin(a) + my * math.cos(a)
        exp = (rx + tx, ry + ty, mz + tz, 1.0)
        comp += int(all(abs(g - e) < 1e-6 for g, e in zip(got, exp, strict=True)))
        # inverse: T*R*S inverted via S^-1 R^-1 T^-1
        Mi = _mul(scale(1 / s), _mul(rotz(-a), translate(-tx, -ty, -tz)))
        back = _mv(Mi, got)
        inv += int(all(abs(b - o) < 1e-5 for b, o in zip(back, v, strict=True)))
        # perspective: point on -z axis maps w=-z, x' = f*x/z sign flip
        P = persp(1.0, 0.1, 100.0)
        pv = _mv(P, (0.5, 0.0, -2.0, 1.0))
        proj += int(abs(pv[3] - 2.0) < 1e-6 and pv[0] > 0)
    return {
        "synthetic_trs_composed": float(comp / trials),
        "synthetic_inverse_exact": float(inv / trials),
        "synthetic_perspective_w": float(proj / trials),
    }
