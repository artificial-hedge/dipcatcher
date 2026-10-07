"""SDF sphere-tracing raymarcher (wave 293) (SYNTHETIC).

March p += d*sdf(p) until |sdf|<eps; hit distance vs analytic
ray-sphere quadratic solution on two spheres + one miss.
"""

import numpy as np

_SEED = 20261231 + 843


def sphere_sdf(c: np.ndarray, r: float, p: np.ndarray) -> float:
    return float(np.linalg.norm(p - c) - r)


def raymarch(sdf, ro: np.ndarray, rd: np.ndarray, tmax: float = 50.0, eps: float = 1e-4) -> float:
    t = 0.0
    for _ in range(10000):
        d = sdf(ro + rd * t)
        if d < eps:
            return t
        t += d
        if t > tmax:
            return -1.0
    return -1.0


def _analytic(ro: np.ndarray, rd: np.ndarray, c: np.ndarray, r: float) -> float:
    oc = ro - c
    b = float(oc @ rd)
    cc = float(oc @ oc - r * r)
    disc = b * b - cc
    if disc < 0:
        return -1.0
    t = -b - np.sqrt(disc)
    return float(t) if t > 0 else -1.0


def bench_sdf_raymarch(seed: int = _SEED) -> dict[str, float]:
    c1 = np.array([0, 0, 5.0])
    ro = np.array([0, 0, 0.0])
    rd = np.array([0, 0, 1.0])
    hit = raymarch(lambda p: sphere_sdf(c1, 1.0, p), ro, rd)
    ok = int(abs(hit - 4.0) < 1e-3 and abs(_analytic(ro, rd, c1, 1.0) - 4.0) < 1e-9)
    # off-axis ray
    ro2 = np.array([1.5, 0, 0.0])
    rd2 = np.array([-0.1, 0, 1.0])
    rd2 /= np.linalg.norm(rd2)
    hit2 = raymarch(lambda p: sphere_sdf(c1, 1.0, p), ro2, rd2)
    exp2 = _analytic(ro2, rd2, c1, 1.0)
    ok += int(hit2 > 0 and abs(hit2 - exp2) < 5e-3)
    # miss
    rd3 = np.array([1, 0, 0.0])
    ok += int(
        raymarch(lambda p: sphere_sdf(c1, 1.0, p), ro, rd3) < 0 and _analytic(ro, rd3, c1, 1.0) < 0
    )
    # closer of two spheres = min sdf
    hit4 = raymarch(
        lambda p: min(sphere_sdf(c1, 1.0, p), sphere_sdf(np.array([0, 0, 3.0]), 0.5, p)), ro, rd
    )
    ok += int(abs(hit4 - 2.5) < 1e-3)
    return {"synthetic_raymarch": float(ok == 4)}
