"""Deferred shading pipeline (wave 293) (SYNTHETIC).

Geometry pass writes G-buffer (albedo, normal, world pos); lighting
pass shades each pixel once — result equals per-fragment forward
shading oracle, at 1 light-eval per pixel instead of per-fragment-per-light.
"""

import numpy as np

_SEED = 20261231 + 842


def gbuffer(pos: np.ndarray, nrm: np.ndarray, alb: np.ndarray) -> dict[str, np.ndarray]:
    return {"pos": pos.copy(), "nrm": nrm.copy(), "alb": alb.copy()}


def deferred_light(
    g: dict[str, np.ndarray], lights: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    out = np.zeros_like(g["alb"])
    n = g["nrm"] / np.linalg.norm(g["nrm"], axis=-1, keepdims=True)
    for lp, lc in lights:
        d = lp - g["pos"]
        dist2 = (d * d).sum(-1, keepdims=True)
        d = d / np.sqrt(dist2)
        lam = np.maximum((n * d).sum(-1, keepdims=True), 0.0)
        out += g["alb"] * lc * lam / np.maximum(dist2, 1e-6)
    return out


def bench_deferred_shade(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1, 1, (8, 8, 3))
    nrm = np.tile(np.array([0, 0, 1.0]), (8, 8, 1))
    alb = rng.uniform(0.1, 0.9, (8, 8, 3))
    g = gbuffer(px, nrm, alb)
    lights = [
        (np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.9, 0.8])),
        (np.array([-1.0, 0.0, 2.0]), np.array([0.2, 0.3, 0.5])),
    ]
    d = deferred_light(g, lights)
    # forward oracle: identical expression evaluated per pixel directly
    f = np.zeros_like(alb)
    for lp, lc in lights:
        dv = lp - px
        dist2 = (dv * dv).sum(-1, keepdims=True)
        lam = np.maximum((nrm * (dv / np.sqrt(dist2))).sum(-1, keepdims=True), 0.0)
        f += alb * lc * lam / np.maximum(dist2, 1e-6)
    return {"synthetic_deferred": float(np.abs(d - f).max() < 1e-10 and (d > 0).all())}
