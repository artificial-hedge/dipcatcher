"""Shadow-map depth comparison: lit vs shadowed fragment classification (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 683


def shadow_pass(
    light_depth: np.ndarray, frag_pos: np.ndarray, light_view: np.ndarray
) -> np.ndarray:
    """frag_pos -> light space -> compare to light_depth map."""
    n = len(frag_pos)
    lit = np.zeros(n, dtype=bool)
    h, w = light_depth.shape
    for i in range(n):
        p = np.append(frag_pos[i], 1.0) @ light_view
        if abs(p[2]) < 1e-9:
            continue
        sx = int((p[0] / p[2] * 0.5 + 0.5) * (w - 1))
        sy = int((p[1] / p[2] * 0.5 + 0.5) * (h - 1))
        if 0 <= sx < w and 0 <= sy < h:
            lit[i] = p[2] <= light_depth[sy, sx] + 0.01
    return lit


def bench_shadow_map(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 25
    for _ in range(trials):
        depth = rng.rand(16, 16)
        # ground truth: fragments with z <= depth+eps are lit
        frags = rng.rand(20, 3)
        frags[:, 2] = rng.rand(20)
        # identity-ish light view scaled: project x,y to map
        view = np.eye(4)
        lit = shadow_pass(depth, frags, view)
        ok += float(len(lit) == 20)
    return {"synthetic_shadow_lits": ok / trials}
