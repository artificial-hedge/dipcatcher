"""Block-matching stereo disparity; recovers planted shift (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 703


def disparity(il: np.ndarray, ir: np.ndarray, max_d: int = 8, win: int = 5) -> np.ndarray:
    h, w = il.shape
    d = np.zeros((h, w), dtype=int)
    r = win // 2
    for y in range(r, h - r):
        for x in range(r + max_d, w - r):
            best_d, best_e = 0, np.inf
            for dd in range(max_d):
                patch_l = il[y - r : y + r + 1, x - r : x + r + 1]
                patch_r = ir[y - r : y + r + 1, x - dd - r : x - dd + r + 1]
                e = np.abs(patch_l - patch_r).sum()
                if e < best_e:
                    best_e, best_d = e, dd
            d[y, x] = best_d
    return d


def bench_stereo_disparity(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 10
    for _ in range(trials):
        img = rng.rand(24, 40)
        true_d = int(rng.randint(2, 6))
        il = img[:, :-true_d]
        ir = img[:, true_d:]
        d = disparity(il, ir, max_d=8)
        center = d[8:16, 14:22]
        ok += float(np.abs(np.median(center) - true_d) <= 1)
    return {"synthetic_disparity_recovers": ok / trials}
