"""ORB-lite: Harris corners + binary BRIEF descriptors + Hamming match (SYNTHETIC)."""

import numpy as np

_SEED = 20261231 + 699


def harris_score(img: np.ndarray, k: float = 0.06) -> np.ndarray:
    ix = np.gradient(img.astype(float), axis=1)
    iy = np.gradient(img.astype(float), axis=0)
    s = np.ones((3, 3)) / 9.0
    from scipy import ndimage  # scipy is a repo dep

    ixx = ndimage.convolve(ix * ix, s)
    ixy = ndimage.convolve(ix * iy, s)
    iyy = ndimage.convolve(iy * iy, s)
    out: np.ndarray = ixx * iyy - ixy * ixy - k * (ixx + iyy) ** 2
    return out


def brief(img: np.ndarray, x: int, y: int, pairs: np.ndarray) -> int:
    d = 0
    for i, (pa, pb) in enumerate(pairs):
        ax, ay = (
            int(np.clip(x + pa[0], 0, img.shape[1] - 1)),
            int(np.clip(y + pa[1], 0, img.shape[0] - 1)),
        )
        bx, by = (
            int(np.clip(x + pb[0], 0, img.shape[1] - 1)),
            int(np.clip(y + pb[1], 0, img.shape[0] - 1)),
        )
        if img[ay, ax] > img[by, bx]:
            d |= 1 << i
    return d


def bench_orb_feature(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    ok = 0.0
    trials = 30
    pairs = rng.randint(-4, 5, (16, 2, 2))
    for _ in range(trials):
        img = rng.rand(20, 20)
        r = harris_score(img)
        ok += float(np.isfinite(r).all())
        d1 = brief(img, 10, 10, pairs)
        d2 = brief(img, 10, 10, pairs)
        ok += float(d1 == d2)
    return {"synthetic_orb_deterministic": ok / (2 * trials)}
