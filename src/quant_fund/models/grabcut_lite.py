"""GrabCut-lite: 2-component Gaussian EM foreground/background segmentation
iterating inside a bounding box.

SYNTHETIC bench only.
"""

import numpy as np

_SEED = 20261231 + 912


def _fit_gmm2(x: np.ndarray, iters: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.quantile(x, [0.3, 0.7])
    mu = np.array([lo, hi])
    var = np.array([x.var() + 1e-3, x.var() + 1e-3])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = np.stack(
            [
                np.log(w[k]) - 0.5 * np.log(var[k]) - 0.5 * (x - mu[k]) ** 2 / var[k]
                for k in range(2)
            ]
        )
        ll -= ll.max(axis=0)
        r = np.exp(ll)
        r /= r.sum(axis=0)
        for k in range(2):
            w[k] = float(r[k].mean())
            mu[k] = float((r[k] * x).sum() / max(r[k].sum(), 1e-9))
            var[k] = float((r[k] * (x - mu[k]) ** 2).sum() / max(r[k].sum(), 1e-9)) + 1e-6
    return (
        np.asarray(w, dtype=np.float64),
        np.asarray(mu, dtype=np.float64),
        np.asarray(var, dtype=np.float64),
    )


def grabcut(img: np.ndarray, box: tuple[int, int, int, int], iters: int = 4) -> np.ndarray:
    """box=(y0,x0,h,w) seeding foreground; returns prob-foreground map."""
    img = np.asarray(img, dtype=np.float64)
    y0, x0, h, w = box
    fg_seed = img[y0 : y0 + h, x0 : x0 + w].ravel()
    mask = np.ones(img.shape, dtype=bool)
    mask[y0 : y0 + h, x0 : x0 + w] = False
    bg_seed = img[mask]
    prob = np.zeros(img.shape)
    mu_f, mu_b = fg_seed.mean(), bg_seed.mean()
    for _ in range(iters):
        # fixed seeds for component params (lite: no graph-cut smoothing)
        _, mf, vf = _fit_gmm2(fg_seed, 6)
        _, mb, vb = _fit_gmm2(bg_seed, 6)
        kf = int(np.argmax(mf)) if mu_f > mu_b else int(np.argmin(mf))
        kb = int(np.argmax(mb)) if mu_b > mu_f else int(np.argmin(mb))
        lf = -0.5 * np.log(vf[kf]) - 0.5 * (img - mf[kf]) ** 2 / vf[kf]
        lb = -0.5 * np.log(vb[kb]) - 0.5 * (img - mb[kb]) ** 2 / vb[kb]
        d = lf - lb
        prob = 1.0 / (1.0 + np.exp(-d / (np.abs(d).max() + 1e-9) * 6.0))
        fg_seed = img[prob > 0.7]
        bg_seed = img[prob < 0.3]
        if fg_seed.size < 10 or bg_seed.size < 10:
            break
    return np.asarray(prob, dtype=np.float64)


def bench_grabcut_lite(seed: int = _SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    score = 0.0
    n = 60
    img = rng.normal(0.2, 0.05, (n, n))
    truth = np.zeros((n, n), dtype=bool)
    truth[15:45, 15:45] = True
    img[truth] = rng.normal(0.85, 0.05, int(truth.sum()))
    prob = grabcut(img, (17, 17, 26, 26), iters=5)
    pred = prob > 0.5
    iou = float((pred & truth).sum() / (pred | truth).sum())
    score += 1.0 if iou > 0.7 else 0.0
    # prob map in [0,1] and seeds retained
    score += 1.0 if prob.min() >= 0 and prob.max() <= 1 and prob[30, 30] > 0.8 else 0.0
    # swapped contrast (dark fg on bright bg) also works
    img2 = rng.normal(0.85, 0.05, (n, n))
    truth2 = np.zeros((n, n), dtype=bool)
    truth2[10:30, 30:55] = True
    img2[truth2] = rng.normal(0.15, 0.05, int(truth2.sum()))
    p2 = grabcut(img2, (12, 32, 16, 21), iters=5)
    iou2 = float(((p2 > 0.5) & truth2).sum() / ((p2 > 0.5) | truth2).sum())
    score += 1.0 if iou2 > 0.7 else 0.0
    # uniform image → degenerate (no confident fg), handled without crash
    p3 = grabcut(np.full((n, n), 0.5) + rng.normal(0, 0.01, (n, n)), (15, 15, 30, 30))
    score += 1.0 if np.all(np.isfinite(p3)) else 0.0
    return {"synthetic_grabcut_lite": score / 4.0}
